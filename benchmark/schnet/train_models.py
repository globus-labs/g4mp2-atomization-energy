"""Script for training ML models for the SchNet benchmark

Based off of: https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/scripts/schnetpack_qm9.py

Recreated so (1) that I can understand what it does, and (2) to work with our testing needs
"""

from schnetpack.data import AtomsLoader, StatisticsAccumulator
from schnetpack.atomistic import AtomisticModel
from schnetpack.train import hooks, Trainer
from schnetpack import nn
from datetime import datetime
from tempfile import mkdtemp
from math import ceil
import pickle as pkl
import numpy as np
import argparse
import logging
import shutil
import torch
import json
import sys
import os

# Hard-coded options
batch_size = 100
validation_frac = 0.1
chkp_interval = lambda n: ceil(100000 / n)
lr_patience = lambda n: ceil(25 * 100000 / n)
lr_start = 1e-4
lr_decay = 0.5
lr_min = 1e-6

if __name__ == "__main__":
    # Make the parser
    parser = argparse.ArgumentParser()

    # Add the arguments to the parser
    parser.add_argument('-d', '--device', help='Name of device to train on', type=str, default='cpu')
    parser.add_argument('-n', '--num-workers', help='Number of workers to use for data loaders',
                        type=int, default=4)
    parser.add_argument('--copy', help='Create a copy of the training SQL to a local folder',
                        type=str, default=None)
    parser.add_argument('--parallel', help='Run a multi-GPU model training', action='store_true')
    parser.add_argument('--cache', help='Load training sets into memory', action='store_true')
    parser.add_argument('name', help='Name of the model to be trained', type=str)
    parser.add_argument('size', help='Size of training set to use', type=int)

    # Parse the arguments
    args = parser.parse_args()

    # Configure the logger
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(args.name)

    logger.info('Training {} on a training set size of {} on {}'.format(
        args.name, args.size, args.device))

    # Open the training data
    with open('train_dataset.pkl', 'rb') as fp:
        full_train_data = pkl.load(fp)

    # Get the total training set size
    logger.info('Loaded the training data: Size = {}'.format(len(full_train_data)))

    # Adjust the size, if needed
    if args.size <= 0:
        args.size = len(full_train_data)

    # Make sure model hasn't finished training already
    net_dir = os.path.join('networks', args.name)  # Path to this network type
    work_dir = os.path.join(net_dir, str(args.size))  # Path to the specific run
    if os.path.isfile(os.path.join(work_dir, 'finished')):
        logger.info('Model has already finished training')
        exit()

    # Loading the model from disk
    model = torch.load(os.path.join(net_dir, 'architecture.pth'))
    with open(os.path.join(net_dir, 'options.json')) as fp:
        options = json.load(fp)
    logger.info('Loaded model from {}'.format(net_dir))

    # Load in the training data
    if not os.path.exists('splits'):
        os.mkdir('splits')
    split_file = os.path.join('splits', '{}.npz'.format(args.size))

    td = mkdtemp(dir=args.copy)
    try:
        # If desired, copy the SQL file to a temporary directory
        if args.copy is not None:
            new_sql_path = os.path.join(td, os.path.basename(full_train_data.dbpath))
            shutil.copyfile(full_train_data.dbpath, new_sql_path)
            full_train_data.dbpath = new_sql_path
            logger.info('Copied db path to: {}'.format(new_sql_path))

        # Get the training and validation size
        validation_size = int(args.size * validation_frac)
        train_size = args.size - validation_size
        train_data, valid_data, _ = full_train_data.create_splits(train_size, validation_size,
                                                                  split_file)

        # If desired, cache the data
        if args.cache:
            train_data = train_data.load_into_cache(args.num_workers)
            valid_data = valid_data.load_into_cache(args.num_workers)
            logger.info("Loaded training data into memory")

        # Make the data loaders
        train_load = AtomsLoader(train_data, shuffle=True,
                                 num_workers=0 if args.cache else args.num_workers,
                                 batch_size=batch_size)
        valid_load = AtomsLoader(valid_data,
                                 num_workers=0 if args.cache else args.num_workers,
                                 batch_size=batch_size)
        logger.info('Made training set loader. Workers={}, Train Size={}, '
                    'Validation Size={}, Batch Size={}'.format(
            train_load.num_workers, len(train_data), len(valid_data), batch_size))

        # Update the mean and standard deviation of the dataset
        atomref = None
        if os.path.isfile(os.path.join(net_dir, 'atomref.npy')):
            atomref = np.load(os.path.join(net_dir, 'atomref.npy'))
        if options.get('delta', None) is not None:
            # Compute the stats for the delta property
            delta_prop = options['delta']
            statistic = StatisticsAccumulator(batch=True)
            for d in train_load:
                d['delta_temp'] = d[options['output_props'][0]] - d[delta_prop]
                train_load._update_statistic(True, atomref, 'delta_temp', d, statistic)
            mean, std = statistic.get_statistics()
            mean = (mean,) # Make them a tuple
            std = (std,)
            logger.info('Computed statistics for delta-learning model')
        else:
            if atomref is not None:
                mean, std = zip(*[train_load.get_statistics(x, per_atom=True, atomrefs=ar[:, None])
                                  for x, ar in zip(options['output_props'], atomref.T)])
            else:
                mean, std = zip(*[train_load.get_statistics(x, per_atom=True)
                                  for x in options['output_props']])

        # Add them to the output module
        output_mods = model.output_modules if isinstance(model, AtomisticModel) else model[-1].output_modules
        output_mods.standardize = nn.base.ScaleShift(torch.cat(mean), torch.cat(std))
            
        # Move the model to the GPU
        model.to(args.device)

        # Make the loss function, optimizer, and hooks -> Add them to a trainer
        def loss(b, p):
            y_true = torch.stack(tuple(torch.squeeze(b[s]) for s in options['output_props']), 1)
            return torch.nn.functional.mse_loss(p['y'], y_true)

        #  Get only the fittable parameters
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        opt = torch.optim.Adam(trainable_params, lr=1e-4)

        hook_list = [hooks.CSVHook(work_dir, []),
                     hooks.ReduceLROnPlateauHook(opt, patience=lr_patience(args.size),
                                                 factor=lr_decay, min_lr=lr_min,
                                                 stop_after_min=True)]
        logger.info('Created loss, hooks, and optimizer')

        # Create the trainer
        if args.parallel and torch.cuda.device_count() > 1: 
            model = torch.nn.DataParallel(model)
            logger.info('Created a multi-GPU model. GPU count: {}'.format(torch.cuda.device_count()))

        trainer = Trainer(work_dir, model, loss, opt, train_load, valid_load,
                          hooks=hook_list, checkpoint_interval=chkp_interval(args.size))

        # Run the training
        logger.info('Started training')
        sys.stdout.flush()
        trainer.train(args.device)

        # Mark training as complete
        with open(os.path.join(work_dir, 'finished'), 'w') as fp:
            print(str(datetime.now()), file=fp)
        logger.info('Training finished')
    finally:
        shutil.rmtree(td)