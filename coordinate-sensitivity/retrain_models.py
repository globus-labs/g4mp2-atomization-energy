"""Script for re-training models on generated coordinates

Uses the weights of models that were trained on the B3LYP coordinates as a starting point
"""

from schnetpack.atomistic import AtomisticModel
from schnetpack.train import hooks, Trainer
from schnetpack.data import AtomsLoader, StatisticsAccumulator
from schnetpack import nn
from tempfile import TemporaryDirectory
from datetime import datetime
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
schnet_dir = os.path.join('..', 'benchmark', 'schnet')
size = 117232

if __name__ == "__main__":
    # Make the parser
    parser = argparse.ArgumentParser()

    # Add the arguments to the parser
    parser.add_argument('-d', '--device', help='Name of device to train on', type=str, default='cpu')
    parser.add_argument('-n', '--num-workers', help='Number of workers to use for data loaders',
                        type=int, default=4)
    parser.add_argument('--copy', help='Create a copy of the training SQL to a local folder',
                        type=str, default=None)
    parser.add_argument('name', help='Name of the model to be trained', type=str)
    parser.add_argument('jitter', help='Amount to perturb atomic coordinates', type=float)

    # Parse the arguments
    args = parser.parse_args()

    # Configure the logger
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(args.name)

    logger.info('Retraining {} with generated coordinates on {} with a jitter of {}'.format(
        args.name, args.device, args.jitter))

    # Open the training data with the generated coordinates
    with open(os.path.join('train_dataset.pkl'), 'rb') as fp:
        full_train_data = pkl.load(fp)

    # Get the total training set size
    logger.info('Loaded the training data: Size = {}'.format(len(full_train_data)))

    # Make sure model hasn't finished training already
    net_dir = os.path.join('networks', args.name)  # Path to this network type
    work_dir = os.path.join(net_dir, '{:.4f}'.format(args.jitter))
    if os.path.isfile(os.path.join(work_dir, 'finished')):
        logger.info('Model has already finished training')
        exit()

    # Loading the model from the SchNet directory
    model = torch.load(os.path.join(schnet_dir, net_dir, 'architecture.pth'),
                       map_location=args.device)
    with open(os.path.join(schnet_dir, net_dir, 'options.json')) as fp:
        options = json.load(fp)
    logger.info('Loaded model architecture from {}'.format(os.path.join(schnet_dir, net_dir)))

    # Load in the weights
    weights_dir = os.path.join(schnet_dir, net_dir, str(size))
    if not os.path.isfile(os.path.join(weights_dir, 'finished')):
        logger.error('Previous model not yet finished training')
        exit(1)
    weights_dict = torch.load(os.path.join(weights_dir, 'best_model'), map_location=args.device)
    model.load_state_dict(weights_dict)
    logger.info('Loaded model weights from {}'.format(weights_dir))

    # Make the split file
    if not os.path.exists('splits'):
        os.mkdir('splits')
    split_file = os.path.join('splits', '{}.npz'.format(size))

    with TemporaryDirectory(dir=args.copy) as td:
        # If desired, copy the SQL file to a temporary directory
        if args.copy is not None:
            new_sql_path = os.path.join(td, os.path.basename(full_train_data.dbpath))
            shutil.copyfile(full_train_data.dbpath, new_sql_path)
            full_train_data.dbpath = new_sql_path
            logger.info('Copied db path to: {}'.format(new_sql_path))

        # Get the training and validation size
        validation_size = int(size * validation_frac)
        train_size = size - validation_size
        
        train_data, valid_data, _ = full_train_data.create_splits(train_size, validation_size,
                                                                  split_file)

        # Set the jitter parameter for the training data, not the validation set
        if args.jitter > 0:
            train_data.rattle_atoms = args.jitter

        # Make the loaders
        train_load = AtomsLoader(train_data, shuffle=True,
                                 num_workers=args.num_workers, batch_size=batch_size)
        valid_load = AtomsLoader(valid_data, num_workers=args.num_workers, batch_size=batch_size)
        logger.info('Made training set loader. Workers={}, Train Size={}, '
                    'Validation Size={}, Batch Size={}'.format(
            args.num_workers, len(train_data), len(valid_data), batch_size))

        # Get the baseline statistics
        atomref = None
        if os.path.isfile(os.path.join(schnet_dir, net_dir, 'atomref.npy')):
            atomref = np.load(os.path.join(schnet_dir, net_dir, 'atomref.npy'))
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
        
        # move the model to the GPU
        model.to(args.device)

        # Make the loss function, optimizer, and hooks -> Add them to a trainer
        def loss(b, p):
            y_true = torch.stack(tuple(torch.squeeze(b[s]) for s in options['output_props']), 1)
            return torch.nn.functional.mse_loss(p['y'], y_true)

        #  Get only the fittable parameters
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        opt = torch.optim.Adam(trainable_params, lr=1e-4)

        hook_list = [hooks.CSVHook(work_dir, []),
                     hooks.ReduceLROnPlateauHook(opt, patience=lr_patience(len(train_data)),
                                                 factor=lr_decay, min_lr=lr_min,
                                                 stop_after_min=True)]
        logger.info('Created loss, hooks, and optimizer')
        trainer = Trainer(work_dir, model, loss, opt, train_load, valid_load,
                          hooks=hook_list, checkpoint_interval=chkp_interval(len(train_data)))

        # Run the training
        logger.info('Started training')
        sys.stdout.flush()
        trainer.train(args.device)

        # Mark training as complete
        with open(os.path.join(work_dir, 'finished'), 'w') as fp:
            print(str(datetime.now()), file=fp)
        logger.info('Training finished')
