"""Utilities associated with using SchNetPack"""

from schnetpack.atomistic import AtomisticModel
from schnetpack.md import AtomsConverter
from schnetpack.data import AtomsData
from ase.io.xyz import read_xyz
from io import StringIO
from tqdm import tqdm
import numpy as np
import shutil
import torch
import json
import os


def make_schnetpack_data(dataset, dbpath, properties,
                         xyz_col='xyz', conformers=None, overwrite=True):
    """Convert a Pandas dictionary to a SchNet database

    Args:
        dataset (pd.DataFrame): Dataset to convert
        dbpath (string): Path to database to be saved
        properties ([string]): List of properties to include in the dataset
        conformers (str): Name of column with conformers as xyz
        xyz_col (string): Name of the column with the XYZ data
        overwrite (True): Whether to overwrite the database
    """

    # If needed, delete the previous database
    if os.path.exists(dbpath) and overwrite:
        os.unlink(dbpath)

    # Convert all entries to ase.Atoms objects
    atoms = dataset[xyz_col].apply(lambda x: read_xyz(StringIO(x)).__next__())

    # Every column besides the training set will be a property
    prop_cols = set(properties).difference([xyz_col])
    property_list = [dict(zip(prop_cols, [np.atleast_1d(row[p]) for p in prop_cols])) for i, row in
                     dataset.iterrows()]

    # Add conformers to the property list, but it isn't a required property when loading entries
    if conformers is not None:
        for d, c in zip(property_list, dataset[conformers]):
            d['conformers'] = np.atleast_1d(c)

    # Initialize the object
    db = AtomsData(dbpath, required_properties=properties, conformers=conformers is not None)

    # Add every system to the db object
    db.add_systems(atoms, property_list)
    return db


def run_model(model, data, xyz_col, additional_cols=None, progbar=True):
    """Runs a SchNetPack model on the column of a dataframe containing XYZ files

    Args:
        model (AtomisticModel): Model to be evaluated
        data (DataFrame): Data to be evaluated
        xyz_col (string): Column containing the XYZ data
        additional_cols ([string]): Any other columns to add to the input (e.g., B3LYP results)
        progbar (boolean): Whether to display a progress bar
    Returns:
        (ndarray) Predictions from the model
    """

    # Get default value for additional_cols
    if additional_cols is None:
        additional_cols = []

    # Make the tool to convert ase.Atoms to SchNet inputs
    c = AtomsConverter()

    results = []
    for xyz, more_data in tqdm(list(zip(data[xyz_col], data[additional_cols].values)),
                               disable=not progbar, leave=False):
        # Convert the XYZ file to an ASE object
        atoms = next(read_xyz(StringIO(xyz)))

        # Generate it in the input format needed
        inputs = c.convert_atoms(atoms)

        # Add in the additional columns
        for i, col in enumerate(additional_cols):
            inputs[col] = torch.Tensor(np.expand_dims(more_data[i], 0))

        # Run it through the model
        outputs = model(inputs)

        # Get the value in numpy format
        results.append(np.squeeze(outputs['y'].cpu().data.numpy()))

    return np.array(results)


def load_model(name, *modifiers, arch_dir='.', weights_dir='.', device='cpu'):
    """Load in a certain model from the directory structure used in training scripts

    See the source code for how the path names to weights and architecture files are generated.
    I was about to copy-paste that the code here to show how the directories are joined, and
    it wasn't really more self-explanatory than the source code.

    Args:
        name (str): Name of model to be loaded
        modifiers ([string, int, etc]): Any additional qualifiers to the model (e.g., training set size)
        arch_dir (str): Path to the test directory containing the architecture
        weights_dir (str): Path to the test directory containing the weights file
        device (str): Name of device to which model should be loaded
    Returns:
        (AtomisticModel) Torch model
    """

    # Load in the model architecture
    model = torch.load(os.path.join(arch_dir, 'networks', name, 'architecture.pth'),
                       map_location=device)

    # Load in the best model, set it as the state of the architecture
    state = torch.load(os.path.join(weights_dir, 'networks', name,
                                    *map(str, modifiers), 'best_model'),
                       map_location=device)
    model.load_state_dict(state)
    return model


def save_model(output, out_dir, output_props=('g4mp2_0k',), overwrite=False, delta=None):
    """Save a model to disk in a form ready to be trained

    Args:
        output (AtomisticModel): Model to be saved
        out_dir (string): Path in which to save model training components
        output_props ([string]): List of output properties
        overwrite (bool): Whether to overwrite existing models
        delta (str): Baseline property (None if not delta)
    """

    # Get the output directory
    if os.path.isdir(out_dir):
        if overwrite:
            shutil.rmtree(out_dir)
        else:
            print('Model already saved. Skipping.')
            return
    os.makedirs(out_dir)

    # Save the model
    torch.save(output, os.path.join(out_dir, 'architecture.pth'))

    # If needed, save the atomrefs (used when setting the mean and std of training set)
    output_mods = output.output_modules if isinstance(output, AtomisticModel) else output[-1].output_modules
    output_mods = output_mods[-1] if isinstance(output_mods, torch.nn.Sequential) else output_mods
    if output_mods.atomref is not None:
        weights = output_mods.atomref.weight.detach().numpy()
        np.save(os.path.join(out_dir, 'atomref.npy'), weights)

    # Save the training details
    with open(os.path.join(out_dir, 'options.json'), 'w') as fp:
        json.dump({
            'output_props': list(output_props),
            'delta': delta
        }, fp)
