from jcesr_ml.utils import compute_atomization_energy
from schnetpack.md import AtomsConverter
from ase.io.xyz import read_xyz
from typing import List
from io import StringIO
import pickle as pkl
import numpy as np
import torch
import os

# Set the batch size
batch_size = 2

# Load in the environment calculator
with open('train_dataset.pkl', 'rb') as fp:
    environment = pkl.load(fp).environment_provider

# Load in the model
model = torch.load(os.path.join('model', 'architecture.pth'), map_location='cpu')
state = torch.load(os.path.join('model', 'best_model'), map_location='cpu')
model.load_state_dict(state)


def evaluate_molecules(molecules: List[str], b3lyp_energies: List[float]) -> List[float]:
    """Compute the atomization energy of molecules

    Args:
        molecules ([str]): XYZ-format molecular structures. Assumed to be
            fully-relaxed
        b3lyp_energies ([float]): B3LYP total energies of structures
    Returns:
        ([float]): Estimated G4MP2 atomization energies of molecules
    """

    # Convert the molecules to atoms objects
    atoms = [next(read_xyz(StringIO(x))) for x in molecules]

    # Generate the local environment for each atom
    conv = AtomsConverter(environment)
    inputs = [conv.convert_atoms(atom) for atom in atoms]

    # Add the b3lyp_energies to each atom object
    for i, e in zip(inputs, b3lyp_energies):
        i['u0'] = torch.Tensor(np.expand_dims(e, 0))

    # Execute in batches
    results = []
    for i in inputs:
        outputs = model(i)
        results.append(np.squeeze(outputs['y'].cpu().data.numpy()))

    # Return atomization energy
    return [compute_atomization_energy(a, e, 'g4mp2') for a, e in zip(atoms, results)]


if __name__ == "__main__":
    # Get some data
    from jcesr_ml.benchmark import load_benchmark_data
    train_data, _ = load_benchmark_data()
    train_data = train_data.sample(100)

    # Evaluate the energies
    pred = evaluate_molecules(train_data['xyz'], train_data['u0'])
    assert np.subtract(pred, train_data['g4mp2_atom']).abs().mean() < 1e-3
