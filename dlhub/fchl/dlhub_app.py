from jcesr_ml.utils import compute_atomization_energy
from ase.io.xyz import read_xyz
from qml.data import Compound
from typing import List
from io import StringIO
import pickle as pkl
import numpy as np
import gzip
import json

# Hard-coded representation size
max_size = 64

# Global storage for the model and fidelity
model = None
fidelity = None

def _load_model():
    global model, fidelity
    # Load the model in memory
    if model is None:
        with gzip.open('model.pkl.gz') as fp:
            model = pkl.load(fp)

    if fidelity is None:
        with open('fidelity.json') as fp:
            fidelity = json.load(fp)
        



def evaluate_molecules(molecules : List[str], b3lyp_energies: List[float]) -> List[float]:
    """Compute the atomization energy of molecules

    Args:
        molecules ([str]): XYZ-format molecular structures. Assumed to be
            fully-relaxed
        b3lyp_energies ([float]): Total energies of structures
    Returns:
        ([float]): Estimated G4MP2 atomization energies of molecules
    """
    _load_model()

    # Convert all of the molecules to the qml representation
    compnds = [Compound(StringIO(x)) for x in molecules]

    # Compute the atomization energy for each compound
    b3lyp_atom = [compute_atomization_energy(next(read_xyz(StringIO(x))), u0, 'b3lyp')
                  for x, u0 in zip(molecules, b3lyp_energies)]

    # Compute the representaiton for each compound
    def compute_rep(x):
        """Generates representation and returns the values"""
        x.generate_fchl_representation(max_size)
        return x.representation
    reps = np.array(list(map(compute_rep, compnds)))

    # Compute the delta between B3LYP and G4MP2
    delta = model.predict(reps)

    # Return the sum of the two
    return np.add(b3lyp_atom, delta)
