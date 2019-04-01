from jcesr_ml.utils import compute_atomization_energy
from ase.io.xyz import read_xyz
from qml.data import Compound
from typing import List
from io import StringIO
import pickle as pkl
import numpy as np
import gzip


# Load the model
with gzip.open('model.pkl.gz') as fp:
    model = pkl.load(fp)

# Hard-coded representation size
max_size = 64


def evaluate_molecules(molecules : List[str], b3lyp_energies: List[float]) -> List[float]:
    """Compute the atomization energy of molecules

    Args:
        molecules ([str]): XYZ-format molecular structures. Assumed to be
            fully-relaxed
        b3lyp_energies ([float]): B3LYP total energies of structures
    Returns:
        ([float]): Estimated G4MP2 atomization energies of molecules
    """

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

if __name__ == "__main__":
    print(evaluate_molecules(["""7
H4 C2 O1
C -0.002945 1.509914 0.008673
C 0.026083 0.003276 -0.037459
O 0.942288 -0.655070 -0.456826
H 0.922788 1.926342 -0.391466
H -0.862015 1.878525 -0.564795
H -0.150506 1.843934 1.042891
H -0.894430 -0.486434 0.357749"""], [0]))
