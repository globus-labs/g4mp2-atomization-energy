"""Utilities for generating coordinates from SMILES string"""
from openbabel import OBConformerSearch, OBForceField
from pybel import readstring, Molecule
import pybel


def generate_atomic_coordinates(smiles) -> str:
    """Attempt to further refine the molecular structure through a rotor search

    Code adapted from: http://forums.openbabel.org/OpenBabel-Conformer-Search-td4177357.html

    Args:
        smiles (string): Smiles string of molecule to be generated
    Returns:
        (string): XYZ coordinates of molecule
    """

    # Convert it to a OpenBabel molecule
    mol = readstring('smi', smiles)

    # Generate initial 3D coordinates
    mol.make3D()

    # Try to get a forcefield that works with this molecule
    ff = get_forcefield(mol)

    # initial cleanup before the weighted search
    ff.SteepestDescent(500, 1.0e-4)
    ff.WeightedRotorSearch(100, 20)
    ff.ConjugateGradients(500, 1.0e-6)
    ff.GetCoordinates(mol.OBMol)

    return mol.write("xyz")


def get_forcefield(mol: Molecule) -> OBForceField:
    """Given a molecule, get a suitable forcefield

    Args:
        mol (Molecule): Molecule to be initialized
    Returns:
        (OBForcefield): Forcefield initialized with the molecule
    """
    ff = pybel._forcefields["mmff94"]
    success = ff.Setup(mol.OBMol)
    if not success:
        ff = pybel._forcefields["uff"]
        success = ff.Setup(mol.OBMol)
        if not success:
            raise Exception('Forcefield setup failed')

    return ff


def generate_conformers(xyz, n=10, relax=False):
    """Generate conformers for a molecule

    Args:
        xyz (str): XYZ coordinates of molecule
        n (int): Maximum number of conformers to generate
        relax (bool): Whether to relax the structure
    Returns:
        [str]: List of conformers
    """

    # Parse the groundstate molecule
    mol = readstring('xyz', xyz)

    # If mol has no rotors, return just the molecule
    if mol.OBMol.NumRotors() == 0:
        return [xyz]

    # Initialize the search tool
    conf = OBConformerSearch()
    conf.Setup(mol.OBMol, n)

    # Run the search and output results
    conf.Search()
    conf.GetConformers(mol.OBMol)

    # Get the conformers as strings
    output = []
    for i in range(mol.OBMol.NumConformers()):
        mol.OBMol.SetConformer(i)

        # Relax if desired
        if relax:
            ff = get_forcefield(mol)
            ff.ConjugateGradients(500, 1.0e-6)
            ff.GetCoordinates(mol.OBMol)

        output.append(mol.write('xyz'))

    return output


def get_rmsd(xyz_a, xyz_b):
    """Generate the RMSD two molecules

    Args:
        xyz_a (string): Coordinates of one molecule
        xyz_b (string): Coordinates of a second molecule
    Return:
        (float) RMSD between the molecules
    """

    # Make the tool to match the molecules
    align = pybel.ob.OBAlign()
    align.SetRefMol(readstring("xyz", xyz_a).OBMol)
    align.SetTargetMol(readstring("xyz", xyz_b).OBMol)

    # Perform the alignment
    align.Align()
    return align.GetRMSD()
