"""Utilities to describe the local environment of atoms"""

from openbabel import (OBConversion, OBAtomBondIter, OBMol,
                       OBMolAtomIter, OBElementTable, OBRingTyper)
from collections import Counter

# Make the atom type converter
converter = OBConversion()
converter.SetInAndOutFormats("xyz", "smiles")


# Make the periodic table
elem_table = OBElementTable()


def get_atomic_fragments_from_mol(xyz):
    """Get the fragments around each atom

    Args:
        xyz (string): XYZ file for the
    """

    # Read in the molecule
    mol = OBMol()
    converter.ReadString(mol, xyz)

    # Determine ring types
    OBRingTyper().AssignTypes(mol)

    # For each atom, get its local environment
    return [get_atomic_environment(atom) for atom in OBMolAtomIter(mol)]


def get_atomic_environment(atom):
    """Get the environment of a single atom

    Args:
        atom (OBAtom): Atom in question
    Returns:
        (dict): Data about the atom's local environment
             type (string): Type of this atom
             bonds (tuple): Description of bond types:
                0 (count): Number of times this bond occurs
                1 (int): Order of bonds
                2 (string): Identity of neighbor
                3 (bool): Whether bond is aromatic
                4 (string): Type of the ring
    """

    # Get the type this atom
    my_type = elem_table.GetSymbol(atom.GetAtomicNum())

    # Get data on all of the bonds
    bonds = []
    for bond in OBAtomBondIter(atom):
        # Get information about the bond
        o_type = bond.GetEndAtom().GetAtomicNum() if bond.GetBeginAtomIdx() == atom.GetIdx() \
            else bond.GetBeginAtom().GetAtomicNum()
        o_type = elem_table.GetSymbol(o_type)
        order = bond.GetBondOrder()
        is_aro = bond.IsAromatic()

        # Get information about the
        ring = bond.FindSmallestRing()
        ring_type = 'none' if ring is None else ring.GetType()

        # Combine into a tuple
        bonds.append((order, o_type, is_aro, ring_type))

    # Count the types of bonds
    bonds_counter = Counter(bonds)
    bonds_counted = [(v,) + k for k, v in bonds_counter.items()]

    return {
        'type': my_type,
        'bonds': bonds_counted
    }
