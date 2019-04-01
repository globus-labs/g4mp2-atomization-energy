from ase.io import read, write
import sys, glob, os,shutil

def find_number_of_non_hydrogen_atoms(filename):
    count = 0
    f = read(filename)
    symbol_list = f.get_chemical_symbols()
    for item in symbol_list:
        if item == 'H':
            count = count + 1
    non_hydrogen = f.get_number_of_atoms() - count
    return (non_hydrogen)


for filename in glob.glob("*.xyz"):
    non_hydrogen = find_number_of_non_hydrogen_atoms(filename)
    print(filename, non_hydrogen)
    if non_hydrogen > 10:
        shutil.copy(filename, "qm11")

