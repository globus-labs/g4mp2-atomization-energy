import os
import glob

# content of test_sample.py
def extract_scalars(lines):
    """
    Extract the various fields from the xyz file
    """
    
    n_atom = int(lines[0].split()[0])
    index =  lines[1].split()[1]
    homo  =  lines[1].split()[7]
    lumo  =  lines[1].split()[8]
    gap   =  lines[1].split()[9]
    zpe   =  lines[1].split()[11]
    U0    =  lines[1].split()[12]
    U     =  lines[1].split()[13]
    H     =  lines[1].split()[14]
    G     =  lines[1].split()[15]
    Cv    =  lines[1].split()[16]
            
    return {
            "index": index, 
            "n_atom": n_atom, 
            "homo": homo, 
            "lumo": lumo, 
            "bandgap": gap, 
            "zpe": zpe, 
            "u0": U0, 
            "u": U, 
            "h": H, 
            "g": G, 
            "cv": Cv
    }


def test_scalars():
    base_dir = '../data/input'
    path_g4mp2 = os.path.join(base_dir,'g4mp2', '*.xyz')
    files = glob.glob(path_g4mp2)
    
    for file in files:
        filename = (file.split('/')[-1])

        fl_g4mp2 = open(file, "r")
        l_g4mp2 = fl_g4mp2.readlines()

        fl_gdb9 = open(file.replace('g4mp2','gdb9'), "r")
        l_gdb9 = fl_gdb9.readlines()

        scalars_g4mp2 = extract_scalars(l_g4mp2[0:-10])
        scalars_gdb9 = extract_scalars(l_gdb9)

        assert(scalars_g4mp2 == scalars_gdb9)
