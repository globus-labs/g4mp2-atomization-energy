from jcesr_ml.coordinates import (generate_atomic_coordinates, get_rmsd,
                                  generate_conformers, get_forcefield)
from pybel import readstring
from math import isclose


def test_generate():
    # Make methane
    result = generate_atomic_coordinates('C')
    assert result.startswith("5")
    assert result.count("C") == 1
    assert result.count("H") == 4

    # Make ethanol
    result = generate_atomic_coordinates('CCO')
    assert result.startswith("9")
    assert result.count("O") == 1
    assert result.count("H") == 6


def test_conformers():
    methane = generate_atomic_coordinates('C')
    assert 1 == len(generate_conformers(methane))

    propanol = generate_atomic_coordinates('CCcO')
    unrelaxed = generate_conformers(propanol)
    assert 3 == len(set(unrelaxed))

    propanol = generate_atomic_coordinates('CCcO')
    relaxed = generate_conformers(propanol, relax=True)
    assert 3 == len(set(relaxed))

    assert min(get_forcefield(readstring('xyz', m)).Energy() for m in unrelaxed) > \
           min(get_forcefield(readstring('xyz', m)).Energy() for m in relaxed)


def test_rmsd():
    methane = generate_atomic_coordinates('C')
    assert isclose(get_rmsd(methane, methane), 0)


if __name__ == "__main__":
    print(generate_atomic_coordinates('CCcO'))
    print(generate_conformers(generate_atomic_coordinates('CCcO')))
