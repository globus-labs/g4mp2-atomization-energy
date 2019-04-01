from jcesr_ml.atom_environment import get_atomic_fragments_from_mol
import pytest


@pytest.fixture
def acetaldehyde():
    return """7
H4 C2 O1
C -0.002945 1.509914 0.008673
C 0.026083 0.003276 -0.037459
O 0.942288 -0.655070 -0.456826
H 0.922788 1.926342 -0.391466
H -0.862015 1.878525 -0.564795
H -0.150506 1.843934 1.042891
H -0.894430 -0.486434 0.357749"""


@pytest.fixture()
def pyridinamine():
    """http://www.chemspider.com/Chemical-Structure.23347315.html?rid=93a5734d-1d0b-4030-ad8e-556177a68815"""
    return """13
H4 C5 N2 F2
N -0.043769 1.369972 -0.141917
C -0.013177 0.001013 -0.060144
N -1.171768 -0.662888 -0.037241
C -1.129877 -1.978978 -0.012254
F -2.313801 -2.588614 0.008415
C 0.018618 -2.755319 -0.000335
C 1.232652 -2.064684 -0.012472
C 1.210033 -0.689467 -0.043303
F 2.345348 0.032645 -0.048226
H 0.788712 1.860866 0.140758
H -0.920794 1.795822 0.113493
H -0.036935 -3.834759 0.023290
H 2.184060 -2.583669 0.002193"""


def test_ace(acetaldehyde):
    result = get_atomic_fragments_from_mol(acetaldehyde)
    assert len(result) == 7
    assert result[0] == { # 3 single bonded H's, 1 single bond C
        'type': 'C',
        'bonds': [(3, 1, 'H', False, 'none'),
                  (1, 1, 'C', False, 'none')]
    }


def test_pyri(pyridinamine):
    result = get_atomic_fragments_from_mol(pyridinamine)
    assert len(result) == 13
    assert result[2] == {  # The N in the ring
        'type': 'N',
        'bonds': [(2, 2, 'C', False, '')]
    }
