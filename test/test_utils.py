from jcesr_ml.utils import compute_atomization_energy
from ase.io.xyz import read_xyz
from math import isclose
from io import StringIO
import pytest


@pytest.fixture
def acetaldehyde():
    return next(read_xyz(StringIO("""7
H4 C2 O1
C -0.002945 1.509914 0.008673
C 0.026083 0.003276 -0.037459
O 0.942288 -0.655070 -0.456826
H 0.922788 1.926342 -0.391466
H -0.862015 1.878525 -0.564795
H -0.150506 1.843934 1.042891
H -0.894430 -0.486434 0.357749""")))


def test_correct(acetaldehyde):
    baseline = 1.5
    form = compute_atomization_energy(acetaldehyde, baseline, 'b3lyp')
    assert isclose(baseline, 1.5)
    assert isclose(form, baseline - 4 * -0.500273 - 2 * -37.846772 - -75.064579)
