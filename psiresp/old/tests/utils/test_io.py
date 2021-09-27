import pytest
import os
import numpy as np
from numpy.testing import assert_equal

from psiresp.mixins import IOMixin
from psiresp.utils import io as ioutils


class IOTest(IOMixin):
    @ioutils.datafile
    def compute_array_arange_12(self):
        raise FileNotFoundError()

    @ioutils.datafile("test.xyz")
    def get_text(self):
        return "🐈"

    @ioutils.datafile("mishmash.argle")
    def invalid_format(self):
        return


@pytest.fixture()
def instance_in():
    return IOTest(load_input=True)


@pytest.fixture()
def instance_out():
    return IOTest(save_output=True)


def test_datafile_name_completion(tmpdir, instance_in):
    array = np.arange(12)
    with tmpdir.as_cwd():
        filename = "array_arange_12.dat"
        np.savetxt(filename, array)
        assert_equal(instance_in.compute_array_arange_12(), array)


def test_load_text(tmpdir, instance_in):
    cat = "🐈"
    with tmpdir.as_cwd():
        filename = "test.xyz"
        with open(filename, "w") as f:
            f.write(cat)
        assert instance_in.get_text() == cat


def test_invalid_format(instance_in):
    with pytest.raises(ValueError, match="Can't find loader for argle file"):
        ioutils.load_data("mishmash.argle")


def test_invalid_format_noerr(instance_in):
    assert instance_in.invalid_format() is None


def test_save_text(tmpdir, instance_out):
    with tmpdir.as_cwd():
        assert not os.path.exists(os.path.abspath("test.xyz"))
        cat = instance_out.get_text()
        with open("test.xyz", "r") as f:
            assert f.read() == cat
