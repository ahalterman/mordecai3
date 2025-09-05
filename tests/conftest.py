import pytest

from mordecai3.geoparse import Geoparser


@pytest.fixture(scope='session', autouse=True)
def geo():
    return Geoparser()
