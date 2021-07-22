from ..geoparse import Geoparser
import pytest

import spacy

@pytest.fixture(scope='session', autouse=True)
def geo():
    return Geoparser()
