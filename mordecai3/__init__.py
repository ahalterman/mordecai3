
import logging

from .geoparse import Geoparser

__version__ = "3.0.0"

# Adjust logging levels for urllib3 and elasticsearch to reduce verbosity
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.ERROR)