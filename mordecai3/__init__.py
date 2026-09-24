
from .geoparse import Geoparser
from .exceptions import SpacyModelError, ElasticsearchConnectionError, GeonamesIndexError

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mordecai3")
except PackageNotFoundError:  # running from a checkout without installing
    __version__ = "unknown"





def run_streamlit_app():
    """Launch the Streamlit demo app."""
    import subprocess
    import sys
    from pathlib import Path
    
    app_path = Path(__file__).parent / "mordecai_streamlit.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


__all__ = [
    "Geoparser",
    "SpacyModelError",
    "ElasticsearchConnectionError",
    "GeonamesIndexError",
    "run_streamlit_app",
    "__version__",
]
