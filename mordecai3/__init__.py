from .geoparse import Geoparser

__version__ = "3.0.0"





def run_streamlit_app():
    """Launch the Streamlit demo app."""
    import subprocess
    import sys
    from pathlib import Path
    
    app_path = Path(__file__).parent / "mordecai_streamlit.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


__all__ = [
    "Geoparser",
    "run_streamlit_app",
    "__version__",
]