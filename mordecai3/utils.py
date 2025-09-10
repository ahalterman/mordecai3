
import spacy

def check_spacy_model() -> bool:
    """Check if the spacy model is available."""
    try:
        spacy.load("en_core_web_trf")
        return True
    except OSError:
        return False