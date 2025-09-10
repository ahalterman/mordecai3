
import spacy

#from .mordecai_utilities import spacy_doc_setup

# need so that spacy.load doesn't throw an error about missing factory, #33
#spacy_doc_setup()

def check_spacy_model() -> bool:
    """Check if the spacy model is available."""
    try:
        spacy.load("en_core_web_trf")
        return True
    except OSError:
        return False