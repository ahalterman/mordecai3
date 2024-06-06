import numpy as np
from spacy.language import Language
from spacy.tokens import Token

    
def spacy_doc_setup():
    try:
        Token.set_extension('tensor', default=False)
    except ValueError:
        pass
    try:
        @Language.component("token_tensors")
        def token_tensors(doc):
            tensors = doc._.trf_data.last_hidden_layer_state
            for n, d in enumerate(doc):
                if tensors[n]:
                    d._.set('tensor', tensors[n].data)
                else:
                    d._.set('tensor',  np.zeros(tensors[0].shape[-1]))
            return doc
    except ValueError:
        pass