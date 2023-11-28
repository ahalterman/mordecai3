from spacy.language import Language
import numpy as np

def spacy_doc_setup():
    try:
        Token.set_extension('tensor', default=False)
    except ValueError:
        pass

    @Language.component("token_tensors")
    def token_tensors(doc):
        for n, token in enumerate(doc):
            ragged_tensor = doc._.trf_data.last_hidden_layer_state[n].data
            mean_tensor = np.mean(ragged_tensor, axis=0)
            token._.set('tensor', mean_tensor)
        return doc