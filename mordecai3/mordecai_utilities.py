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
            trf_data = doc._.trf_data
            
            # Get the hidden states (first tensor array)
            hidden_states = trf_data.tensors[0]  # Shape: (1, num_wordpieces, 768)
            
            # Remove batch dimension
            if len(hidden_states.shape) == 3:
                hidden_states = hidden_states[0]  # Shape: (num_wordpieces, 768)
            
            # Use spaCy's alignment to map transformer tokens to spaCy tokens
            alignment = trf_data.align
            
            for token_idx, token in enumerate(doc):
                # Get the wordpiece indices that correspond to this spaCy token
                wordpiece_indices = alignment[token_idx].data
                
                if len(wordpiece_indices) > 0:
                    # Average the embeddings of all wordpieces for this token
                    token_embeddings = hidden_states[wordpiece_indices]
                    averaged_embedding = np.mean(token_embeddings, axis=0)
                    token._.set('tensor', averaged_embedding)
                else:
                    # Fallback: zero vector
                    embedding_dim = hidden_states.shape[-1]
                    token._.set('tensor', np.zeros(embedding_dim))
            
            return doc
    except ValueError:
        pass