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

            # Get the hidden states - keep all chunks!
            hidden_states = trf_data.tensors[0]  # Shape: (num_chunks, wordpieces_per_chunk, 768)

            # Flatten the hidden states to get all wordpieces in one array
            num_chunks, wordpieces_per_chunk, embedding_dim = hidden_states.shape
            flattened_hidden_states = hidden_states.reshape(-1, embedding_dim)  # Shape: (total_wordpieces, 768)

            # Use spaCy's alignment to map transformer tokens to spaCy tokens
            alignment = trf_data.align

            for token_idx, token in enumerate(doc):
                # Get the wordpiece indices that correspond to this spaCy token
                wordpiece_indices = alignment[token_idx].data

                # Filter out indices that are out of bounds (shouldn't happen now, but safety first)
                valid_indices = [idx for idx in wordpiece_indices if 0 <= idx < flattened_hidden_states.shape[0]]

                if len(valid_indices) > 0:
                    # Average the embeddings of all wordpieces for this token
                    token_embeddings = flattened_hidden_states[valid_indices]
                    averaged_embedding = np.mean(token_embeddings, axis=0)
                    token._.set('tensor', averaged_embedding)
                else:
                    # Fallback: zero vector
                    token._.set('tensor', np.zeros(embedding_dim))

            return doc
    except ValueError:
        pass