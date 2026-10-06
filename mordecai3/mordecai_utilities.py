from contextlib import contextmanager

import numpy as np
from spacy.language import Language
from spacy.tokens import Token
from spacy.tokens import _serialize as _spacy_serialize


def _segment_means(data, lengths):
    """Mean of `data`'s rows within each consecutive segment given by `lengths`.

    `data` is (total_pieces, dim) and `lengths` is (n_tokens,) giving how many
    pieces belong to each token. Done with one cumulative sum rather than a
    slice-and-mean per token: on GPU that is the difference between one kernel
    launch per document and one per token, which is most of the cost of this
    pipeline component.

    The prefix sum runs in float64. In float32 the running total drifts over a
    long document and the per-token means come out ~1e-4 off the per-segment
    mean they are meant to reproduce; in float64 the round trip is exact to
    float32 precision.

    Works for either numpy or cupy input; returns the same array type and dtype.
    """
    module = type(data).__module__.split(".")[0]
    xp = np if module == "numpy" else __import__(module)

    lengths = xp.asarray(lengths)
    n_tokens = int(lengths.shape[0])
    if data.shape[0] == 0:
        return xp.zeros((n_tokens, data.shape[1]), dtype=data.dtype)

    offsets = xp.zeros(n_tokens + 1, dtype="int64")
    xp.cumsum(lengths, out=offsets[1:])
    # A leading zero row makes each segment sum a difference of prefix sums.
    csum = xp.zeros((data.shape[0] + 1, data.shape[1]), dtype="float64")
    xp.cumsum(data, axis=0, out=csum[1:], dtype="float64")
    sums = csum[offsets[1:]] - csum[offsets[:-1]]
    # Tokens with no pieces get a zero vector, matching the per-token fallback.
    denom = xp.maximum(lengths, 1).reshape(-1, 1)
    return (sums / denom).astype(data.dtype)


def spacy_doc_setup():
    try:
        Token.set_extension('tensor', default=False)
    except ValueError:
        pass

    try:
        @Language.component("token_tensors")
        def token_tensors(doc):
            trf_data = doc._.trf_data

            # Check if we're using the new curated transformers (spaCy 3.7+)
            if hasattr(trf_data, 'last_hidden_layer_state'):
                # New spaCy 3.7+ with curated transformers
                # Get the last hidden layer state - this is a Ragged tensor
                hidden_states = trf_data.last_hidden_layer_state
                data = getattr(hidden_states, 'data', hidden_states)
                lengths = getattr(hidden_states, 'lengths', None)
                if lengths is None:
                    return doc

                tensors = _segment_means(data, lengths)
                # One host transfer per doc. Everything downstream (np.vstack in
                # geoparse/train, pickling of training data) wants numpy, and
                # pulling each token's vector across separately was costing more
                # than the transformer forward pass itself.
                if hasattr(tensors, 'get'):
                    tensors = tensors.get()

                embedding_dim = data.shape[-1]
                zero = np.zeros(embedding_dim, dtype=tensors.dtype)
                for token_idx, token in enumerate(doc):
                    if token_idx < len(tensors):
                        token._.set('tensor', tensors[token_idx])
                    else:
                        # Fallback for tokens beyond the piece alignment
                        token._.set('tensor', zero)

            else:
                # Legacy spaCy 3.0-3.6 with spacy-transformers
                # This is your original code for older versions
                hidden_states = trf_data.tensors[0]
                num_chunks, wordpieces_per_chunk, embedding_dim = hidden_states.shape
                flattened_hidden_states = hidden_states.reshape(-1, embedding_dim)

                alignment = trf_data.align

                for token_idx, token in enumerate(doc):
                    wordpiece_indices = alignment[token_idx].data
                    valid_indices = [idx for idx in wordpiece_indices if 0 <= idx < flattened_hidden_states.shape[0]]

                    if len(valid_indices) > 0:
                        token_embeddings = flattened_hidden_states[valid_indices]
                        averaged_embedding = np.mean(token_embeddings, axis=0)
                        token._.set('tensor', averaged_embedding)
                    else:
                        token._.set('tensor', np.zeros(embedding_dim))

            return doc

    except ValueError:
        pass


@contextmanager
def fast_docbin_io():
    """Make DocBin.to_disk skip zlib compression inside this block.

    DocBin serializes through `zlib.compress(...)` at the default level. Our
    docs carry a 768-float tensor per token, which is essentially
    incompressible: on the training corpus zlib spends ~93s to make the cache
    3.3GB instead of 3.7GB. Writing at level 0 costs ~4s for the same data.

    Level 0 still emits a valid zlib stream, so files written in this block are
    read back by an unmodified `DocBin.from_disk` -- there is no new format and
    nothing else needs to know about this.

    Scoped to spaCy's serializer module rather than to `zlib` itself, so
    unrelated compression in the process is untouched. If a future spaCy stops
    routing through a module-level `zlib`, this quietly does nothing.
    """
    module = getattr(_spacy_serialize, "zlib", None)
    if module is None or not hasattr(module, "compress"):
        yield
        return

    class _StoreOnly:
        error = module.error
        decompress = staticmethod(module.decompress)

        @staticmethod
        def compress(data, level=0):
            return module.compress(data, 0)

    _spacy_serialize.zlib = _StoreOnly
    try:
        yield
    finally:
        _spacy_serialize.zlib = module
