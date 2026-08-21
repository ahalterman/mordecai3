"""Pull en_core_web_trf's fine-tuned roberta-base out of spaCy into a HF model.

en_core_web_trf 3.8 is a spacy-curated-transformers RobertaTransformer whose
124M-parameter encoder is roberta-base *fine-tuned inside the spaCy pipeline*
on OntoNotes tagging/parsing/NER (config.cfg: frozen=false, name=roberta-base).
That fine-tuning is exactly the signal the encoder-scoping pilot found the
ranker's mention slot depends on, and it is the natural initialisation for a
place-specialised NER.

Curated-transformers fuses Q/K/V into one `mha.input` matrix; everything else
is a rename. `--parity` checks the port by comparing per-token vectors against
spaCy's own `._.tensor` on short documents (short so that spaCy's
WithStridedSpans window of 144 word pieces is not in play).
"""
import os
import sys

import numpy as np
import torch

REPO = "/home/andy/projects/mordecai3"
os.chdir(REPO)
sys.path.insert(0, REPO)
OUT = ("/tmp/claude-1000/-home-andy-projects-mordecai3/"
       "a19357d6-880a-4251-9cdd-97e2235f449c/scratchpad/ner/roberta_onto")


def curated_state_dict():
    import spacy
    nlp = spacy.load("en_core_web_trf")
    m = nlp.get_pipe("transformer").model
    pt = m.layers[0].layers[0].layers[1].layers[0]
    return pt.shims[0]._model.state_dict()


def to_hf(sd):
    from transformers import RobertaConfig, RobertaModel
    cfg = RobertaConfig(vocab_size=50265, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072,
                        max_position_embeddings=514, type_vocab_size=1,
                        layer_norm_eps=1e-5, pad_token_id=1)
    model = RobertaModel(cfg, add_pooling_layer=False)
    e = "curated_encoder.embeddings.inner."
    new = {
        "embeddings.word_embeddings.weight": sd[e + "word_embeddings.weight"],
        "embeddings.token_type_embeddings.weight": sd[e + "token_type_embeddings.weight"],
        "embeddings.position_embeddings.weight": sd[e + "position_embeddings.weight"],
        "embeddings.LayerNorm.weight": sd[e + "layer_norm.weight"],
        "embeddings.LayerNorm.bias": sd[e + "layer_norm.bias"],
    }
    for i in range(12):
        c = f"curated_encoder.layers.{i}."
        h = f"encoder.layer.{i}."
        qkv_w = sd[c + "mha.input.weight"].chunk(3, dim=0)
        qkv_b = sd[c + "mha.input.bias"].chunk(3, dim=0)
        for name, w, b in zip(["query", "key", "value"], qkv_w, qkv_b):
            new[h + f"attention.self.{name}.weight"] = w
            new[h + f"attention.self.{name}.bias"] = b
        new[h + "attention.output.dense.weight"] = sd[c + "mha.output.weight"]
        new[h + "attention.output.dense.bias"] = sd[c + "mha.output.bias"]
        new[h + "attention.output.LayerNorm.weight"] = sd[c + "attn_output_layernorm.weight"]
        new[h + "attention.output.LayerNorm.bias"] = sd[c + "attn_output_layernorm.bias"]
        new[h + "intermediate.dense.weight"] = sd[c + "ffn.intermediate.weight"]
        new[h + "intermediate.dense.bias"] = sd[c + "ffn.intermediate.bias"]
        new[h + "output.dense.weight"] = sd[c + "ffn.output.weight"]
        new[h + "output.dense.bias"] = sd[c + "ffn.output.bias"]
        new[h + "output.LayerNorm.weight"] = sd[c + "ffn_output_layernorm.weight"]
        new[h + "output.LayerNorm.bias"] = sd[c + "ffn_output_layernorm.bias"]
    missing, unexpected = model.load_state_dict(new, strict=False)
    missing = [k for k in missing if "position_ids" not in k]
    assert not missing and not unexpected, (missing, unexpected)
    return model


def parity(model, n_docs=6, max_tok=90):
    """Compare per-token vectors against spaCy's own on short documents."""
    import spacy
    from spacy.tokens import DocBin
    from transformers import AutoTokenizer
    from mordecai3.mordecai_utilities import spacy_doc_setup
    spacy_doc_setup()
    tok = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
    blank = spacy.blank("en")
    db = DocBin().from_disk("raw_data/spacyed/source_lgl.spacy")
    docs = [d for d in db.get_docs(blank.vocab) if 20 < len(d) <= max_tok][:n_docs]
    if not docs:
        db = DocBin().from_disk("raw_data/spacyed/source_gwn.spacy")
        docs = [d for d in db.get_docs(blank.vocab) if 20 < len(d) <= max_tok][:n_docs]
    model.eval()
    worst = 0.0
    for d in docs:
        words = [t.text for t in d]
        enc = tok(words, is_split_into_words=True, return_tensors="pt")
        with torch.no_grad():
            h = model(**enc).last_hidden_state[0]
        wid = enc.word_ids(0)
        mine = np.zeros((len(d), 768), dtype="float32")
        for w in range(len(d)):
            idx = [i for i, x in enumerate(wid) if x == w]
            mine[w] = h[idx].mean(0).numpy() if idx else 0.0
        theirs = np.vstack([t._.tensor for t in d])
        cos = (mine * theirs).sum(1) / (np.linalg.norm(mine, axis=1) *
                                        np.linalg.norm(theirs, axis=1) + 1e-9)
        worst = max(worst, float(np.abs(mine - theirs).max()))
        print(f"  doc len {len(d):3d}  mean cos {cos.mean():.5f}  "
              f"min cos {cos.min():.5f}  max|diff| {np.abs(mine - theirs).max():.4f}")
    return worst


if __name__ == "__main__":
    sd = curated_state_dict()
    model = to_hf(sd)
    print("parity against spaCy's own token tensors (short docs):")
    parity(model)
    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(OUT)
    print("saved to", OUT)
