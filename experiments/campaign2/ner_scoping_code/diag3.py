import os, sys, numpy as np, torch
REPO="/home/andy/projects/mordecai3"; os.chdir(REPO); sys.path.insert(0,REPO)
import spacy
from transformers import AutoTokenizer, RobertaModel
nlp=spacy.load("en_core_web_trf")
m=nlp.get_pipe("transformer").model
pt=m.layers[0].layers[0].layers[1].layers[0]
mod=pt.shims[0]._model.eval()
print(type(mod))
import inspect
print(inspect.signature(mod.forward))
tok=AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
words="A Turkish policeman fatally shot Russia 's ambassador to Turkey on Monday .".split()
enc=tok(words, is_split_into_words=True, return_tensors="pt")
ids=enc.input_ids
print("ids", ids.tolist())
with torch.no_grad():
    out=mod(ids, attention_mask=None)
print(type(out))
print([x for x in dir(out) if not x.startswith('_')])
h1=out.last_hidden_layer_states
print(type(h1), getattr(h1,'shape',None))
h1=h1.detach().numpy() if hasattr(h1,'detach') else np.asarray(h1)
OUT="/tmp/claude-1000/-home-andy-projects-mordecai3/a19357d6-880a-4251-9cdd-97e2235f449c/scratchpad/ner/roberta_onto"
hf=RobertaModel.from_pretrained(OUT, add_pooling_layer=False).eval()
with torch.no_grad(): h2=hf(ids).last_hidden_state
a=np.asarray(h1).reshape(-1,768); b=h2[0].numpy()
print("shapes",a.shape,b.shape)
n=min(len(a),len(b))
c=(a[:n]*b[:n]).sum(1)/(np.linalg.norm(a[:n],axis=1)*np.linalg.norm(b[:n],axis=1)+1e-9)
print("curated-vs-HF mean cos", c.mean(), "min", c.min(), "max|diff|", np.abs(a[:n]-b[:n]).max())
# also print spaCy's own piece ids for these words
doc=nlp(" ".join(words))
