import os, sys, numpy as np, torch
REPO="/home/andy/projects/mordecai3"; os.chdir(REPO); sys.path.insert(0,REPO)
import spacy
from spacy.tokens import DocBin
from transformers import AutoTokenizer, RobertaModel
from mordecai3.mordecai_utilities import spacy_doc_setup
spacy_doc_setup()
blank=spacy.blank("en")
db=DocBin().from_disk("raw_data/spacyed/source_lgl.spacy")
d=[x for x in db.get_docs(blank.vocab) if 20<len(x)<=60][0]
stored=np.vstack([t._.tensor for t in d])
nlp=spacy.load("en_core_web_trf"); nlp.add_pipe("token_tensors")
fresh_doc=nlp(d.text)
fresh=np.vstack([t._.tensor for t in fresh_doc])
print("fresh vs stored: same n tokens?", len(fresh_doc)==len(d))
c=(fresh*stored).sum(1)/(np.linalg.norm(fresh,axis=1)*np.linalg.norm(stored,axis=1)+1e-9)
print("spaCy CPU-now vs DocBin: mean cos", c.mean(), "min", c.min(), "max|diff|", np.abs(fresh-stored).max())
OUT="/tmp/claude-1000/-home-andy-projects-mordecai3/a19357d6-880a-4251-9cdd-97e2235f449c/scratchpad/ner/roberta_onto"
model=RobertaModel.from_pretrained(OUT, add_pooling_layer=False).eval()
tok=AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
words=[t.text for t in fresh_doc]
enc=tok(words, is_split_into_words=True, return_tensors="pt")
with torch.no_grad(): h=model(**enc).last_hidden_state[0]
wid=enc.word_ids(0)
mine=np.zeros((len(fresh_doc),768),dtype='float32')
for w in range(len(fresh_doc)):
    idx=[i for i,x in enumerate(wid) if x==w]
    mine[w]=h[idx].mean(0).numpy() if idx else 0.
c2=(mine*fresh).sum(1)/(np.linalg.norm(mine,axis=1)*np.linalg.norm(fresh,axis=1)+1e-9)
print("my HF port vs spaCy CPU-now: mean cos", c2.mean(), "min", c2.min(), "max|diff|", np.abs(mine-fresh).max())
