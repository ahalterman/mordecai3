import os, sys, numpy as np, torch
REPO="/home/andy/projects/mordecai3"; os.chdir(REPO); sys.path.insert(0,REPO)
sys.path.insert(0, os.path.dirname("/tmp/claude-1000/-home-andy-projects-mordecai3/a19357d6-880a-4251-9cdd-97e2235f449c/scratchpad/ner/"))
import spacy
from spacy.tokens import DocBin
from transformers import AutoTokenizer, RobertaModel
from mordecai3.mordecai_utilities import spacy_doc_setup
spacy_doc_setup()
OUT="/tmp/claude-1000/-home-andy-projects-mordecai3/a19357d6-880a-4251-9cdd-97e2235f449c/scratchpad/ner/roberta_onto"
model=RobertaModel.from_pretrained(OUT, add_pooling_layer=False).eval()
tok=AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
blank=spacy.blank("en")
db=DocBin().from_disk("raw_data/spacyed/source_lgl.spacy")
d=[x for x in db.get_docs(blank.vocab) if 20<len(x)<=60][0]
words=[t.text for t in d]
print(repr(d.text[:200]))
print(words[:20])
enc=tok(words, is_split_into_words=True, return_tensors="pt")
print("n pieces", enc.input_ids.shape)
with torch.no_grad(): h=model(**enc).last_hidden_state[0]
wid=enc.word_ids(0)
mine=np.zeros((len(d),768),dtype='float32')
for w in range(len(d)):
    idx=[i for i,x in enumerate(wid) if x==w]
    mine[w]=h[idx].mean(0).numpy() if idx else 0.
theirs=np.vstack([t._.tensor for t in d])
cos=(mine*theirs).sum(1)/(np.linalg.norm(mine,axis=1)*np.linalg.norm(theirs,axis=1)+1e-9)
for i,t in enumerate(d):
    print(f"{i:3d} {t.text!r:20s} ws={t.whitespace_!r:4s} cos={cos[i]:.4f}")
