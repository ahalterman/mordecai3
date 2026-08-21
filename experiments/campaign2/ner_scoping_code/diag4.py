import os, sys, numpy as np, torch
REPO="/home/andy/projects/mordecai3"; os.chdir(REPO); sys.path.insert(0,REPO)
import spacy
from spacy.tokens import DocBin
from transformers import AutoTokenizer, RobertaModel
from mordecai3.mordecai_utilities import spacy_doc_setup
spacy_doc_setup()
blank=spacy.blank("en")
db=DocBin().from_disk("raw_data/spacyed/source_lgl.spacy")
docs=[x for x in db.get_docs(blank.vocab) if 20<len(x)<=100][:8]
OUT="/tmp/claude-1000/-home-andy-projects-mordecai3/a19357d6-880a-4251-9cdd-97e2235f449c/scratchpad/ner/roberta_onto"
model=RobertaModel.from_pretrained(OUT, add_pooling_layer=False).eval()
tok=AutoTokenizer.from_pretrained("roberta-base")
for d in docs:
    enc=tok(d.text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=512)
    om=enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad(): h=model(**enc).last_hidden_state[0].numpy()
    # map pieces to spaCy tokens by char overlap
    bytok=[[] for _ in range(len(d))]
    tstart=[t.idx for t in d]; tend=[t.idx+len(t.text) for t in d]
    j=0
    for i,(a,b) in enumerate(om):
        if b<=a: continue
        for k in range(len(d)):
            if a < tend[k] and b > tstart[k]:
                bytok[k].append(i); break
    mine=np.zeros((len(d),768),dtype='float32')
    for k in range(len(d)):
        if bytok[k]: mine[k]=h[bytok[k]].mean(0)
    theirs=np.vstack([t._.tensor for t in d])
    c=(mine*theirs).sum(1)/(np.linalg.norm(mine,axis=1)*np.linalg.norm(theirs,axis=1)+1e-9)
    print(f"len {len(d):3d} pieces {len(om):3d}  mean cos {c.mean():.5f} min {c.min():.5f} max|diff| {np.abs(mine-theirs).max():.4f}")
