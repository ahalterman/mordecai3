from spacy.tokens import Span
from transformers import pipeline

model_name = "deepset/roberta-base-squad2"

# a) Get predictions
def setup_qa():
    trf = pipeline('question-answering', model=model_name, tokenizer=model_name)
    return trf

def add_event_loc(orig_doc, res):
    loc_start = [i.i for i in orig_doc if i.idx == res['start']][0]
    loc_end = [i.i for i in orig_doc if len(i) + i.idx == res['end']][0]
    new_doc = orig_doc

    loc_ent = Span(new_doc, loc_start, loc_end+1, label="EVENT_LOC") # create a Span for the new entity
    
    loc_i = set([i.i for i in loc_ent])
    new_ents = []
    for e in new_doc.ents:
        e_locs = set([i.i for i in e])
        if not e_locs.intersection(loc_i):
            new_ents.append(e)
    
    new_doc.ents = list(new_ents) + [loc_ent]
    return new_doc



if __name__ == "__main__":
    trf = setup_qa()

    QA_input = {
        'question': 'Where was the meeting?',
        'context': 'German Chancellor Merkel and President Obama attended a summit in Berlin to discuss the protest in the Damascus suburb of Abya.'
    }
    trf(QA_input)