import pandas as pd 
from collections import Counter

import spacy
from spacy.language import Language
from spacy.tokens import Token, Doc
from spacy.pipeline import Pipe
import numpy as np
import jsonlines
from tqdm import tqdm
import re
import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import elastic_utilities as es_util
from format_geoparsing_data import doc_to_ex_expanded
from torch_bert_placename_compare import ProductionData, embedding_compare
from roberta_qa import setup_qa, add_event_loc

from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Search, Q

try:
    Token.set_extension('tensor', default=False)
except ValueError:
    pass

try:
    @Language.component("token_tensors")
    def token_tensors(doc):
        chunk_len = len(doc._.trf_data.tensors[0][0])
        token_tensors = [[]]*len(doc)
        
        for n, i in enumerate(doc):
            wordpiece_num = doc._.trf_data.align[n]
            for d in wordpiece_num.dataXd:
                which_chunk = int(np.floor(d[0] / chunk_len))
                which_token = d[0] % chunk_len
                ## You can uncomment this to see that spaCy tokens are being aligned with the correct 
                ## wordpieces.
                #wordpiece = doc._.trf_data.wordpieces.strings[which_chunk][which_token]
                #print(n, i, wordpiece)
                token_tensors[n] = token_tensors[n] + [doc._.trf_data.tensors[0][which_chunk][which_token]]
        for n, d in enumerate(doc):
            if token_tensors[n]:
                d._.set('tensor', np.mean(np.vstack(token_tensors[n]), axis=0))
            else:
                d._.set('tensor',  np.zeros(doc._.trf_data.tensors[0].shape[-1]))
        return doc
except ValueError:
    pass


events = pd.read_csv("events.20210301073501.Release507.csv", sep="\t")

event_dict = events.to_dict('records')


#[i['Event Sentence'] for i in event_dict if i['Story ID'] == 52656797]
#The agreement was signed by the Minister of Labor and Minister of State for Investment Affairs, Dr. Maan Al-Qatamin, and the CEO of the 'MedLab Laboratories' group, as it is the only one accredited by the American College of Pathology 'CAP' and 'ISO 15189'

