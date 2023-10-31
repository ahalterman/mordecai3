import numpy as np
from spacy.language import Language
from spacy.tokens import Token

#def make_country_dict():
#    country = pd.read_csv("assets/wikipedia-iso-country-codes.txt")
#    country_dict = {i:n for n, i in enumerate(country['Alpha-3 code'].to_list())}
#    country_dict["CUW"] = len(country_dict)
#    country_dict["XKX"] = len(country_dict)
#    country_dict["SCG"] = len(country_dict)
#    country_dict["SSD"] = len(country_dict)
#    country_dict["BES"] = len(country_dict)
#    country_dict["NULL"] = len(country_dict)
#    country_dict["NA"] = len(country_dict)
#    return country_dict
#
#
#with open("assets/feature_code_dict.json", "r") as f:
#    feature_code_dict = json.load(f)
#

def spacy_doc_setup():
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