## Read in the BERT embedding for each place name
## and predict the country using pytorch
import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from pandas import read_csv
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(levelname)-8s %(name)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.WARN)


class ProductionData(Dataset):
    def __init__(self, es_data, max_choices=25, max_codes=50):
        self.max_choices = max_choices
        self.max_codes = max_codes
        self.country_dict = self._make_country_dict()
        self.feature_code_dict = self._make_feature_code_dict()
        self.placename_tensor = np.array([i['tensor'] for i in es_data]).astype(np.float32)
        self.doc_tensor = np.array([i['doc_tensor'] for i in es_data]).astype(np.float32)
        self.other_locs_tensor = np.array([i['locs_tensor'] for i in es_data]).astype(np.float32)
        self.feature_codes = self.create_feature_codes(es_data)
        self.country_codes = self.create_country_codes(es_data)
        self.gaz_info = self.create_gaz_features(es_data).astype(np.float32)
        #self.gaz_info[n][-1] = np.array([0] * 9)
        
    def __getitem__(self, index):
        return {"placename_tensor": self.placename_tensor[index],  
                "doc_tensor": self.doc_tensor[index], 
                "other_locs_tensor": self.other_locs_tensor[index],
                "feature_codes": self.feature_codes[index], 
                "country_codes": self.country_codes[index],
                "gaz_info": self.gaz_info[index]}
        
    
    def __len__ (self):
        return len(self.placename_tensor)

    # need to make this into a one-hot matrix, not a vector.
    # Inside the model, it should be a 3d one hot tensor, not binary.
    def create_feature_codes(self, es_data):
        all_feature_codes = []
        for ex in es_data:
            feature_code_raw = [i['feature_code'] for i in ex['es_choices'][0:self.max_choices]]
            feature_code_raw += ['NULL'] * (self.max_choices - len(feature_code_raw))
            feature_code_raw = feature_code_raw[0:self.max_choices]
            ## Pytorch embedding layers need indices, not one-hot
            feature_codes = [self.feature_code_dict[i] if i in self.feature_code_dict else len(self.feature_code_dict)+1 for i in feature_code_raw]
            # the last one is an other/not present category
            feature_codes[-1] = 53
            feature_codes = np.array(feature_codes, dtype="int")
            all_feature_codes.append(feature_codes)
        all_feature_codes = np.array(all_feature_codes).astype(np.int32)
        return all_feature_codes

    def create_country_codes(self, es_data):
        all_country_codes = []
        for ex in es_data:
            country_code_raw = [i['country_code3'] for i in ex['es_choices'][0:self.max_choices]]
            country_code_raw += ['NULL'] * (self.max_choices - len(country_code_raw))
            country_code_raw = country_code_raw[0:self.max_choices]
            country_codes = [self.country_dict[i] for i in country_code_raw]
            country_codes = np.array(country_codes, dtype="int")
            all_country_codes.append(country_codes)
        all_country_codes = np.array(all_country_codes).astype(np.int32)
        return all_country_codes

    def create_gaz_features(self, es_data):
        """
        Format all non-query/gazetteer-only features.

        Specifically, this includes edit distance features and adm1 and country overlap
        """
        edit_info = []
        for ex in es_data:
            alt_name_length = [i['alt_name_length'] for i in ex['es_choices'][0:self.max_choices]]
            alt_name_length += [99] * (self.max_choices - len(alt_name_length))
            min_dist = [i['min_dist'] for i in ex['es_choices'][0:self.max_choices]]
            min_dist += [99] * (self.max_choices - len(min_dist))
            max_dist = [i['max_dist'] for i in ex['es_choices'][0:self.max_choices]]
            max_dist += [99] * (self.max_choices - len(max_dist))
            avg_dist = [i['avg_dist'] for i in ex['es_choices'][0:self.max_choices]]
            avg_dist += [99] * (self.max_choices - len(avg_dist))
            ascii_dist = [i['ascii_dist'] for i in ex['es_choices'][0:self.max_choices]]
            ascii_dist += [99] * (self.max_choices - len(ascii_dist))
            adm1_overlap = [i['adm1_count'] for i in ex['es_choices'][0:self.max_choices]]
            adm1_overlap += [0] * (self.max_choices - len(adm1_overlap))
            country_overlap = [i['country_count'] for i in ex['es_choices'][0:self.max_choices]]
            country_overlap += [0] * (self.max_choices - len(country_overlap))
            in_adm1 = [i['admin1_parent_match'] for i in ex['es_choices'][0:self.max_choices]]
            in_adm1 += [0] * (self.max_choices - len(in_adm1))
            in_country = [i['country_code_parent_match'] for i in ex['es_choices'][0:self.max_choices]]
            in_country += [0] * (self.max_choices - len(in_country))
            #es_position = normalize(es_position)
            alt_name_length[-1] = -1 
            max_dist[-1] = -1
            avg_dist[-1] = -1 
            min_dist[-1] = -1 
            ascii_dist[-1] = -1 
            adm1_overlap[-1] = -1 
            country_overlap[-1] = -1
            in_adm1[-1] = -1
            in_country[-1] = -1
            ed = np.transpose(np.array([alt_name_length, max_dist, avg_dist, min_dist, 
                                        ascii_dist, adm1_overlap, country_overlap, in_adm1, in_country]))
            edit_info.append(ed)
        ed_stack = np.stack(edit_info)
        return ed_stack

    def _make_country_dict(self):
        pt = os.path.dirname(os.path.realpath(__file__))
        fn = os.path.join(pt, "assets", "wikipedia-iso-country-codes.txt")
        country = read_csv(fn)
        country_dict = {i:n for n, i in enumerate(country['Alpha-3 code'].to_list())}
        country_dict["CUW"] = len(country_dict)
        country_dict["XKX"] = len(country_dict)
        country_dict["SCG"] = len(country_dict)
        country_dict["SSD"] = len(country_dict)
        country_dict["BES"] = len(country_dict)
        country_dict["SXM"] = len(country_dict)
        country_dict["NULL"] = len(country_dict)
        country_dict["NA"] = len(country_dict)
        return country_dict

    def _make_feature_code_dict(self):
        pt = os.path.dirname(os.path.realpath(__file__))
        fn = os.path.join(pt, "assets", "feature_code_dict.json")
        with open(fn, "r") as f:
            feature_code_dict = json.load(f)
            return feature_code_dict


class TrainData(ProductionData):
    def __init__(self, es_data, max_choices=25, max_codes=50):
        super().__init__(es_data, max_choices, max_codes)
        self.labels, self.countries = self.create_labels(es_data)

    def __getitem__(self, index):
        return (self.labels[index],
                self.countries[index],
               {"placename_tensor": self.placename_tensor[index],  
                "doc_tensor": self.doc_tensor[index], 
                "other_locs_tensor": self.other_locs_tensor[index],
                "feature_codes": self.feature_codes[index], 
                "country_codes": self.country_codes[index],
                "gaz_info": self.gaz_info[index]}) 

    def create_labels(self, es_data):
        """Create an array with the location of the correct geonames entry"""
        all_labels = []
        all_countries = []
        for n, ex in enumerate(es_data):
            labels = np.zeros(self.max_choices)
            if np.sum(ex['correct']) == 0:
               labels[-1] = 1
               all_countries.append(self.country_dict["NULL"])
               # make an array of 0s of length 9 for gaz_info
            else:
                correct_num = np.where(np.array(ex['correct']))[0]
                labels[correct_num] = 1
                try:
                    cn = correct_num[0]
                    country_code = ex['es_choices'][cn]['country_code3']
                    all_countries.append(self.country_dict[country_code])

                except Exception as e:
                    print(e)
                    print("subsetting number: ", cn)
            ## HACK here: convert back to index, not one-hot
            labels = np.argmax(labels)
            all_labels.append(labels)
        all_labels = np.array(all_labels).astype(np.int32)
        all_countries = np.array(all_countries).astype(np.int32)
        return all_labels, all_countries



class geoparse_model(nn.Module):
    def __init__(self, device, 
                bert_size, 
                num_feature_codes, 
                country_size=24, 
                code_size=8, 
                dropout=0.2,
                mix_dim=24,
                country_pred=False):
        super(geoparse_model, self).__init__()
        self.device = device
        self.country_pred = country_pred
        # embeddings setup
        try:
            pt = os.path.dirname(os.path.realpath(__file__))
            fn = os.path.join(pt, "assets", "country_bert_768.npy")
        except NameError:
            fn = os.path.join("assets", "country_bert_768.npy")
        pretrained_country = np.load(fn)
        pretrained_country = torch.FloatTensor(pretrained_country)
        logger.debug("Pretrained country embedding dim: {}".format(pretrained_country.shape))
        self.code_emb = nn.Embedding(num_feature_codes, code_size)
        self.country_emb = nn.Embedding.from_pretrained(pretrained_country, freeze=True)
        self.country_embed_transform = nn.Linear(bert_size, country_size) 

        # text layers
        self.text_to_country = nn.Linear(bert_size, country_size) 
        self.context_to_country = nn.Linear(bert_size, country_size) 
        self.text_to_code = nn.Linear(bert_size, code_size) 

        # transformation layers
        gaz_feature_count = 13
        self.mix_linear = nn.Linear(gaz_feature_count, mix_dim) # number of comparisons --> mix 
        self.mix_linear2 = nn.Linear(mix_dim, mix_dim) # mix --> mix
        self.last_linear = nn.Linear(mix_dim, 1) # mix --> final
        self.mix_country = nn.Linear(pretrained_country.shape[0], pretrained_country.shape[0],
                                    bias=False)
        self.country_predict = nn.Linear(country_size, pretrained_country.shape[0],
                                    bias=False)
        
        # activations and similarities
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout) 
        self.similarity = nn.CosineSimilarity(dim=2)
        #self.similarity_country = nn.CosineSimilarity(dim=2)

    def forward(self, input):
        ## TODO: this whole forward pass can probably be done with fewer permutations and transposes.

        # Unpack the dictionary here. Sending the data to device within the forward
        # function isn't standard, but it makes the training loop code easier to follow.
        placename_tensor = input['placename_tensor'].to(self.device)
        other_locs_tensor = input['other_locs_tensor'].to(self.device)
        doc_tensor = input['doc_tensor'].to(self.device)
        feature_codes = input['feature_codes'].to(self.device)
        country_codes = input['country_codes'].to(self.device)
        gaz_info = input['gaz_info'].to(self.device)
        logger.debug("feature_code input shape:{}".format(feature_codes.shape))

        ###### Text info setup  ######
        ### Apply linear layers to each of the inputs (placename tensor, other locs tensor,
        ###  full document tensor)
        x = self.dropout(self.text_to_country(placename_tensor))
        x_code = self.dropout(self.text_to_code(placename_tensor))
        x_other_locs = self.dropout(self.context_to_country(other_locs_tensor))
        x_doc = self.dropout(self.context_to_country(doc_tensor))
        logger.debug(f"x shape: {x.shape}")

        ####### Gazetteer entries setup ######
        ### Set up all the comparisions
        fc = self.dropout(self.code_emb(feature_codes))
        cc = self.country_embed_transform(self.dropout(self.country_emb(country_codes)))
        # to match the stacked value below, rearrange so it's
        # (choices, batch_size, embed_size)
        fc = fc.permute(1, 0, 2)
        cc = cc.permute(1, 0, 2)
        logger.debug("cc shape: {}, fc shape: {}".format(cc.shape, fc.shape))

        # Next, turn x from (batch_size, choices) into (1, batch_size, choices)
        # so it can be broadcast into a similarity comparison with all the ys.
        x_stack_country = torch.unsqueeze(x, 0) 
        x_stack_code = torch.unsqueeze(x_code, 0) 
        x_stack_locs = torch.unsqueeze(x_other_locs, 0)
        x_stack_doc = torch.unsqueeze(x_doc, 0)
        logger.debug("x_stack_country shape: {}".format(x_stack_country.shape))
        # x_stack is (choices, batch_size, embed_size)
        
        ## Do the similiary comparisons
        cos_sim_country = self.similarity(x_stack_country, cc)
        cos_sim_code = self.similarity(x_stack_code, fc)
        cos_sim_other_locs = self.similarity(x_stack_locs, cc)
        cos_sim_doc = self.similarity(x_stack_doc, cc)
        logger.debug("cos_sim_country: {}, cos_sim_code: {}, cos_sim_doc: {}".format(cos_sim_country.shape, cos_sim_country.shape, cos_sim_doc.shape))
        # put all the similarities into the shape (batch size, choices)  
        cos_sim_country = torch.unsqueeze(torch.transpose(cos_sim_country, 0, 1), 2)
        cos_sim_code = torch.unsqueeze(torch.transpose(cos_sim_code, 0, 1), 2)
        cos_sim_other_locs = torch.unsqueeze(torch.transpose(cos_sim_other_locs, 0, 1), 2)
        cos_sim_doc = torch.unsqueeze(torch.transpose(cos_sim_doc, 0, 1), 2)
        logger.debug("cos_sim_country shape: {}".format(cos_sim_country.shape))
        both_sim = torch.cat((cos_sim_country, cos_sim_code, cos_sim_other_locs, cos_sim_doc, gaz_info), 2)
        # the gaz_info features are (batch_size, choices, 6), to make 10 in the last dim.
        logger.debug(f"concat shape: {both_sim.shape}")  # (batch_size, choices, 10)
        last = self.dropout(self.sigmoid(self.mix_linear(both_sim)))
        last = self.dropout(self.sigmoid(self.mix_linear2(last)))
        # after applying last_layer, the output is dim 1 per choice. Squeeze that to produce a 
        # final output that's (batch_size, choices).
        last = torch.squeeze(self.last_linear(last), dim=2)  
        logger.debug(f"last shape: {last.shape}")  # (batch_size, choices)
        # softmax over the choices dimension so each location's choices will sum to 1
        out = self.softmax(last) 
        logger.debug("out shape: {}".format(out.shape))  # should be (batch_size, choices)  (44, 25) 
        # try the country prediction again...
        if self.country_pred:
            country_pred = self.softmax(self.country_predict(self.sigmoid(x)))
            return out, country_pred
        else:
            return out



