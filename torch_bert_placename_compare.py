## Read in the BERT embedding for each place name
## and predict the country using pytorch
import numpy as np
import random
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from error_utils import make_wandb_dict, evaluate_results

import pickle
import json
from pandas import read_csv

import logging
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.WARN)

#import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


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
        all_feature_codes = np.array(all_feature_codes).astype(np.long)
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
        all_country_codes = np.array(all_country_codes).astype(np.long)
        return all_country_codes

    def create_gaz_features(self, es_data):
        """
        Format all non-query/gazetteer-only features.

        Specifically, this includes edit distance features and adm1 and country overlap
        """
        edit_info = []
        for ex in es_data:
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
            ed = np.transpose(np.array([max_dist, avg_dist, min_dist, ascii_dist, adm1_overlap, country_overlap]))
            edit_info.append(ed)
        ed_stack = np.stack(edit_info)
        return ed_stack

    def _make_country_dict(self):
        country = read_csv("data/wikipedia-iso-country-codes.txt")
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
        with open("data/feature_code_dict.json", "r") as f:
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
        for ex in es_data:
            labels = np.zeros(self.max_choices)
            if np.sum(ex['correct']) == 0:
               labels[-1] = 1
               all_countries.append(self.country_dict["NULL"])
            else:
                correct_num = np.where(np.array(ex['correct']))[0]
                if correct_num[0] >= self.max_choices:
                    # TODO: make a better missing/NA prediction.
                    labels[-1] = 1
                    all_countries.append(self.country_dict["NULL"])
                else:
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
        all_labels = np.array(all_labels).astype(np.long)
        all_countries = np.array(all_countries).astype(np.long)
        return all_labels, all_countries



def binary_acc(y_pred, y_test):
    y_pred_tag = torch.argmax(y_pred, axis=1)
    #y_test_cat = torch.argmax(y_test, axis=1) 
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_pred.shape[0]
    acc = torch.round(acc * 100)
    return acc


class embedding_compare(nn.Module):
    def __init__(self, device, bert_size, num_feature_codes):
        super(embedding_compare, self).__init__()
        self.device = device
        # embeddings setup
        pretrained_country = np.load("data/country_bert_768.npy")
        pretrained_country = torch.FloatTensor(pretrained_country)
        logger.debug("Pretrained country embedding dim: {}".format(pretrained_country.shape))
        self.code_emb = nn.Embedding(num_feature_codes, 8)
        self.country_emb = nn.Embedding.from_pretrained(pretrained_country, freeze=True)
        self.country_embed_transform = nn.Linear(bert_size, 24) 

        # text layers
        self.text_to_country = nn.Linear(bert_size, 24) 
        self.context_to_country = nn.Linear(bert_size, 24) 
        self.text_to_code = nn.Linear(bert_size, 8) 

        # transformation layers
        self.mix_linear = nn.Linear(10, 10) # number of comparisons --> number of comparisons
        self.mix_linear2 = nn.Linear(10, 10) # number of comparisons --> number of comparisons
        self.last_linear = nn.Linear(10, 1) # number of comparisons --> final
        self.mix_country = nn.Linear(pretrained_country.shape[0], pretrained_country.shape[0],
                                    bias=False)
        self.country_predict = nn.Linear(bert_size, pretrained_country.shape[0],
                                    bias=False)
        
        # activations and similarities
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.2) 
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
        #logger.debug("cos_sim_code shape: {}".format(cos_sim_code.shape))
        #logger.debug("gaz_info info shape: {}".format(gaz_info.shape))
        #both_sim = torch.cat((cos_sim_country, cos_sim_code, cos_sim_other_locs, cos_sim_doc), 2)
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
        return out



def load_data(limit_es_results):
    if limit_es_results:
        path_mod = "pa_only"
    else:
        path_mod = "all_loc_types"
    with open(f'training/{path_mod}/es_formatted_prodigy.pkl', 'rb') as f:
        es_data_prod = pickle.load(f)
    
    with open(f'training/{path_mod}/es_formatted_tr.pkl', 'rb') as f:
        es_data_tr = pickle.load(f)
    
    with open(f'training/{path_mod}/es_formatted_lgl.pkl', 'rb') as f:
        es_data_lgl = pickle.load(f)

    with open(f'training/{path_mod}/es_formatted_gwn.pkl', 'rb') as f:
        es_data_gwn = pickle.load(f)
    
    with open(f'training/{path_mod}/es_formatted_syn_cities.pkl', 'rb') as f:
        es_data_syn = pickle.load(f)

    with open(f'training/{path_mod}/es_formatted_syn_caps.pkl', 'rb') as f:
        es_data_syn_caps = pickle.load(f)

    def split_list(data, frac=0.7):
        split = round(frac*len(data))
        return data[0:split], data[split:]

    es_data_prod, es_data_prod_val = split_list(es_data_prod)
    es_data_tr, es_data_tr_val = split_list(es_data_tr)
    es_data_lgl, es_data_lgl_val = split_list(es_data_lgl)
    es_data_gwn, es_data_gwn_val = split_list(es_data_gwn)
    es_data_syn, es_data_syn_val = split_list(es_data_syn)
    train_data = es_data_prod + es_data_tr + es_data_lgl + es_data_gwn + es_data_syn
    random.seed(617)
    random.shuffle(train_data)
    return train_data, es_data_prod_val, es_data_tr_val, es_data_lgl_val, es_data_gwn_val, es_data_syn_val

if __name__ == "__main__":
    wandb.init(project="mordecai3", entity="ahalt")

    config = wandb.config          # Initialize config
    config.batch_size = 32         # input batch size for training 
    config.test_batch_size = 64    # input batch size for testing 
    config.epochs = 25           # number of epochs to train 
    config.lr = 0.01               # learning rate 
    config.seed = 42               # random seed (default: 42)
    config.log_interval = 10     # how many batches to wait before logging training status
    config.max_choices = 500
    config.avg_params = True
    config.limit_es_results = False

    es_train_data, es_data_prod_val, es_data_tr_val, es_data_lgl_val, es_data_gwn_val, es_data_syn_val  = load_data(config.limit_es_results)
    logger.info(f"Total training examples: {len(es_train_data)}")
    
    train_data = TrainData(es_train_data, max_choices=config.max_choices)
    tr_data = TrainData(es_data_tr_val, max_choices=config.max_choices)
    prod_data = TrainData(es_data_prod_val, max_choices=config.max_choices)
    lgl_data = TrainData(es_data_lgl_val, max_choices=config.max_choices)
    gwn_data = TrainData(es_data_gwn_val, max_choices=config.max_choices)
    syn_data = TrainData(es_data_syn_val, max_choices=config.max_choices)

    train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True)
    train_val_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=False)
    prod_loader = DataLoader(dataset=prod_data, batch_size=config.test_batch_size, shuffle=False)
    tr_loader = DataLoader(dataset=tr_data, batch_size=config.test_batch_size, shuffle=False)
    lgl_loader = DataLoader(dataset=lgl_data, batch_size=config.test_batch_size, shuffle=False)
    gwn_loader = DataLoader(dataset=gwn_data, batch_size=config.test_batch_size, shuffle=False)
    syn_loader = DataLoader(dataset=syn_data, batch_size=config.test_batch_size, shuffle=False)

    datasets = [es_train_data, es_data_prod_val, es_data_tr_val, es_data_lgl_val, es_data_gwn_val, es_data_syn_val]
    data_loaders = [train_val_loader, prod_loader, tr_loader, lgl_loader, gwn_loader, syn_loader]
    names = ["mixed training", "prodigy", "TR", "LGL", "GWN", "Synth"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = embedding_compare(device = device,
                              bert_size = train_data.placename_tensor.shape[1],
                              num_feature_codes=53+1)
    #model = torch.nn.DataParallel(model)
    model.to(device)
    # Future work: Can add  an "ignore_index" argument so that some inputs don't have losses calculated
    loss_func=nn.CrossEntropyLoss() # single label, multi-class
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    if config.avg_params:
        from torch.optim.swa_utils import AveragedModel, SWALR
        from torch.optim.lr_scheduler import CosineAnnealingLR

        swa_model = AveragedModel(model)
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs+1)
        swa_start = 5
        swa_scheduler = SWALR(optimizer, swa_lr=0.05)
    #else:
    #    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs+1)


    wandb.watch(model, log='all')
    logger.setLevel(logging.INFO)

    model.train()
    for epoch in range(1, config.epochs+1):
        epoch_loss = 0
        epoch_acc = 0

        for label, country, input in train_loader:
            optimizer.zero_grad()
            label_pred = model(input)

            #logger.debug(country_pred[1])
            loss = loss_func(label_pred, label)
            #loss_country = loss_func(country_pred, country)
            #loss = loss_label + loss_country
            acc = binary_acc(label_pred, label)
            #country_acc = binary_acc(country_pred, country)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        if config.avg_params:
            if epoch > swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()

        wandb_dict = make_wandb_dict(names, datasets, data_loaders, model)
        wandb_dict['loss'] = epoch_loss/len(train_loader)

        print(f"Epoch {epoch+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Exact Match: {wandb_dict['exact_match_avg']:.3f} | Country Match: {wandb_dict['country_avg']:.3f}")  # | Prodigy Acc: {epoch_acc_prod/len(prod_loader):.3f} | TR Acc: {epoch_acc_tr/len(tr_loader):.3f} | LGL Acc: {epoch_acc_lgl/len(lgl_loader):.3f} | GWN Acc: {epoch_acc_gwn/len(gwn_loader):.3f} | Syn Acc: {epoch_acc_syn/len(syn_loader):.3f}')
        wandb.log(wandb_dict)

    logger.info("Saving model...")
    torch.save(model.state_dict(), "data/mordecai2.pt")
    logger.info("Run complete.")

