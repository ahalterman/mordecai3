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
logger.setLevel(logging.DEBUG)

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
        Format all non-query/gazetteer-only feautures.

        Specifically, this includes edit distance features and adm1 and country overlap
        """
        edit_info = []
        for ex in es_data:
            min_dist = [i['min_dist'] for i in ex['es_choices'][0:self.max_choices]]
            min_dist += [1] * (self.max_choices - len(min_dist))
            max_dist = [i['max_dist'] for i in ex['es_choices'][0:self.max_choices]]
            max_dist += [1] * (self.max_choices - len(max_dist))
            avg_dist = [i['avg_dist'] for i in ex['es_choices'][0:self.max_choices]]
            avg_dist += [1] * (self.max_choices - len(avg_dist))
            adm1_overlap = [i['adm1_count'] for i in ex['es_choices'][0:self.max_choices]]
            adm1_overlap += [1] * (self.max_choices - len(adm1_overlap))
            country_overlap = [i['country_count'] for i in ex['es_choices'][0:self.max_choices]]
            country_overlap += [1] * (self.max_choices - len(country_overlap))
            ed = np.transpose(np.array([max_dist, avg_dist, min_dist, adm1_overlap, country_overlap]))
            edit_info.append(ed)
        ed_stack = np.stack(edit_info)
        return ed_stack

    def _make_country_dict(self):
        country = read_csv("wikipedia-iso-country-codes.txt")
        country_dict = {i:n for n, i in enumerate(country['Alpha-3 code'].to_list())}
        country_dict["CUW"] = len(country_dict)
        country_dict["XKX"] = len(country_dict)
        country_dict["SCG"] = len(country_dict)
        country_dict["SSD"] = len(country_dict)
        country_dict["BES"] = len(country_dict)
        country_dict["NULL"] = len(country_dict)
        country_dict["NA"] = len(country_dict)
        return country_dict

    def _make_feature_code_dict(self):
        with open("feature_code_dict.json", "r") as f:
            feature_code_dict = json.load(f)
            return feature_code_dict


class TrainData(ProductionData):
    def __init__(self, es_data, max_choices=25, max_codes=50):
        super().__init__(es_data, max_choices, max_codes)
        self.labels = self.create_labels(es_data)

    def __getitem__(self, index):
        return (self.labels[index],
            {"placename_tensor": self.placename_tensor[index],  
                "doc_tensor": self.doc_tensor[index], 
                "other_locs_tensor": self.other_locs_tensor[index],
                "feature_codes": self.feature_codes[index], 
                "country_codes": self.country_codes[index],
                "gaz_info": self.gaz_info[index]}) 

    def create_labels(self, es_data):
        """Create an array with the location of the correct geonames entry"""
        all_labels = []
        for ex in es_data:
            labels = np.zeros(self.max_choices)
            if np.sum(ex['correct']) == 0:
               labels[-1] = 1 
            else:
                correct_num = np.where(np.array(ex['correct']))[0]
                if correct_num[0] >= self.max_choices:
                    # TODO: make a better missing/NA prediction.
                    labels[-1] = 1
                else:
                    labels[correct_num] = 1
            ## HACK here: convert back to index, not one-hot
            labels = np.argmax(labels)
            all_labels.append(labels)
        all_labels = np.array(all_labels).astype(np.long)
        return all_labels





def binary_acc(y_pred, y_test):
    y_pred_tag = torch.argmax(y_pred, axis=1)
    #y_test_cat = torch.argmax(y_test, axis=1) 
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_pred.shape[0]
    acc = torch.round(acc * 100)
    return acc


class embedding_compare(nn.Module):
    def __init__(self, device, bert_size, num_feature_codes, max_choices):
        super(embedding_compare, self).__init__()
        self.device = device
        pretrained_country = np.load("country_bert_768.npy")
        pretrained_country = torch.FloatTensor(pretrained_country)
        logging.debug("Pretrained country embedding dim: {}".format(pretrained_country.shape))
        self.text_to_country = nn.Linear(bert_size, 24) 
        self.context_to_country = nn.Linear(bert_size, 24) 
        self.country_layer = nn.Linear(bert_size, 24) 
        #self.text2 = nn.Linear(64, 32)
        self.text_to_code = nn.Linear(bert_size, 8) 
        self.code_emb = nn.Embedding(num_feature_codes, 8)
        #self.country_emb = nn.Embedding(len(country_dict), 24)
        self.country_emb = nn.Embedding.from_pretrained(pretrained_country, freeze=False)
        #self.country_emb.weight.data.copy_(torch.from_numpy(pretrained_country))

        self.sigmoid = nn.Sigmoid()
        self.last_linear = nn.Linear(9, 1) # number of comparisons --> final
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.2) 
        self.similarity = nn.CosineSimilarity(dim=2)
        #nn.Linear()

    def forward(self, input):
        # Unpack the dictionary here. Sending the data to device within the forward
        # function isn't standard, but it makes the training loop code easier to follow.
        placename_tensor = input['placename_tensor'].to(self.device)
        other_locs_tensor = input['other_locs_tensor'].to(self.device)
        doc_tensor = input['doc_tensor'].to(self.device)
        feature_codes = input['feature_codes'].to(self.device)
        country_codes = input['country_codes'].to(self.device)
        gaz_info = input['gaz_info'].to(self.device)

        logger.debug("feature_code input shape:{}".format(feature_codes.shape))
        x = self.text_to_country(placename_tensor)
        x_code = self.text_to_code(placename_tensor)
        x_other_locs = self.context_to_country(other_locs_tensor)
        x_doc = self.context_to_country(doc_tensor)
        logger.debug(f"x shape: {x.shape}")
        #x = self.text2(x)
        logger.debug("x_dim: {}".format(x.shape))
        #print("country shape: ", country.shape)
        fc = self.code_emb(feature_codes)
        cc = self.country_layer(self.country_emb(country_codes))
        # to match the stacked value below, rearrange so it's
        # (choices, batch_size, embed_size)
        fc = fc.permute(1, 0, 2)
        fc = self.dropout(fc)
        cc = cc.permute(1, 0, 2)
        cc = self.dropout(cc)
        logger.debug("feature_code_emb: {}".format(fc.shape))
        logger.debug("country_code_emb: {}".format(cc.shape))
        #embed_stack = torch.cat((fc, cc), 2)
        #logger.debug("stack_emb: {}".format(embed_stack.shape))
        # turn x from (batch_size, choices) into (1, batch_size, choices)
        # so it can be broadcast into a similarity comparison with all the ys.
        x_stack_country = torch.unsqueeze(x, 0) #torch.stack([x, x])
        x_stack_code = torch.unsqueeze(x_code, 0) #torch.stack([x, x])
        x_stack_locs = torch.unsqueeze(x_other_locs, 0)
        x_stack_doc = torch.unsqueeze(x_doc, 0)
        logger.debug("x shape: {}".format(cc.shape))
        # x_stack is (choices, batch_size, embed_size)
        cos_sim_country = self.similarity(x_stack_country, cc)
        cos_sim_code = self.similarity(x_stack_code, fc)
        cos_sim_other_locs = self.similarity(x_stack_locs, cc)
        cos_sim_doc = self.similarity(x_stack_doc, cc)
        logger.debug("cos_sim_country: {}".format(cos_sim_country.shape))
        logger.debug("cos_sim_code: {}".format(cos_sim_country.shape))
        logger.debug("cos_sim_doc: {}".format(cos_sim_doc.shape))
        # put cos_sim into (batch size, choices)  
        cos_sim_country = torch.unsqueeze(torch.transpose(cos_sim_country, 0, 1), 2)
        cos_sim_code = torch.unsqueeze(torch.transpose(cos_sim_code, 0, 1), 2)
        cos_sim_other_locs = torch.unsqueeze(torch.transpose(cos_sim_other_locs, 0, 1), 2)
        cos_sim_doc = torch.unsqueeze(torch.transpose(cos_sim_doc, 0, 1), 2)
        logger.debug("cos_sim_country shape: {}".format(cos_sim_country.shape))
        logger.debug("cos_sim_code shape: {}".format(cos_sim_code.shape))
        logger.debug("gaz_info info shape: {}".format(gaz_info.shape))
        #both_sim = torch.cat((cos_sim_country, cos_sim_code, cos_sim_other_locs, cos_sim_doc), 2)
        both_sim = torch.cat((cos_sim_country, cos_sim_code, cos_sim_other_locs, cos_sim_doc, gaz_info), 2)
        # so the new gaz_info features need to be (batch_size, choices, 3), to make 7 in the last dim.
        logger.debug(f"concat shape: {both_sim.shape}")  # (batch_size, choices, 4)
        #both_sim = self.dropout(both_sim)
        last = torch.squeeze(self.last_linear(both_sim), dim=2)  
        logger.debug(f"last shape: {last.shape}")  # (batch_size, choices)
        logger.debug(f"last max: {torch.max(last)}")
        out = self.softmax(last) 
        logger.debug("out shape: {}".format(out.shape))  # should be (batch_size, choices)  (44, 25) 
        return out

def load_data():
    with open('es_formatted_prodigy.pkl', 'rb') as f:
        es_data = pickle.load(f)
    
    with open('es_formatted_tr.pkl', 'rb') as f:
        es_data_tr = pickle.load(f)
    
    with open('es_formatted_lgl.pkl', 'rb') as f:
        es_data_lgl = pickle.load(f)

    with open('es_formatted_gwn.pkl', 'rb') as f:
        es_data_gwn = pickle.load(f)
    
    train_data = es_data + es_data_tr + es_data_lgl + es_data_gwn
    random.seed(617)
    random.shuffle(es_data)
    return train_data, es_data_tr

if __name__ == "__main__":
    wandb.init(project="mordecai3", entity="ahalt")
    es_data, es_data_tr_eval = load_data()
    split = round(len(es_data)*0.7)
    logger.info(f"Total training examples: {split}")
    train_data = TrainData(es_data[0:split])
    val_data = TrainData(es_data[split:])
    tr_data = TrainData(es_data_tr_eval)

    config = wandb.config          # Initialize config
    config.batch_size = 64         # input batch size for training (default: 64)
    config.test_batch_size = 64    # input batch size for testing (default: 1000)
    config.epochs = 40           # number of epochs to train (default: 10)
    config.lr = 0.01               # learning rate (default: 0.01)
    config.seed = 42               # random seed (default: 42)
    config.log_interval = 10     # how many batches to wait before logging training status

    train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=config.test_batch_size, shuffle=True)
    tr_loader = DataLoader(dataset=tr_data, batch_size=config.test_batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = embedding_compare(device = device,
                              bert_size = train_data.placename_tensor.shape[1],
                              num_feature_codes=53+1, 
                              max_choices=25)
    #model = torch.nn.DataParallel(model)
    model.to(device)
    # Future work: Can add  an "ignore_index" argument so that some inputs don't have losses calculated
    loss_func=nn.CrossEntropyLoss() # single label, multi-class
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    wandb.watch(model)
    logger.setLevel(logging.INFO)

    model.train()
    for e in range(1, config.epochs+1):
        epoch_loss = 0
        epoch_acc = 0
        epoch_loss_val = 0
        epoch_acc_val = 0
        epoch_acc_tr = 0

        for label, input in train_loader:
            optimizer.zero_grad()
            label_pred = model(input)

            loss = loss_func(label_pred, label)
            acc = binary_acc(label_pred, label)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        # evaluate once per epoch
        with torch.no_grad():
            model.eval()
            for label_val, input_val in val_loader:

                pred_val = model(input_val)
                val_acc = binary_acc(pred_val, label_val)
                epoch_acc_val += val_acc.item()

            for label_tr, input_tr in tr_loader:
                pred_tr = model(input_tr)
        #        val_loss = loss_func(label_val, pred_label)
                tr_acc = binary_acc(pred_tr, label_tr)
        #        epoch_loss_val += val_loss.item()
                epoch_acc_tr += tr_acc.item() 

        print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f} | Val Acc: {epoch_acc_val/len(val_loader):.3f} | TR Acc: {epoch_acc_tr/len(tr_loader):.3f}')

        wandb.log({
            "Train Loss": epoch_loss/len(train_loader),
            "Test Accuracy": epoch_acc/len(train_loader),
            "Val Accuracy": epoch_acc_val/len(val_loader),
            "TR Accuracy": epoch_acc_tr/len(tr_loader)})
    logger.info("Saving model...")
    torch.save(model.state_dict(), "mordecai2.pt")
    logger.info("Run complete.")

