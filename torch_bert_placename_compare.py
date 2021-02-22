## Read in the BERT embedding for each place name
## and predict the country using pytorch
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import pickle
import json

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


with open('es_formatted.pkl', 'rb') as f:
    # Pickle dictionary using protocol 0.
    es_data = pickle.load(f)




# Get the label for whether each guess is correct or not

# convert feature code to numeric value, handling padding and 
# unknown valudes

max_choices = 25

import pandas as pd

country = pd.read_csv("wikipedia-iso-country-codes.txt")
country_dict = {i:n for n, i in enumerate(country['Alpha-3 code'].to_list())}
country_dict["CUW"] = len(country_dict)+1
country_dict["XKX"] = len(country_dict)+1
country_dict["SCG"] = len(country_dict)+1
country_dict["SSD"] = len(country_dict)+1
country_dict["NULL"] = len(country_dict)+1
country_dict["NA"] = len(country_dict)+1

with open("feature_code_dict.json", "r") as f:
    feature_code_dict = json.load(f)


#all_country_codes = []
#for ex in es_data:
#    country_code_raw = [i['country_code3'] for i in ex['es_choices']]
#    country_code_raw += ['NULL'] * (max_choices - len(country_code_raw))
#    country_codes = [country_dict[i] for i in country_code_raw]
#    # the last one is an other/not present category
#    feature_codes[-1] = "OTHER" 
#    all_feature_codes.append(feature_codes)


class TrainData(Dataset):
    def __init__(self, es_data, max_choices=25, max_codes=50):
        self.max_codes = max_codes
        #self.X_data = X_data.astype(np.float32) #torch.tensor(X_data, dtype=torch.float32)
        #self.country_code = y_data.astype(np.long) #torch.tensor(y_data, dtype=torch.long)
        self.X_data = np.array([i['tensor'] for i in es_data]).astype(np.float32)
        self.labels = self.create_labels(es_data)
        self.feature_codes = self.create_feature_codes(es_data)
        self.country_codes = self.create_country_codes(es_data)
        
    def __getitem__(self, index):
        return self.X_data[index], self.labels[index], self.feature_codes[index], self.country_codes[index]
        
    def __len__ (self):
        return len(self.X_data)

    def create_labels(self, es_data):
        """Create an array with the location of the correct geonames entry"""
        all_labels = []
        for ex in es_data:
            labels = np.zeros(max_choices)
            if np.sum(ex['correct']) == 0:
               labels[-1] = 1 
            else:
                correct_num = np.where(np.array(ex['correct']))[0]
                if correct_num[0] >= max_choices:
                    # TODO: make a better missing/NA prediction. Maybe the first one??
                    labels[-1] = 1
                else:
                    labels[correct_num] = 1
            ## HACK here: convert back to index, not one-hot
            labels = np.argmax(labels)
            all_labels.append(labels)
        all_labels = np.array(all_labels).astype(np.long)
        return all_labels

    # need to make this into a one-hot matrix, not a vector.
    # Inside the model, it should be a 3d one hot tensor, not binary.
    def create_feature_codes(self, es_data):
        all_feature_codes = []
        for ex in es_data:
            feature_code_raw = [i['feature_code'] for i in ex['es_choices']]
            feature_code_raw += ['NULL'] * (max_choices - len(feature_code_raw))
            feature_code_raw = feature_code_raw[0:max_choices]
            feature_codes = [feature_code_dict[i] if i in feature_code_dict else len(feature_code_dict)+1 for i in feature_code_raw]
            # the last one is an other/not present category
            feature_codes[-1] = 53
            feature_codes = np.array(feature_codes, dtype="int")
            ## DON'T DO THIS: Pytorch embedding layers need indices, not one-hot
            # convert from a vector [0, 3, 1, 18,...] to a one-hot matrix
            #one_hot = np.zeros((max_choices, 53+1))
            #one_hot[np.arange(feature_codes.size), feature_codes] = 1
            #all_feature_codes.append(one_hot)
            all_feature_codes.append(feature_codes)
        all_feature_codes = np.array(all_feature_codes).astype(np.long)
        return all_feature_codes

    def create_country_codes(self, es_data):
        all_country_codes = []
        for ex in es_data:
            country_code_raw = [i['country_code3'] for i in ex['es_choices']]
            country_code_raw += ['NULL'] * (max_choices - len(country_code_raw))
            country_code_raw = country_code_raw[0:max_choices]
            country_codes = [country_dict[i] for i in country_code_raw]
            country_codes = np.array(country_codes, dtype="int")
            all_country_codes.append(country_codes)
        all_country_codes = np.array(all_country_codes).astype(np.long)
        return all_country_codes

split = round(len(es_data)*0.7)
train_data = TrainData(es_data[0:split])
val_data = TrainData(es_data[split:])


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.argmax(y_pred, axis=1)
    #y_test_cat = torch.argmax(y_test, axis=1) 
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_pred.shape[0]
    acc = torch.round(acc * 100)
    return acc


class embedding_compare(nn.Module):
    def __init__(self, bert_size, num_countries, num_feature_codes,
                max_choices):
        super(embedding_compare, self).__init__()
        # Take in two inputs: the text, and the country. Learn an embedding for country.
        self.text_layer = nn.Linear(bert_size, 64) 
        self.text2 = nn.Linear(64, 32)
        self.code_emb = nn.Embedding(num_feature_codes, 8)
        self.country_emb = nn.Embedding(num_countries, 24)
        
        self.sigmoid = nn.Sigmoid()
        self.last_linear = nn.Linear(max_choices, max_choices)
        self.softmax = nn.Softmax(dim=0)
        self.dropout = nn.Dropout(p=0.2) 
        self.similarity = nn.CosineSimilarity(dim=2)
        #self.layer_out = nn.Linear(1, 1)

    def forward(self, text, feature_code, country_code):
        logger.debug("feature_code input shape:{}".format(feature_code.shape))
        x = self.sigmoid(self.text_layer(text))
        x = self.text2(x)
        logger.debug("x_dim: {}".format(x.shape))
        #print("country shape: ", country.shape)
        fc = self.code_emb(feature_code)
        cc = self.country_emb(country_code)
        # to match the stacked value below, rearrange so it's
        # (choices, batch_size, embed_size)
        fc = fc.permute(1, 0, 2)
        fc = self.dropout(fc)
        cc = cc.permute(1, 0, 2)
        cc = self.dropout(cc)
        logger.debug("feature_code_emb: {}".format(fc.shape))
        logger.debug("country_code_emb: {}".format(cc.shape))
        embed_stack = torch.cat((fc, cc), 2)
        logger.debug("stack_emb: {}".format(embed_stack.shape))
        #y_stack = torch.stack([y1, y2])
        # turn x from (batch_size, choices) into (1, batch_size, choices)
        # so it can be broadcast into a similarity comparison with all the ys.
        x_stack = torch.unsqueeze(x, 0) #torch.stack([x, x])
        #x_stack = self.dropout(x_stack)
        logger.debug("x shape: {}".format(x_stack.shape))
        # x_stack is (choices, batch_size, embed_size)
        cos_sim = self.similarity(x_stack, embed_stack)
        logger.debug("cos_sim:{}".format(cos_sim.shape))
        # make sure cos_sim is (batch size, choices)  (44, 129)
        cos_sim = torch.transpose(cos_sim, 0, 1)
        logger.debug("cos_sim: {}".format(cos_sim.shape))
        #cos_sim = self.dropout(cos_sim)
        #last = self.last_linear(cos_sim)
        #out = self.softmax(cos_sim)
        out = cos_sim
        logger.debug("out shape: {}".format(out.shape))  # should be (batch_size, choices)  (44, 129)
        return out


EPOCHS = 30 
BATCH_SIZE = 44
LEARNING_RATE = 0.01

#X_train, y_train, labels_train, X_test, y_test, labels_test = load_data()
## Create the pytorch data objects
#train_data = CountryData(X_train, 
#                       y_train,
#                       labels_train)
#val_data = CountryData(X_test, 
#                        y_test,
#                        labels_test)

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = embedding_compare(bert_size = train_data.X_data.shape[1],
                            num_feature_codes=53+1, 
                            num_countries=len(country_dict)+1,
                            max_choices=25)
model.to(device)
#print(model)
# Can add  an "ignore_index" argument so that some inputs don't have losses calculated
loss_func=nn.CrossEntropyLoss() #BCELoss()
#loss_func = nn.NLLLOSS()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

logger.setLevel(logging.INFO)

model.train()
for e in range(1, EPOCHS+1):
    epoch_loss = 0
    epoch_acc = 0
    epoch_loss_val = 0
    epoch_acc_val = 0
    for X_batch, label_batch, feature_codes, country_codes in train_loader:
        X_batch, feature_codes, label_batch, country_codes = X_batch.to(device), feature_codes.to(device), label_batch.to(device), country_codes.to(device)
        optimizer.zero_grad()
        label_pred = model(X_batch, feature_codes, country_codes)
        
        loss = loss_func(label_pred, label_batch)
        acc = binary_acc(label_pred, label_batch)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    # evaluate once per epoch
    with torch.no_grad():
        model.eval()
        for X_val, label_val, code_val, country_val in val_loader:
            X_val = X_val.to(device)
            label_val = label_val.to(device)
            code_val = code_val.to(device)
            country_val = country_val.to(device)

            pred_val = model(X_val, code_val, country_val)
    #        val_loss = loss_func(label_val, pred_label)
            val_acc = binary_acc(pred_val, label_val)
    #        epoch_loss_val += val_loss.item()
            epoch_acc_val += val_acc.item() 
    
    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f} | Val Acc: {epoch_acc_val/len(val_loader):.3f}')

