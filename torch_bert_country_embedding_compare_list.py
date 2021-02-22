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


def augment_data(y):
    """randomly permute the labels for an array of one-hot labels"""
    y2 = y.copy()
    orig_pos = np.argmax(y2, axis=1)
    # generate a random offset
    new_pos =  (orig_pos + random.choices([-2, -1, 1, 2], k=y.shape[0])) % y.shape[1]
    # overwrite the old labels
    y2[list(range(0, len(y2))),orig_pos] = 0               
    # add the new fake labels
    y2[list(range(0, len(y2))),new_pos] = 1
    return y2

def load_data():
    # duplicate so we get some negative examples
    X_train = np.load("X_train.npy")

    X_test = np.load("X_test.npy")

    y_test = np.load("y_test.npy")
    y_test_cat = np.argmax(y_test, axis=1).astype("int")
    y_test2_cat = np.argmax(y_test2, axis=1).astype("int")
    y_test_both = np.array(list(zip(y_test_cat, y_test2_cat)))

    y_train = np.load("y_train.npy")

    labels_train = y_train #np.argmax(y_train, axis=1).astype("int")
    labels_test = y_test #np.argmax(y_test, axis=1).astype("int")
    
    return (X_train, y_train, labels_train, 
            X_test, y_test, labels_test)


class CountryData(Dataset):
    def __init__(self, X_data, y_data, labels):
        assert X_data.shape[0] == X_data.shape[0]
        #assert X_data.shape[0] == labels.shape[0]
        self.X_data = X_data.astype(np.float32) #torch.tensor(X_data, dtype=torch.float32)
        self.y_data = y_data.astype(np.long) #torch.tensor(y_data, dtype=torch.long)
        self.labels = labels.astype(np.float32) #torch.tensor(labels, dtype=torch.long)
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index], self.labels[index]
        
    def __len__ (self):
        return len(self.X_data)
        


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.argmax(label_pred, axis=0)
    y_test_cat = torch.argmax(y_test, axis=0) 
    correct_results_sum = (y_pred_tag == y_test_cat).sum().float()
    acc = correct_results_sum/y_pred.shape[1]
    acc = torch.round(acc * 100)
    return acc



class embedding_compare(nn.Module):
    def __init__(self, bert_size, num_embeddings, embedding_dim=32):
        super(embedding_compare, self).__init__()
        # Take in two inputs: the text, and the country. Learn an embedding for country.
        self.text_layer = nn.Linear(bert_size, 64) 
        self.text2 = nn.Linear(64, 32)
        self.emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)
        self.dropout = nn.Dropout(p=0.2) # not used for now
        self.similarity = nn.CosineSimilarity(dim=2)
        #self.layer_out = nn.Linear(1, 1)

    def forward(self, text, country):
        #text, country = inputs
        #text = X_train[0]
        #country1 = y_train[0]
        #country2 = y_train2[0]
        #print("country1:", country1, "country2:", country2)
        x = self.sigmoid(self.text_layer(text))
        x = self.text2(x)
        #print("x_dim:", x.shape)
        y = self.emb_layer(country)
        # to match the stacked value below, rearrange so it's
        # (choices, batch_size, embed_size)
        y = y.permute(1, 0, 2)
        y = self.dropout(y)
        #print("y_dim:", y.shape)
        #y_stack = torch.stack([y1, y2])
        # turn x from (batch_size, choices) into (1, batch_size, choices)
        # so it can be broadcast into a similarity comparison with all the ys.
        x_stack = torch.unsqueeze(x, 0) #torch.stack([x, x])
        #x_stack = self.dropout(x_stack)
        #print("x shape:", x_stack.shape)
        # x_stack is (choices, batch_size, embed_size)
        cos_sim = torch.transpose(self.similarity(x_stack, y), 0, 1)
        # make sure cos_sim is (batch size, choices)
        #print("cos_sim:", cos_sim.shape)
        cos_sim = self.dropout(cos_sim)
        out = self.softmax(cos_sim)
        #print("out shape:", out.shape)
        return out


EPOCHS = 100
BATCH_SIZE = 44
LEARNING_RATE = 0.001

X_train, y_train, labels_train, X_test, y_test, labels_test = load_data()
## Create the pytorch data objects
train_data = CountryData(X_train, 
                       y_train,
                       labels_train)
val_data = CountryData(X_test, 
                        y_test,
                        labels_test)

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = embedding_compare(bert_size = X_train.shape[1],
                            num_embeddings=y_train.shape[1]+1, 
                            embedding_dim=32)
model.to(device)
print(model)
loss_func=nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


model.train()
for e in range(1, EPOCHS+1):
    epoch_loss = 0
    epoch_acc = 0
    epoch_loss_val = 0
    epoch_acc_val = 0
    for X_batch, y_batch, label_batch in train_loader:
        X_batch, y_batch, label_batch = X_batch.to(device), y_batch.to(device), label_batch.to(device)
        optimizer.zero_grad()
        label_pred = model(X_batch, y_batch)
        
        loss = loss_func(label_pred, label_batch)
        acc = binary_acc(label_pred, label_batch)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    # evaluate once per epoch
    with torch.no_grad():
        model.eval()
        for X_val, y_val, label_val in val_loader:
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            label_val = label_val.to(device)
            
            pred_label_val = model(X_val, y_val)
    #        val_loss = loss_func(label_val, pred_label)
            val_acc = binary_acc(pred_label_val, label_val)
    #        epoch_loss_val += val_loss.item()
            epoch_acc_val += val_acc.item() 
    
    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f} | Val Acc: {epoch_acc_val/len(val_loader):.3f}')

