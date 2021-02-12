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



class trainData(Dataset):
    def __init__(self, X_data, y_data, labels):
        assert X_data.shape[0] == X_data.shape[0]
        assert X_data.shape[0] == labels.shape[0]
        self.X_data = X_data.astype(np.float32) #torch.tensor(X_data, dtype=torch.float32)
        self.y_data = y_data.astype(np.long) #torch.tensor(y_data, dtype=torch.long)
        self.labels = labels.astype(np.float32) #torch.tensor(labels, dtype=torch.long)
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index], self.labels[index]
        
    def __len__ (self):
        return len(self.X_data)

## test data    
class testData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = torch.tensor(X_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.long)
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


class embedding_compare(nn.Module):
    def __init__(self, bert_size, num_embeddings, embedding_dim=32):
        super(embedding_compare, self).__init__()
        # Take in two inputs: the text, and the country. Learn an embedding for country.
        self.text_layer = nn.Linear(bert_size, 64) 
        self.text2 = nn.Linear(64, 32)
        self.emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        self.similarity = nn.CosineSimilarity()
        #self.layer_out = nn.Linear(1, 1)
        
    def forward(self, inputs: list):
        text, country = inputs
        x = self.sigmoid(self.text_layer(text))
        x = self.text2(x)
        #print("x_dim:", x.shape)
        y = self.emb_layer(country)
        #print("y_dim:", y.shape)
        cos_sim = self.similarity(x, y)
        #print("cos_sim:", cos_sim.shape)
        out = self.sigmoid(cos_sim)
        return out



def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

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

# duplicate so we get some negative examples
X_train = np.load("X_train.npy")
X_train = np.vstack([X_train, X_train])

X_test = np.load("X_test.npy")
X_test = np.vstack([X_test, X_test])

y_test = np.load("y_test.npy")
y_test2 = augment_data(y_test)
y_test = np.vstack([y_test, y_test2])
y_test_cat = np.argmax(y_test, axis=1).astype("int")

y_train = np.load("y_train.npy")
y_train2 = augment_data(y_train)
y_train = np.vstack([y_train, y_train2])
y_train_cat = np.argmax(y_train, axis=1).astype("int")

labels_train = np.concatenate([np.ones(y_train2.shape[0]), np.zeros(y_train2.shape[0])]).astype("int")
#labels_train = np.ones(y_train.shape[0]).astype("int")
labels_test = np.concatenate([np.ones(y_test2.shape[0]), np.zeros(y_test2.shape[0])]).astype("int")
#labels_test = np.ones(y_test.shape[0]).astype("int")


## Create the pytorch data objects
train_data = trainData(X_train, 
                       y_train_cat,
                       labels_train)
test_data = testData(X_test, 
                     y_test_cat)


EPOCHS = 100
BATCH_SIZE = 44
LEARNING_RATE = 0.001

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = embedding_compare(bert_size = X_train.shape[1],
                            num_embeddings=y_train.shape[1], 
                            embedding_dim=32)
model.to(device)
print(model)
loss_func=nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


model.train()
for e in range(1, EPOCHS+1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch, label_batch in train_loader:
        X_batch, y_batch, label_batch = X_batch.to(device), y_batch.to(device), label_batch.to(device)
        optimizer.zero_grad()
        label_pred = model([X_batch, y_batch])
        #print(torch.mean(label_batch)) 
        
        loss = loss_func(label_pred, label_batch)
        acc = binary_acc(label_pred, label_batch)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

