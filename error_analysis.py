from torch_bert_placename_compare import TrainData, ProductionData, load_data, embedding_compare
from torch.utils.data import Dataset, DataLoader
import torch
from collections import Counter
import numpy as np
from rich.console import Console
from rich.table import Table

from error_utils import evaluate_results, make_wandb_dict

import logging
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

console = Console()

max_choices = 50
test_batch_size = 64


#
#train_loader = DataLoader(dataset=train_data, batch_size=test_batch_size, shuffle=False)
#prod_loader = DataLoader(dataset=prod_data, batch_size=test_batch_size, shuffle=False)
#tr_loader = DataLoader(dataset=tr_data, batch_size=test_batch_size,   shuffle=False)
#lgl_loader = DataLoader(dataset=lgl_data, batch_size=test_batch_size, shuffle=False)
#gwn_loader = DataLoader(dataset=gwn_data, batch_size=test_batch_size, shuffle=False)
#syn_loader = DataLoader(dataset=syn_data, batch_size=test_batch_size, shuffle=False)
#
#datasets = [es_train_data, es_data_prod_val, es_data_tr_val, es_data_lgl_val, es_data_gwn_val, es_data_syn_val]
#data_loaders = [train_loader, prod_loader, tr_loader, lgl_loader, gwn_loader, syn_loader]
#names = ["mixed training", "prodigy", "TR", "LGL", "GWN", "Synth"]

import wandb

config = wandb.config          # Initialize config
config.batch_size = 32         # input batch size for training (default: 64)
config.test_batch_size = 64    # input batch size for testing (default: 1000)
config.epochs = 15           # number of epochs to train (default: 10)
config.lr = 0.01               # learning rate 
config.seed = 42               # random seed (default: 42)
config.log_interval = 10     # how many batches to wait before logging training status
config.max_choices = 500
config.avg_params = False

es_train_data, es_data_prod_val, es_data_tr_val, es_data_lgl_val, es_data_gwn_val, es_data_syn_val  = load_data()

logger.info(f"Total training examples: {len(es_train_data)}")

train_data = TrainData(es_train_data, max_choices=max_choices)
tr_data = TrainData(es_data_tr_val, max_choices=max_choices)
prod_data = TrainData(es_data_prod_val, max_choices=max_choices)
lgl_data = TrainData(es_data_lgl_val, max_choices=max_choices)
gwn_data = TrainData(es_data_gwn_val, max_choices=max_choices)
syn_data = TrainData(es_data_syn_val, max_choices=max_choices)


train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True)
prod_loader = DataLoader(dataset=prod_data, batch_size=config.test_batch_size, shuffle=True)
tr_loader = DataLoader(dataset=tr_data, batch_size=config.test_batch_size, shuffle=True)
lgl_loader = DataLoader(dataset=lgl_data, batch_size=config.test_batch_size, shuffle=True)
gwn_loader = DataLoader(dataset=gwn_data, batch_size=config.test_batch_size, shuffle=True)
syn_loader = DataLoader(dataset=syn_data, batch_size=config.test_batch_size, shuffle=True)

datasets = [es_train_data, es_data_prod_val, es_data_tr_val, es_data_lgl_val, es_data_gwn_val, es_data_syn_val]
data_loaders = [train_loader, prod_loader, tr_loader, lgl_loader, gwn_loader, syn_loader]
names = ["mixed training", "prodigy", "TR", "LGL", "GWN", "Synth"]

def make_missing_table(cutoff):
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Dataset")
    table.add_column(f"Percentage Missing at {cutoff}", justify="right")

    for nn, data in zip(names, datasets): 
        c = Counter([np.sum(i['correct'][:cutoff]) == 0 for i in data])
        frac_missing = str(round(100 * c[True] / len(data), 1)) + "%"
        table.add_row(nn, frac_missing)
    console.print(table)

#make_missing_table(2000)
#    ┃ Dataset        ┃ Percentage Missing at 2000 ┃
#    ┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
#    │ mixed training │                       4.1% │
#    │ prodigy        │                       0.0% │
#    │ TR             │                       2.6% │
#    │ LGL            │                       3.3% │
#    │ GWN            │                       6.0% │
#    │ Synth          │                      22.8% │




def load_model(path="data/mordecai2.pt"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = embedding_compare(device = device,
                                bert_size = 768,
                                num_feature_codes=54) 
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def make_table():
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Dataset")
    table.add_column(f"Exact match", justify="right")
    table.add_column(f"Correct Country", justify="right")
    table.add_column(f"Correct Feature Code", justify="right")
    table.add_column(f"Correct ADM1", justify="right")

    for nn, data, loader in zip(names, datasets, data_loaders): 
        c_country, c_code, c_adm1, c_geoid = evaluate_results(data, loader, model)
        country_perc = str(round(100 * c_country, 1)) + "%"
        geoid_perc = str(round(100 * c_geoid, 1)) + "%"
        code_perc = str(round(100 * c_code, 1)) + "%"
        adm1_perc = str(round(100 * c_adm1, 1)) + "%"
        table.add_row(nn, geoid_perc, country_perc, code_perc, adm1_perc)
    console.print(table)

# ┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
# ┃ Dataset        ┃ Exact match ┃ Correct Country ┃ Correct Feature Code ┃ Correct ADM1 ┃
# ┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
# │ mixed training │       83.7% │           93.5% │                88.9% │        86.8% │
# │ prodigy        │       86.4% │           95.6% │                88.0% │        92.8% │
# │ TR             │       78.4% │           91.4% │                84.0% │        81.3% │
# │ LGL            │       74.7% │           92.3% │                82.4% │        78.4% │
# │ GWN            │       86.3% │           91.6% │                87.6% │        89.9% │
# │ Synth          │       64.0% │           70.3% │                71.7% │        69.5% │
# └────────────────┴─────────────┴─────────────────┴──────────────────────┴──────────────┘

if __name__ == "__main__":
    make_missing_table(500)
    make_missing_table(50)
    model = load_model('data/mordecai_prod.pt')
    make_table()
    t = make_wandb_dict(names, datasets, data_loaders, model)
    print(t)
