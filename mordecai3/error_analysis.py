from mordecai3.torch_model import TrainData, geoparse_model
from mordecai3.train import load_data
from torch.utils.data import DataLoader
import torch
from collections import Counter
import numpy as np
from rich.console import Console
from rich.table import Table
import typer
from pathlib import Path

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



def make_missing_table(cutoff, names, datasets):
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




def load_model(path="assets/mordecai2.pt"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = geoparse_model(device = device,
                                bert_size = 768,
                                num_feature_codes=54)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def make_table(names, datasets, data_loaders, model):
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Dataset")
    table.add_column("Eval N", justify="right")
    table.add_column(f"Exact match", justify="right")
    table.add_column(f"Mean Error", justify="right")
    table.add_column(f"Correct Country", justify="right")
    table.add_column(f"Correct Feature Code", justify="right")
    table.add_column(f"Correct ADM1", justify="right")
    table.add_column(f"Acc @161km", justify="right")

    for nn, data, loader in zip(names, datasets, data_loaders):
        correct_avg = evaluate_results(data, loader, model)
        country_perc = str(round(100 * correct_avg['correct_country'], 1)) + "%"
        geoid_perc = str(round(100 * correct_avg['exact_match'], 1)) + "%"
        code_perc = str(round(100 * correct_avg['correct_code'], 1)) + "%"
        adm1_perc = str(round(100 * correct_avg['correct_adm1'], 1)) + "%"
        avg_dist = str(round(correct_avg['avg_dist'], 1))
        correct_161 = str(round(100 * correct_avg['acc_at_161'], 1))
        table.add_row(nn, str(len(data)), geoid_perc, avg_dist, country_perc, code_perc, adm1_perc, correct_161)
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

def main(data_dir: Path,
        path: Path):
    config = wandb.config          # Initialize config
    config.batch_size = 32         # input batch size for training (default: 64)
    config.test_batch_size = 64    # input batch size for testing (default: 1000)
    config.epochs = 15           # number of epochs to train (default: 10)
    config.lr = 0.01               # learning rate
    config.seed = 42               # random seed (default: 42)
    config.log_interval = 10     # how many batches to wait before logging training status
    config.max_choices = 500
    config.avg_params = False
    config.fuzzy=0
    config.limit_es_results = "all_loc_types"
    print(config.__dict__)
    
    logger.info("Loading data...")
    es_train_data, es_data_prod_val, es_data_tr_val, es_data_lgl_val, es_data_gwn_val, es_data_syn_val  = load_data(data_dir, max_results=config.max_choices, fuzzy=config.fuzzy, limit_types=config.limit_es_results)

    logger.info(f"Total training examples: {len(es_train_data)}")

    train_data = TrainData(es_train_data, max_choices=config.max_choices)
    tr_data = TrainData(es_data_tr_val, max_choices=config.max_choices)
    prod_data = TrainData(es_data_prod_val, max_choices=config.max_choices)
    lgl_data = TrainData(es_data_lgl_val, max_choices=config.max_choices)
    gwn_data = TrainData(es_data_gwn_val, max_choices=config.max_choices)
    syn_data = TrainData(es_data_syn_val, max_choices=config.max_choices)


    train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=False)
    prod_loader = DataLoader(dataset=prod_data, batch_size=config.test_batch_size, shuffle=False)
    tr_loader = DataLoader(dataset=tr_data, batch_size=config.test_batch_size, shuffle=False)
    lgl_loader = DataLoader(dataset=lgl_data, batch_size=config.test_batch_size, shuffle=False)
    gwn_loader = DataLoader(dataset=gwn_data, batch_size=config.test_batch_size, shuffle=False)
    syn_loader = DataLoader(dataset=syn_data, batch_size=config.test_batch_size, shuffle=False)

    datasets = [es_train_data, es_data_prod_val, es_data_tr_val, es_data_lgl_val, es_data_gwn_val, es_data_syn_val]
    data_loaders = [train_loader, prod_loader, tr_loader, lgl_loader, gwn_loader, syn_loader]
    names = ["mixed training", "prodigy", "TR", "LGL", "GWN", "Synth"]

    make_missing_table(500, names, datasets)
    make_missing_table(50, names, datasets)
    model = load_model(path)
    make_table(names, datasets, data_loaders, model)
    t = make_wandb_dict(names, datasets, data_loaders, model)
    #print(t)

if __name__ == "__main__":
    typer.run(main)

