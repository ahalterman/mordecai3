import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import typer
from error_utils import evaluate_results, make_wandb_dict
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader
from torch_model import TrainData, geoparse_model
from train import load_data

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




def load_model(path="assets/mordecai_new.pt"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = geoparse_model(device = device,
                                bert_size = 768,
                                num_feature_codes=54)
    model.load_state_dict(torch.load(path))
    model.eval()
    model.to(device)
    return model


def make_table(names, datasets, data_loaders, model, latex=False):
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Dataset")
    table.add_column("Eval N", justify="right")
    table.add_column(f"Exact match", justify="right")
    table.add_column(f"Mean Error (km)", justify="right")
    table.add_column(f"Median Error (km)", justify="right")
    table.add_column(f"Correct Country", justify="right")
    table.add_column(f"Correct Feature Code", justify="right")
    table.add_column(f"Correct ADM1", justify="right")
    table.add_column(f"Missing correct", justify="right")
    table.add_column(f"Total missing", justify="right")
    table.add_column(f"Acc @161km", justify="right")

    latex_rows = []

    for nn, data, loader in zip(names, datasets, data_loaders):
        correct_avg = evaluate_results(data, loader, model)
        country_perc = str(round(100 * correct_avg['correct_country'], 1)) + "%"
        geoid_perc = str(round(100 * correct_avg['exact_match'], 1)) + "%"
        code_perc = str(round(100 * correct_avg['correct_code'], 1)) + "%"
        adm1_perc = str(round(100 * correct_avg['correct_adm1'], 1)) + "%"
        missing_correct = str(round(100 * correct_avg['missing_correct'], 1)) + "%"
        total_missing = str(round(100 * correct_avg['total_missing'], 1)) + "%"
        avg_dist = str(round(correct_avg['avg_dist'], 1))
        median_dist = str(round(correct_avg['median_dist'], 1))
        correct_161 = str(round(100 * correct_avg['acc_at_161'], 1))
        results_list = [nn, str(len(data)), geoid_perc, avg_dist, median_dist, country_perc, code_perc, adm1_perc, missing_correct, total_missing, correct_161]
        table.add_row(*results_list)
        latex_rows.append(" & ".join(results_list))
    if latex:
        print(" & ".join(["Dataset", "Eval N", "Exact match", "Mean Error (km)", "Median Error (km)", "Correct Country", "Correct Feature Code", "Correct ADM1", "Missing correct", "Total missing", "Acc @161km"]))
        for row in latex_rows:
            row = row.replace("%", "\%") + "\\\\"
            print(row)
    else:
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

wandb.init()

def main(data_dir: Path,
        path: Path):
    wandb.init()
    config = wandb.config          # Initialize config
    config.batch_size = 32         # input batch size for training (default: 64)
    config.test_batch_size = 64    # input batch size for testing (default: 1000)
    config.epochs = 15             # number of epochs to train (default: 10)
    config.lr = 0.01               # learning rate
    config.seed = 42               # random seed (default: 42)
    config.log_interval = 10       # how many batches to wait before logging training status
    config.max_choices = 500
    config.avg_params = False
    config.fuzzy=0
    config.limit_es_results = "all_loc_types"
    print(config.__dict__)
    
    logger.info("Loading data...")
    train_loader, es_train_data, data_loaders, datasets = load_data(data_dir, 
                  max_results=config.max_choices, 
                  fuzzy=config.fuzzy, 
                  limit_types=config.limit_es_results,
                  batch_size=config.batch_size,
                  test_batch_size=test_batch_size,
                  train_frac=0.7)

    # "prodigy", "TR", "LGL", "GWN", "Synth", "Wiki"
    es_prodigy_data_val, es_data_tr_val, es_data_lgl_val, es_data_gwn_val, es_data_syn_val, es_data_wiki_val = datasets

    # some ugly code to get the full version of the GWN dataset.
    # We don't train on this so we can compare it with the Gritta results.
    _, _, gwn_loader_full, es_data_gwn_full = load_data(data_dir, 
                  max_results=config.max_choices, 
                  fuzzy=config.fuzzy, 
                  limit_types=config.limit_es_results,
                  batch_size=config.batch_size,
                  test_batch_size=test_batch_size,
                  train_frac=0.01,
                  data_sources = ["GWN"])
    gwn_loader_full = gwn_loader_full[0]
    es_data_gwn_full = es_data_gwn_full[0]

    logger.info(f"Total training examples: {len(es_train_data)}")

    train_data = TrainData(es_train_data, max_choices=config.max_choices)
    tr_data = TrainData(es_data_tr_val, max_choices=config.max_choices)
    prod_data = TrainData(es_prodigy_data_val, max_choices=config.max_choices)
    lgl_data = TrainData(es_data_lgl_val, max_choices=config.max_choices)
    gwn_data = TrainData(es_data_gwn_val, max_choices=config.max_choices)
    gwn_full_data = TrainData(es_data_gwn_full, max_choices=config.max_choices)
    syn_data = TrainData(es_data_syn_val, max_choices=config.max_choices)
    wiki_data = TrainData(es_data_wiki_val, max_choices=config.max_choices)


    train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=False)
    prod_loader = DataLoader(dataset=prod_data, batch_size=config.test_batch_size, shuffle=False)
    tr_loader = DataLoader(dataset=tr_data, batch_size=config.test_batch_size, shuffle=False)
    lgl_loader = DataLoader(dataset=lgl_data, batch_size=config.test_batch_size, shuffle=False)
    gwn_loader = DataLoader(dataset=gwn_data, batch_size=config.test_batch_size, shuffle=False)
    gwn_full_loader = DataLoader(dataset=gwn_full_data, batch_size=config.test_batch_size, shuffle=False)
    syn_loader = DataLoader(dataset=syn_data, batch_size=config.test_batch_size, shuffle=False)
    wiki_loader = DataLoader(dataset=wiki_data, batch_size=config.test_batch_size, shuffle=False)

    datasets = [es_train_data, es_prodigy_data_val, es_data_tr_val, es_data_lgl_val, es_data_gwn_val, es_data_gwn_full, es_data_syn_val, es_data_wiki_val]
    data_loaders = [train_loader, prod_loader, tr_loader, lgl_loader, gwn_loader, gwn_full_loader, syn_loader, wiki_loader]
    names = ["training set", "prodigy", "TR", "LGL", "GWN", "GWN_complete", "Synth", "Wiki"]

    make_missing_table(500, names, datasets)
    make_missing_table(50, names, datasets)
    # path = "mordecai_2023-03-23.pt"
    model = load_model(path)
    make_table(names, datasets, data_loaders, model)
    make_table(names, datasets, data_loaders, model, latex=True)
    t = make_wandb_dict(names, datasets, data_loaders, model)
    #print(t)

if __name__ == "__main__":
    typer.run(main)

