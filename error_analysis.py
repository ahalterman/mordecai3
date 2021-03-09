from torch_bert_placename_compare import TrainData, ProductionData, load_data, embedding_compare
from torch.utils.data import Dataset, DataLoader
import torch
from collections import Counter
import numpy as np
from rich.console import Console
from rich.table import Table

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

es_train_data, es_data_prod_val, es_data_tr_val, es_data_lgl_val, es_data_gwn_val, es_data_syn_val  = load_data()
train_data = TrainData(es_train_data, max_choices=max_choices)
tr_data = TrainData(es_data_tr_val, max_choices=max_choices)
prod_data = TrainData(es_data_prod_val, max_choices=max_choices)
lgl_data = TrainData(es_data_lgl_val, max_choices=max_choices)
gwn_data = TrainData(es_data_gwn_val, max_choices=max_choices)
syn_data = TrainData(es_data_syn_val, max_choices=max_choices)

train_loader = DataLoader(dataset=train_data, batch_size=test_batch_size, shuffle=False)
prod_loader = DataLoader(dataset=prod_data, batch_size=test_batch_size, shuffle=False)
tr_loader = DataLoader(dataset=tr_data, batch_size=test_batch_size,   shuffle=False)
lgl_loader = DataLoader(dataset=lgl_data, batch_size=test_batch_size, shuffle=False)
gwn_loader = DataLoader(dataset=gwn_data, batch_size=test_batch_size, shuffle=False)
syn_loader = DataLoader(dataset=syn_data, batch_size=test_batch_size, shuffle=False)

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
make_missing_table(500)
make_missing_table(50)



def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = embedding_compare(device = device,
                                bert_size = 768,
                                num_feature_codes=54) 
    model.load_state_dict(torch.load("mordecai2.pt"))
    model.eval()
    return model

model = load_model()

def correct_country(es_data, loader, model):
    pred_val_list = []
    with torch.no_grad():
        model.eval()
        for label, input in loader:
            pred_val = model(input)
            pred_val_list.append(pred_val)
    pred_array = np.vstack(pred_val_list)

    correct_country = []
    correct_code = []
    correct_adm1 = []
    correct_geoid = []
    for ent, pred in zip(es_data, pred_array):
        for n, i in enumerate(pred):
            if n < len(ent['es_choices']):
                ent['es_choices'][n]['score'] = i
        correct_position = np.where(ent['correct'])[0]
        if len(correct_position) == 0:
            correct_country.append(False)
            correct_code.append(False)
            correct_adm1.append(False)
            correct_geoid.append(False)
            continue
        gold_country = ent['es_choices'][correct_position[0]]['country_code3']
        gold_code = ent['es_choices'][correct_position[0]]['feature_code']
        gold_adm1 = ent['es_choices'][correct_position[0]]['admin1_code']
        predicted_position = np.argmax([i['score'] for i in ent['es_choices'] if 'score' in i.keys()])
        predicted_country = ent['es_choices'][predicted_position]['country_code3']
        predicted_code = ent['es_choices'][predicted_position]['feature_code']
        predicted_adm1 = ent['es_choices'][predicted_position]['admin1_code']
        predicted_geoid = ent['es_choices'][predicted_position]['geonameid']
        correct_country.append(gold_country == predicted_country)
        correct_code.append(gold_code == predicted_code)
        correct_adm1.append(gold_adm1 == predicted_adm1)
        correct_geoid.append(ent['correct_geonamesid'] == predicted_geoid)
    return np.mean(correct_country), np.mean(correct_code), np.mean(correct_adm1), np.mean(correct_geoid)


table = Table(show_header=True, header_style="bold magenta")
table.add_column("Dataset")
table.add_column(f"Exact match", justify="right")
table.add_column(f"Correct Country", justify="right")
table.add_column(f"Correct Feature Code", justify="right")
table.add_column(f"Correct ADM1", justify="right")

for nn, data, loader in zip(names, datasets, data_loaders): 
    c_country, c_code, c_adm1, c_geoid = correct_country(data, loader, model)
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