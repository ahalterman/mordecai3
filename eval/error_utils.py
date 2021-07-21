import torch
import numpy as np

def evaluate_results(es_data, loader, model):
    pred_val_list = []
    with torch.no_grad():
        model.eval()
        for label, country, input in loader:
            pred_val = model(input)
            pred_val_list.append(pred_val)
    pred_array = np.vstack(pred_val_list)

    correct_country = []
    correct_code = []
    correct_adm1 = []
    correct_geoid = []
    for ent, pred in zip(es_data, pred_array):
        for n, score in enumerate(pred):
            #score = i[0] # accounting for country prediction
            if n < len(ent['es_choices']):
                ent['es_choices'][n]['score'] = score
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

def make_wandb_dict(names, datasets, data_loaders, model):
    results = {}
    country_avg = 0
    feature_code_avg = 0
    adm1_avg = 0
    exact_match_avg = 0

    for nn, data, loader in zip(names, datasets, data_loaders): 
        c_country, c_code, c_adm1, c_geoid = evaluate_results(data, loader, model)
        results[f"{nn}_country_acc"] = c_country
        country_avg += c_country
        results[f"{nn}_feature_code"] = c_code
        feature_code_avg += c_code
        results[f"{nn}_adm1"] = c_adm1
        adm1_avg += c_adm1
        results[f"{nn}_exact_match"] = c_geoid
        exact_match_avg += c_geoid

    results['country_avg'] = country_avg / len(names)
    results['feature_code_avg'] = feature_code_avg / len(names)
    results['adm1_avg'] = adm1_avg / len(names)
    results['exact_match_avg'] = exact_match_avg / len(names)
    return results