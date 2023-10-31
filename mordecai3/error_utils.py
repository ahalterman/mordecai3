import haversine as hs
import numpy as np
import torch

#es_data = datasets[2]
#loader = data_loaders[2]
#model = geo.model

def evaluate_results(es_data, loader, model):
    pred_val_list = []
    with torch.no_grad():
        model.eval()
        for label, country, input in loader:
            if model.country_pred:
                pred_val, country_pred = model(input)
            else:
                pred_val = model(input)
            if pred_val.is_cuda:
                pred_val = pred_val.detach().cpu()
            pred_val_list.append(pred_val)
    pred_array = np.vstack(pred_val_list)

    correct_country = []
    correct_code = []
    correct_adm1 = []
    correct_geoid = []
    dists = []
    total_missing = 0
    missing_correct = 0
    for ent, pred in zip(es_data, pred_array):
        if not ent['es_choices']:
            continue
        for n, score in enumerate(pred):
            #score = i[0] # accounting for country prediction
            if n < len(ent['es_choices']):
                ent['es_choices'][n]['score'] = score
        try:
            correct_position = np.where(ent['correct'])[0][0]
        except:
            correct_position = None
        predicted_position = np.argmax([i['score'] for i in ent['es_choices'] if 'score' in i.keys()])
        #if len(correct_position) == 0 and np.sum(ent['correct']) == 0:
        #    correct_country.append(True)
        #    correct_code.append(True)
        #    correct_adm1.append(True)
        #    correct_geoid.append(True)
        #    continue
        # give credit for picking the last position (the "no match" option)
        if correct_position == len(ent['es_choices'])-1 and predicted_position == len(ent['es_choices'])-1:
            total_missing += 1
            missing_correct += 1
            continue
        elif correct_position == None and predicted_position == len(ent['es_choices'])-1:
            total_missing += 1
            missing_correct += 1
            continue
        elif correct_position is None:
            total_missing += 1
            #missing_correct += 0
            continue
        gold_country = ent['es_choices'][correct_position]['country_code3']
        gold_code = ent['es_choices'][correct_position]['feature_code']
        gold_adm1 = ent['es_choices'][correct_position]['admin1_code']
        gold_lat = ent['es_choices'][correct_position]['lat']
        gold_lon = ent['es_choices'][correct_position]['lon']
        predicted_country = ent['es_choices'][predicted_position]['country_code3']
        predicted_code = ent['es_choices'][predicted_position]['feature_code']
        predicted_adm1 = ent['es_choices'][predicted_position]['admin1_code']
        predicted_geoid = ent['es_choices'][predicted_position]['geonameid']
        predicted_lat = ent['es_choices'][predicted_position]['lat']
        predicted_lon = ent['es_choices'][predicted_position]['lon']
        correct_country.append(gold_country == predicted_country)
        correct_code.append(gold_code == predicted_code)
        correct_adm1.append(gold_adm1 == predicted_adm1)
        correct_geoid.append(ent['correct_geonamesid'] == predicted_geoid)
        dist = hs.haversine((gold_lat, gold_lon), (predicted_lat, predicted_lon))
        dists.append(dist)
    if total_missing > 0:
        miss_correct_perc = missing_correct / total_missing
    else:
        miss_correct_perc = 0
    correct_avg = {"correct_country": np.mean(correct_country), 
              "correct_code": np.mean(correct_code),
              "correct_adm1": np.mean(correct_adm1), 
              "exact_match": np.mean(correct_geoid),
              "avg_dist": np.mean(dists),
              "median_dist": np.median(dists),
              "missing_correct": miss_correct_perc,
              "total_missing": total_missing / len(es_data),
              "acc_at_161": np.mean([i <= 161 for i in dists])
    }
    return correct_avg

def make_wandb_dict(names, datasets, data_loaders, model):
    results = {}
    country_avg = 0
    feature_code_avg = 0
    adm1_avg = 0
    exact_match_avg = 0
    dist_avg = 0
    acc_at_161 = 0

    for nn, data, loader in zip(names, datasets, data_loaders): 
        correct_avg = evaluate_results(data, loader, model)
        results[f"{nn}_country_acc"] = correct_avg['correct_country'] 
        country_avg += correct_avg['correct_country']  
        results[f"{nn}_feature_code"] = correct_avg['correct_code'] 
        feature_code_avg += correct_avg['correct_code']
        results[f"{nn}_adm1"] = correct_avg['correct_adm1']
        adm1_avg += correct_avg['correct_adm1']
        results[f"{nn}_exact_match"] = correct_avg['exact_match']
        exact_match_avg += correct_avg['exact_match']
        results[f"{nn}_avg_dist"] = correct_avg['avg_dist']
        dist_avg += correct_avg['avg_dist']
        results[f"{nn}_acc_at_161"] = correct_avg['acc_at_161']
        acc_at_161 += correct_avg['acc_at_161']

    results['country_avg'] = country_avg / len(names)
    results['feature_code_avg'] = feature_code_avg / len(names)
    results['adm1_avg'] = adm1_avg / len(names)
    results['exact_match_avg'] = exact_match_avg / len(names)
    results['dist_avg'] = dist_avg / len(names)
    results['acc_at_161'] = acc_at_161 / len(names)
    return results