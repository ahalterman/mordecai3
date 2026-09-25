"""Held-out scoring for a trained ranker.

`evaluate_results` is the campaign's frozen metric and is deliberately
unchanged: `exact_match` here is what every number in ACCURACY_CAMPAIGN.md and
every `experiments/e*/seed*.json` means, and a rerun of the e29 recipe must
still reproduce `experiments/e29_swa_ep15/seed42.json` byte for byte.

`campaign2_metrics` is the campaign-2 suite, computed from the same forward
pass under the *unified reserved-row convention*
(mordecai3.geoparse, "The reserved-row convention"; the three conventions it
replaces are catalogued in experiments/campaign2/calibration_report.md §2a):

  * candidate rows are `0 .. min(n_choices, W) - 1` minus row `W - 1` when the
    list fills the window -- so a sentinel's score is never attributed to the
    real candidate it overwrote;
  * the reserved row is always in the softmax denominator and its winning the
    argmax is an ABSTENTION, reported as an explicit third outcome rather than
    silently scored as a wrong answer or silently dropped;
  * three accuracies are reported side by side: conditioned on the gold being
    reachable (the frozen metric's denominator), over every held-out mention,
    and the abstention rate that separates them.

Also here, because they are the campaign-2 scoreboard (decision D1):
TLG-hard's per-source ingredient (exact match on non-country golds),
novel-pair exact match (mentions whose (mention, gold id) pair never occurs in
training), and twin-credit when a gold twin cache is supplied.
"""
import haversine as hs
import numpy as np
import torch

from mordecai3.geoparse import candidate_row_count

#es_data = datasets[2]
#loader = data_loaders[2]
#model = geo.model

def _pred_array(loader, model):
    device = next(model.parameters()).device
    pred_val_list = []
    with torch.no_grad():
        model.eval()
        for label, country, input in loader:
            input = {k: v.to(device, non_blocking=True) for k, v in input.items()}
            if model.country_pred:
                pred_val, country_pred = model(input)
            else:
                pred_val = model(input)
            if pred_val.is_cuda:
                pred_val = pred_val.detach().cpu()
            pred_val_list.append(pred_val)
    return np.vstack(pred_val_list)


def _is_country(feature_code):
    """geonames codes for a country row: PCLI, PCLD, PCLIX, PCLS, PCLF, PCL."""
    return str(feature_code or "").startswith("PCL")


def campaign2_metrics(es_data, pred_array, train_pairs=None, twins=None):
    """The campaign-2 metric suite for one held-out source.

    Parameters
    ----------
    es_data : list of dicts
        One held-out source, as `evaluate_results` takes it.
    pred_array : (N, window) array
        The model's scores for those entities (see `_pred_array`).
    train_pairs : set of (search_name, geonameid) or None
        Every (mention, gold id) pair that occurs in the training half. An
        entity whose pair is absent is a *novel pair*: the only accuracy the
        answer-key memorisation described in campaign2/data_quality_report.md
        cannot move.
    twins : list of frozenset or None
        Per entity, the geonameids in the gold answer's A/P twin class (see
        tools/twin_credit_eval.py). A prediction inside it gets twin credit.

    Returns
    -------
    dict of counts (not rates), so several sources can be pooled by addition.
    """
    n = 0
    n_answerable = 0
    n_correct = 0                # unified: right, on an answerable mention
    n_correct_all = 0            # same numerator, denominator = every mention
    n_abstain = 0
    n_abstain_unanswerable = 0
    n_unanswerable = 0
    n_noncountry = 0
    n_noncountry_correct = 0
    n_correct_noabstain = 0      # same, but an abstention is not charged
    n_noncountry_correct_noabstain = 0
    n_novel = 0
    n_novel_correct = 0
    n_seen = 0
    n_seen_correct = 0
    n_twin = 0
    n_twin_credit = 0
    n_twin_scored = 0
    for i, (ent, pred) in enumerate(zip(es_data, pred_array)):
        choices = ent['es_choices']
        if not choices:
            continue
        window = len(pred)
        k = candidate_row_count(len(choices), window)
        if k == 0:
            continue
        n += 1
        gold_idx = next((j for j, v in enumerate(ent.get('correct') or []) if v),
                        None)
        answerable = gold_idx is not None and gold_idx < k
        cand = pred[:k]
        best = int(np.argmax(cand))
        # The reserved row decides *whether* to answer; the candidate rows
        # decide *which* place.
        abstain = bool(pred[window - 1] >= cand[best])
        if abstain:
            n_abstain += 1
            if not answerable:
                n_abstain_unanswerable += 1
        if not answerable:
            n_unanswerable += 1
        # `picked` is the campaign metric's treatment -- always take the best
        # candidate, the reserved row is not consulted; `correct` is the
        # unified one, where an abstention is an answer the user does not get.
        picked = bool(answerable and
                      str(choices[best].get('geonameid')) ==
                      str(ent['correct_geonamesid']))
        correct = bool(picked and not abstain)
        n_correct_all += correct
        if answerable:
            n_answerable += 1
            n_correct += correct
            n_correct_noabstain += picked
            if not _is_country(choices[gold_idx].get('feature_code')):
                n_noncountry += 1
                n_noncountry_correct += correct
                n_noncountry_correct_noabstain += picked
            if train_pairs is not None:
                pair = (str(ent.get('search_name')),
                        str(ent['correct_geonamesid']))
                if pair in train_pairs:
                    n_seen += 1
                    n_seen_correct += correct
                else:
                    n_novel += 1
                    n_novel_correct += correct
            if twins is not None:
                n_twin_scored += 1
                tw = twins[i] if i < len(twins) else frozenset()
                if tw:
                    n_twin += 1
                credited = correct or (
                    not abstain and str(choices[best].get('geonameid')) in tw)
                n_twin_credit += bool(credited)
    return dict(n=n, n_answerable=n_answerable, n_correct=n_correct,
                n_correct_all=n_correct_all, n_abstain=n_abstain,
                n_abstain_unanswerable=n_abstain_unanswerable,
                n_unanswerable=n_unanswerable,
                n_correct_noabstain=n_correct_noabstain,
                n_noncountry=n_noncountry,
                n_noncountry_correct=n_noncountry_correct,
                n_noncountry_correct_noabstain=n_noncountry_correct_noabstain,
                n_novel=n_novel, n_novel_correct=n_novel_correct,
                n_seen=n_seen, n_seen_correct=n_seen_correct,
                n_twin=n_twin, n_twin_credit=n_twin_credit,
                n_twin_scored=n_twin_scored)


def _rate(a, b):
    return float(a) / b if b else float("nan")


def campaign2_report(per_source, headline_sources=("TR", "LGL", "GWN"),
                     drop_from_macro=("Synth",)):
    """Aggregate `campaign2_metrics` counts into the campaign-2 scoreboard.

    `headline_sources` are the three human-annotated news corpora TLG-hard is
    the macro over. `drop_from_macro` is decision D4: Synth stays in the
    training mix and leaves the headline, because its held-out half is 100%
    gazetteer lookups and its split is entity-shuffled.
    """
    def macro(names, num, den):
        vals = [_rate(per_source[s][num], per_source[s][den])
                for s in names if s in per_source and per_source[s][den]]
        return float(np.mean(vals)) if vals else float("nan")

    names = list(per_source)
    keep = [s for s in names if s not in drop_from_macro]
    tlg = [s for s in headline_sources if s in per_source]
    tot = {k: sum(v[k] for v in per_source.values()) for k in
           next(iter(per_source.values()))} if per_source else {}
    return {
        # ---- primary (D1)
        "tlg_hard": macro(tlg, "n_noncountry_correct", "n_noncountry"),
        "tlg_hard_n": sum(per_source[s]["n_noncountry"] for s in tlg),
        # The same metric under the campaign's convention -- abstentions are
        # not charged, because `evaluate_results` never saw them. The gap
        # between the two is what adopting the unified convention costs on
        # paper; it is error that was always there and was never counted.
        "tlg_hard_noabstain": macro(tlg, "n_noncountry_correct_noabstain",
                                    "n_noncountry"),
        "em_conditioned_macro_noabstain": macro(keep, "n_correct_noabstain",
                                                "n_answerable"),
        "em_conditioned_macro_legacy6_noabstain": macro(
            names, "n_correct_noabstain", "n_answerable"),
        # ---- guardrail
        "novel_pair_em": _rate(tot.get("n_novel_correct", 0),
                               tot.get("n_novel", 0)),
        "novel_pair_n": tot.get("n_novel", 0),
        "seen_pair_em": _rate(tot.get("n_seen_correct", 0),
                              tot.get("n_seen", 0)),
        # ---- secondary
        "twin_credit_macro": macro(keep, "n_twin_credit", "n_twin_scored"),
        # ---- the three accuracies (S1)
        "em_conditioned_macro": macro(keep, "n_correct", "n_answerable"),
        "em_all_macro": macro(keep, "n_correct_all", "n"),
        "em_conditioned_macro_legacy6": macro(names, "n_correct",
                                              "n_answerable"),
        "em_conditioned_pooled": _rate(tot.get("n_correct", 0),
                                       tot.get("n_answerable", 0)),
        "em_all_pooled": _rate(tot.get("n_correct_all", 0), tot.get("n", 0)),
        "abstain_rate": _rate(tot.get("n_abstain", 0), tot.get("n", 0)),
        "abstain_precision": _rate(tot.get("n_abstain_unanswerable", 0),
                                   tot.get("n_abstain", 0)),
        "unanswerable_rate": _rate(tot.get("n_unanswerable", 0),
                                   tot.get("n", 0)),
        "per_source": {s: {
            "n": v["n"], "n_answerable": v["n_answerable"],
            "em_conditioned": _rate(v["n_correct"], v["n_answerable"]),
            "em_all": _rate(v["n_correct_all"], v["n"]),
            "em_conditioned_noabstain": _rate(v["n_correct_noabstain"],
                                              v["n_answerable"]),
            "em_noncountry": _rate(v["n_noncountry_correct"], v["n_noncountry"]),
            "em_noncountry_noabstain": _rate(v["n_noncountry_correct_noabstain"],
                                             v["n_noncountry"]),
            "n_noncountry": v["n_noncountry"],
            "novel_pair_em": _rate(v["n_novel_correct"], v["n_novel"]),
            "n_novel": v["n_novel"],
            "twin_credit": _rate(v["n_twin_credit"], v["n_twin_scored"]),
            "abstain_rate": _rate(v["n_abstain"], v["n"]),
            "unanswerable_rate": _rate(v["n_unanswerable"], v["n"]),
        } for s, v in per_source.items()},
    }


def evaluate_results(es_data, loader, model, pred_array=None):
    """The frozen campaign metric. Do not change what it computes.

    `pred_array` lets a caller that already ran the forward pass hand the
    scores in rather than paying for a second one; it is the same array
    `_pred_array` would return.
    """
    if pred_array is None:
        pred_array = _pred_array(loader, model)

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

    # NOTE: nothing may be added to `results` -- it is dumped verbatim into
    # every experiments/e*/seed*.json and those files are the ledger. The
    # campaign-2 suite is `make_campaign2_dict`, written to its own sidecar.
    results['country_avg'] = country_avg / len(names)
    results['feature_code_avg'] = feature_code_avg / len(names)
    results['adm1_avg'] = adm1_avg / len(names)
    results['exact_match_avg'] = exact_match_avg / len(names)
    results['dist_avg'] = dist_avg / len(names)
    results['acc_at_161'] = acc_at_161 / len(names)
    return results


def make_campaign2_dict(names, datasets, data_loaders, model,
                        train_pairs=None, twins=None):
    """The campaign-2 scoreboard for a trained model, one forward pass.

    `twins` is {source name: [frozenset of gids per held-out entity]}, e.g.
    from `tools/twin_credit_eval.py twin-cache`; sources missing from it get no
    twin-credit number.
    """
    per_source = {}
    for nn, data, loader in zip(names, datasets, data_loaders):
        per_source[nn] = campaign2_metrics(
            data, _pred_array(loader, model), train_pairs=train_pairs,
            twins=(twins or {}).get(nn))
    return campaign2_report(per_source)