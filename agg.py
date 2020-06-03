import os
import json
import numpy as np


AGG_NAME   = 'Aggregate (All)'
EXPLAINERS = ['Greedy', 'LIME', 'SHAP']
MODELS     = ['logistic', 'dt', 'rf', 'mlp']
DATASETS   = ['amazon_home', 'amazon_cell', 'imdb', 'newsgroups_atheism',
              'newsgroups_baseball', 'newsgroups_ibm']
BIAS_LENS  = [2]
BUDGETS    = range(1, 6)
SEEDS      = range(1)
IDS        = range(50)
LOG_DIR    = 'logs/boost_test'



def aggregate(model, dataset, bias_len, budget, seed, example):
    bias_words = None
    features = []
    importances = []

    for exp in EXPLAINERS:
        filename = '{:s}_{:1d}_{:03d}_{:02d}_{:03d}.json'.format(
                    exp, bias_len, seed, budget, example)
        path = os.path.join(LOG_DIR, dataset, model, filename)
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except:
            return

        bias  = data['bias_words']
        feats = data['top_features']
        imps  = data['feature_importances']

        if bias_words is None:
            bias_words = bias

        for i, feat in enumerate(feats):
            if feat in features:
                importances[features.index(feat)] += imps[i]
            else:
                features.append(feat)
                importances.append(imps[i])

    # Normalize, sort new importances, and limit to budget
    importances = [i / len(EXPLAINERS) for i in importances]
    pairs = list(zip(features, importances))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    pairs = pairs[:budget]

    top_feats, top_imps = zip(*pairs)

    # Compute recall
    recall = 0
    for word in bias_words:
        if word in top_feats:
            recall += 1
    recall /= bias_len

    # Create new log entry and dump
    log_data = dict()
    log_data['seed'] = seed
    log_data['dataset'] = dataset
    log_data['model_type'] = model
    log_data['bias_len'] = bias_len
    log_data['bias_words'] = bias_words
    log_data['top_features'] = list(top_feats)
    log_data['feature_importances'] = list(top_imps)
    log_data['explainer'] = AGG_NAME
    log_data['budget'] = budget
    log_data['recall'] = recall

    res_filename = '{:s}_{:1d}_{:03d}_{:02d}.json'.format(
                    AGG_NAME, bias_len, seed, budget)
    res_path = os.path.join(LOG_DIR, dataset, model, res_filename)
    print('Writing {}'.format(res_path))

    with open(res_path, 'w') as f:
        json.dump(log_data, f)


def main():
    for model in MODELS:
        for dataset in DATASETS:
            for bias_len in BIAS_LENS:
                for budget in BUDGETS:
                    for seed in SEEDS:
                        for example in IDS:
                            aggregate(model, dataset, bias_len, budget, seed,
                                    example)




if __name__ == "__main__":
    main()
