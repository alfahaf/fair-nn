import sys
import argparse
import yaml
import pickle
import os
from lsh import LSHBuilder
from datasets import get_dataset, DATASETS
from distance import l2, jaccard
from helpers import get_result_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        default="mnist-784-euclidean",
        )
    parser.add_argument(
        '--seed',
        default=3,
        type=int,
    )
    parser.add_argument(
        '--exp-file',
        required=True,
    )
    parser.add_argument(
        '--force',
        action='store_true',
    )
    args = parser.parse_args()

    data, queries, ground_truth = get_dataset(args.dataset)
    with open(args.exp_file) as f:
        exp_file = yaml.load(f)

    data, queries, ground_truth = get_dataset(exp_file['dataset'])

    params = {}

    for k in exp_file['k']:
        for L in exp_file['L']:
            for method in ["opt", "uniform", "weighted_uniform"]:
                lsh = LSHBuilder.build(data.shape[1], 
                    exp_file['dist_threshold'], k, L, exp_file['lsh'])
                res_fn = get_result_fn(exp_file['dataset'], 
                    exp_file['lsh']['type'], method, repr(lsh)) 
                if os.path.exists(res_fn) and not args.force:
                    print(f"{res_fn} exists, skipping.")
                else:
                    params.setdefault((k, L), [])
                    params[(k, L)].append(method)

    for k, L in params.keys():
        lsh = LSHBuilder.build(data.shape[1], 
            exp_file['dist_threshold'], k, L, exp_file['lsh'])

        lsh.preprocess(data)

        for method in params[(k, L)]:
            print(f"Running (k={k}, L={L}) with {method}")
            res_fn = get_result_fn(exp_file['dataset'], 
                exp_file['lsh']['type'], method, repr(lsh)) 
            if os.path.exists(res_fn) and not args.force:
                print(f"{res_fn} exists, skipping.")
                continue

            if method == "opt":
                res = lsh.opt(queries, exp_file['runs'])
            if method == "uniform":
                res = lsh.uniform_query(queries, exp_file['runs'])
            if method == "weighted_uniform":
                res = lsh.weighted_uniform_query(queries, exp_file['runs'])

            res_dict = {
                "name": str(lsh),
                "method" : method,
                "res": res,
                "dataset": exp_file['dataset'],
                "dist_threshold": exp_file['dist_threshold'],
            }

            with open(res_fn, 'wb') as f:
                pickle.dump(res_dict, f, pickle.HIGHEST_PROTOCOL)








