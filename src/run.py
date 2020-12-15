import sys
import argparse
import yaml
import pickle
import os
from lsh import LSHBuilder
from datasets import get_dataset, DATASETS
from distance import l2, jaccard
from helpers import get_result_fn
from main import run_single_exp
from timeit import default_timer

def get_experiments(data, exp_file):
    params = {}
    for k, L in exp_file['combs']:
        for method in LSHBuilder.methods:
            lsh = LSHBuilder.build(len(data[0]),
                exp_file['dist_threshold'], k, L, exp_file['lsh'], validate)
            res_fn = get_result_fn(exp_file['dataset'],
                exp_file['lsh']['type'], method, repr(lsh))
            if os.path.exists(res_fn) and not args.force:
                print(f"{res_fn} exists, skipping.")
            else:
                params.setdefault((k, L), []).append(method)
    return params


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

    with open(args.exp_file) as f:
        exp_file = yaml.load(f)

    data, queries, ground_truth, attrs = get_dataset(exp_file['dataset'])

    validate = True
    if "validate" in exp_file:
        validate = exp_file["validate"]

    params = get_experiments(data, exp_file)

    for k, L in params.keys():
        run_single_exp(exp_file['dataset'], exp_file['dist_threshold'],
                exp_file['lsh']['type'], k, L,
                exp_file['lsh'].get('w', 0), validate, False, exp_file['runs'])

