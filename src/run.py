import sys
import argparse
import yaml
import pickle
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
    args = parser.parse_args()
    data, queries, ground_truth = get_dataset(args.dataset)
    with open(args.exp_file) as f:
        exp_file = yaml.load(f)

    data, queries, ground_truth = get_dataset(exp_file['dataset'])
    for k in exp_file['k']:
        for L in exp_file['L']:
            lsh = LSHBuilder.build(data.shape[1], 
                exp_file['dist_threshold'], k, L, exp_file['lsh'])

            lsh.preprocess(data)

            res_fn = get_result_fn(exp_file['dataset'], exp_file['lsh']['type'], repr(lsh)) 

            res = lsh.opt(queries, exp_file['runs'])

            res_dict = {
                "name" : str(lsh),
                "res" : res
            }

            with open(res_fn, 'wb') as f:
                pickle.dump(res_dict, f, pickle.HIGHEST_PROTOCOL)








