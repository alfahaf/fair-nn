import sys
import argparse
from lsh import E2LSH
from datasets import get_dataset, DATASETS
from distance import l2

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
    args = parser.parse_args()
    data, queries, ground_truth = get_dataset(args.dataset)

    d = data.shape[1]
    n = data.shape[0]
    w = 1700 
    k = 9 
    L = 300
    r = 1500 
    runs = 1
    lsh = E2LSH(k, L, w, d, r, seed=args.seed)
    lsh.preprocess(data)
    results = []
    for i in range(runs):
        results.append(lsh.opt(queries))
        print(len(list(filter(lambda x: x[1] != -1, results[-1]))))
