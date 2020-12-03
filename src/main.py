import argparse
import pickle
import time
from datasets import get_dataset, DATASETS
from helpers import get_result_fn
from lsh import LSHBuilder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        default="mnist-784-euclidean",
        )
    parser.add_argument(
        '-k',
        default=10,
        type=int,
    )
    parser.add_argument(
        '-L',
        default=50,
        type=int
    )
    parser.add_argument(
        '-w',
        default=4,
        type=float
    )
    parser.add_argument(
        '--distance-threshold',
        default=1,
        type=float,
    )
    parser.add_argument(
        '--method',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--runs',
        type=int,
        default=5,
    )

    parser.add_argument(
        '--report-output',
        action='store_true',
    )

    parser.add_argument(
        '--validate',
        action='store_true',
    )

    args = parser.parse_args()

    data, queries, ground_truth, attrs = get_dataset(args.dataset)

    lsh = LSHBuilder.build(len(data[0]), args.distance_threshold,
        args.k, args.L, {"type": args.method, "w": args.w}, args.validate)

    print("Building index...")
    lsh.preprocess(data)

    candidates = lsh.get_query_size(queries)

    if args.report_output:
        print(candidates)
        exit(0)

    for method in LSHBuilder.methods:
        print(f"Running (k={args.k}, L={args.L}) with {method}")
        start = time.time()
        res = LSHBuilder.invoke(lsh, method, queries, args.runs)
        print(f"Run finished in {time.time() - start}s.")

        res_fn = get_result_fn(args.dataset,
            args.method, method, repr(lsh))

        res_dict = {
            "name": str(lsh),
            "method" : method,
            "res": res,
            "dataset": args.dataset,
            "dist_threshold": args.distance_threshold,
            "candidates": candidates,
        }

        with open(res_fn, 'wb') as f:
            pickle.dump(res_dict, f, pickle.HIGHEST_PROTOCOL)





