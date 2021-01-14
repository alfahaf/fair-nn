import argparse
import numpy
import re
from helpers import load_all_results
from metrics import result_prob, total_variation_dist


def get_query_prob(result):
    groundtruth = numpy.array(result["candidates"])
    return 1/groundtruth

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--detail',
        action='store_true',
        help='present the id of all the queries, one per line'
    )
    args = parser.parse_args()

    if args.detail:
        print("dataset,k,L,method,params,query,point,prob")
    else:
        print("dataset,k,L,method,params,tvd,outputsize")
    for result in load_all_results():
        ds = result["dataset"]
        k = int(re.search(r"k=(\d+)", result["name"]).group(1))
        L = int(re.search(r"L=(\d+)", result["name"]).group(1))
        method = result["method"]
        if args.detail:
            for q, probs in result_prob(result["res"]).items():
                for point, prob in probs:
                    print(f"{ds},{k},{L},{method},\"{result['name']}\",{q},{point},{prob}")
        else:
            q = get_query_prob(result)
            tvd, _ = total_variation_dist(get_query_prob(result), result_prob(result["res"]))
            print(f"{ds},{k},{L},{method},\"{result['name']}\", {tvd}, {numpy.mean(1/q)}")

