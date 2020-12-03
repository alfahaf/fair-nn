import numpy
import re
from helpers import load_all_results
from metrics import result_prob, total_variation_dist


def get_query_prob(result):
    groundtruth = numpy.array(result["candidates"])
    return 1/groundtruth


if __name__ == "__main__":
    print("dataset,k,L,method,params,tvd,outputsize")
    for result in load_all_results():
        ds = result["dataset"]
        k = int(re.search(r"k=(\d+)", result["name"]).group(1))
        L = int(re.search(r"L=(\d+)", result["name"]).group(1))
        method = result["method"]
        q = get_query_prob(result)
        tvd = total_variation_dist(get_query_prob(result),
                                   result_prob(result["res"]))
        print(f"{ds},{k},{L},{method},\"{result['name']}\", {tvd}, {numpy.mean(1/q)}")

