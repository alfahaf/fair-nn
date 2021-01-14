from datasets import get_dataset
from lsh import LSHBuilder

def run(fn):
    data, queries, _, _ = get_dataset("random-approximate")

    RUNS = 10000

    counter = {
        987: 0,
        988: 0,
        989: 0,
    }

    for i in range(RUNS):
        if i % 100 == 0:
            print(f"{i}/{RUNS}")
        lsh = LSHBuilder.build(len(data[0]), 0.5, 5, 3, {"type" : "onebitminhash"}, False)
        lsh.preprocess(data)
        res = lsh.opt(queries, 1, False)[0]
        if len(res) > 0:
            res = res[0]
            if res in counter:
                counter[res] += 1
    with open(fn, "w") as f:
        f.write("point,count,runs\n")
        for point, count in counter.items():
            f.write(f"{point},{count},{RUNS}\n") 

if __name__ == "__main__":
    run("approx.csv")