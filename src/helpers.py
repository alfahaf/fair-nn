import os
import pickle

def get_result_fn(dataset, lsh, query_type, rep):
    if not os.path.exists(os.path.join("results", dataset, query_type, lsh)):
        os.makedirs(os.path.join("results", dataset, query_type, lsh))
    return os.path.join(os.path.join("results", dataset, query_type, lsh, rep + ".pickle"))

def load_dataset_results(dataset):
    for (dirpath, _, fns) in os.walk(os.path.join("results", dataset)):
        for fn in fns:
            try:
                with open(os.path.join(dirpath, fn), 'rb') as f:
                    yield pickle.load(f)
            except:
                pass

def load_all_results():
    datasets = [d for d in os.listdir("results") if
        os.path.isdir(os.path.join("results", d))]
    for dataset in datasets:
        yield from load_dataset_results(dataset)

if __name__ == "__main__":
    print(load_all_results())
