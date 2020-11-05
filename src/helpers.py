import os 

def get_result_fn(dataset, lsh, rep):
    if not os.path.exists(os.path.join("results", dataset, lsh)):
        os.makedirs(os.path.join("results", dataset, lsh))
    return os.path.join(os.path.join("results", dataset, lsh, rep))

