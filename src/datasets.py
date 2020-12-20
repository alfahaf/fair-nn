# This code is adapted from github.com/erikbern/ann-benchmarks.

import numpy
import os
import random
import sys
import pickle
from distance import l2, jaccard

try:
        from urllib import urlretrieve
        from urllib import urlopen
except ImportError:
        from urllib.request import urlretrieve
        from urllib.request import urlopen

def download(src, dst):
    if not os.path.exists(dst):
        # TODO: should be atomic
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst)


def get_dataset_fn(dataset):
    if not os.path.exists('data'):
        os.mkdir('data')
    return os.path.join('data', '%s.pickle' % dataset)

def get_dataset(which):
    fn = get_dataset_fn(which)
    if not os.path.exists(fn):
        try:
            url = 'http://itu.dk/people/maau/fairnn/datasets/%s.pickle' % which
            download(url, fn)
        except:
            print("Cannot download %s" % url)
            if which in DATASETS:
                print("Creating dataset locally")
                DATASETS[which](fn)
    with open(fn, 'rb') as f:
        data, queries, ground_truth, attrs = pickle.load(f)
    return data, queries, ground_truth, attrs 

# Everything below this line is related to creating datasets
def bruteforce(X, Y, f, stop=lambda x: False, max_cnt=1e10):
    distances = [[] for _ in range(len(Y))] 
    cnt = 0
    for i, y in enumerate(Y):
        for j, x in enumerate(X):
            if f(x, y):
                distances[i].append(j)
        cnt += stop(distances[i]) 
        if cnt >= max_cnt: 
            break
    return distances

def find_interesting_queries(data, dist_threshold, 
        distance, test_size, query_size, num_neighbors):
    print("Running brute-force scan")

    random.shuffle(data)
    if distance == l2:
        distances = bruteforce(data, data, 
            lambda x, y: l2(x, y) <= dist_threshold,
            lambda x: len(x) >= num_neighbors,
            10 * query_size) 
    if distance == jaccard:
        distances = bruteforce(data, data, 
            lambda x, y: jaccard(x, y) >= dist_threshold,
            lambda x: len(x) >= num_neighbors,
            10 * query_size)
    
    distances = [(i, l) for i, l in enumerate(distances) if len(l) >= num_neighbors]
    random.shuffle(distances)

    assert len(distances) >= query_size 

    distances = distances[:query_size]
    distances_idx = set([i for i, _ in distances])
    X = [data[i] for i in range(len(data)) if i not in distances_idx]
    Y = [data[i] for i in distances_idx]

    assert(len(X)) >= test_size

    random.shuffle(X)

    if distance == l2:
        distances = bruteforce(X, Y, lambda x, y: l2(x, y) <= dist_threshold) 
    if distance == jaccard:
        distances = bruteforce(X, Y, lambda x, y: jaccard(x,y) >= dist_threshold)

    print("Done")

    return X, Y, distances
    
def write_output(X, Y, ground_truth, fn, attrs):
    with open(fn, 'wb') as f:
        pickle.dump([X, Y, ground_truth, attrs], f, pickle.HIGHEST_PROTOCOL)


def glove(out_fn, d):
    import zipfile

    url = 'http://nlp.stanford.edu/data/glove.twitter.27B.zip'
    fn = os.path.join('data', 'glove.twitter.27B.zip')
    download(url, fn)

    attrs = {
        "dist_threshold": 5.1,
        "queries": 50,
        "n": 10_000,
    }

    with zipfile.ZipFile(fn) as z:
        print('preparing %s' % out_fn)
        z_fn = 'glove.twitter.27B.%dd.txt' % d
        X = []
        for line in z.open(z_fn):
            v = [float(x) for x in line.strip().split()[1:]]
            X.append(numpy.array(v))
        
        X, Y, groundtruth = find_interesting_queries(
            random.sample(X, k=attrs["n"] + 6 * attrs["queries"]), 
            dist_threshold=attrs["dist_threshold"], 
            distance=l2, 
            test_size=attrs["n"], 
            query_size=attrs["queries"],
            num_neighbors=40)
        write_output(numpy.array(X), numpy.array(Y), groundtruth, out_fn, attrs)

def _load_texmex_vectors(f, n, k):
    import struct

    v = numpy.zeros((n, k))
    for i in range(n):
        f.read(4)  # ignore vec length
        v[i] = struct.unpack('f' * k, f.read(k * 4))

    return v


def _get_irisa_matrix(t, fn):
    import struct
    m = t.getmember(fn)
    f = t.extractfile(m)
    k, = struct.unpack('i', f.read(4))
    n = m.size // (4 + 4 * k)
    f.seek(0)
    return _load_texmex_vectors(f, n, k)


def sift(out_fn):
    import tarfile
    from sklearn.model_selection import train_test_split

    attrs = {
        "dist_threshold": 300,
        "queries": 50,
        "n": 10000
    }

    url = 'ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz'
    fn = os.path.join('data', 'sift.tar.tz')
    download(url, fn)
    with tarfile.open(fn, 'r:gz') as t:
        train = _get_irisa_matrix(t, 'sift/sift_base.fvecs')
    
    data, queries, groundtruth = find_interesting_queries(
                        random.sample(list(train), k=attrs["n"] + 6 * attrs["queries"]), 
                        query_size=attrs["queries"], 
                        dist_threshold=attrs["dist_threshold"], 
                        distance=l2, test_size=attrs["n"],
                        num_neighbors=40)
    write_output(data, queries, groundtruth, out_fn, attrs)


def _load_mnist_vectors(fn):
    import gzip
    import struct

    print('parsing vectors in %s...' % fn)
    f = gzip.open(fn)
    type_code_info = {
        0x08: (1, "!B"),
        0x09: (1, "!b"),
        0x0B: (2, "!H"),
        0x0C: (4, "!I"),
        0x0D: (4, "!f"),
        0x0E: (8, "!d")
    }
    magic, type_code, dim_count = struct.unpack("!hBB", f.read(4))
    assert magic == 0
    assert type_code in type_code_info

    dimensions = [struct.unpack("!I", f.read(4))[0]
                  for i in range(dim_count)]

    entry_count = dimensions[0]
    entry_size = numpy.product(dimensions[1:])

    b, format_string = type_code_info[type_code]
    vectors = []
    for i in range(entry_count):
        vectors.append([struct.unpack(format_string, f.read(b))[0]
                        for j in range(entry_size)])
    return numpy.array(vectors)


def mnist(out_fn):
    from sklearn.model_selection import train_test_split
    download(
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 'mnist-train.gz')  # noqa

    
    attrs = {
        "dist_threshold": 1500,
        "queries": 50,
        "n": 10000,
    }
    train = _load_mnist_vectors('mnist-train.gz')

    data, queries, groundtruth = find_interesting_queries(
                        random.sample(list(train), k=attrs["n"] + 6 * attrs["queries"]),
                        query_size=attrs["queries"], 
                        dist_threshold=attrs["dist_threshold"], 
                        distance=l2, test_size=attrs["n"],
                        num_neighbors=40)
    write_output(data, queries, groundtruth, out_fn, attrs)

def lastfm(out_fn):
    import zipfile 

    fn = os.path.join('data', 'lastfm.zip')
    download('http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip', fn) # noqa

    attrs = {
        "dist_threshold": 0.15,
        "queries": 50,
    }

    with zipfile.ZipFile(fn) as z:
        data = {}
        lines = z.open('user_artists.dat').readlines()[1:]

        for line in lines:
            user, artist, weight = map(int, line.decode().split("\t"))
            data.setdefault(user, []).append((artist, weight))
    for user in data:
        data[user] = set([x for x, i in sorted(data[user], key=lambda k: -k[1])[:20]])
    data_as_list = list(data.values())

    attrs["n"] = len(data_as_list) - attrs["queries"]

    data, queries, groundtruth = find_interesting_queries(data_as_list, 
                        query_size=attrs["queries"], 
                        dist_threshold=attrs["dist_threshold"], 
                        distance=jaccard, test_size=attrs["n"],
                        num_neighbors=40)
    write_output(data, queries, groundtruth, out_fn, attrs)

def movielens(out_fn):
    import zipfile 

    fn = os.path.join('data', 'movielens.zip')
    download('http://files.grouplens.org/datasets/hetrec2011/hetrec2011-movielens-2k-v2.zip', fn) # noqa

    attrs = {
        "dist_threshold": 0.2,
        "queries": 50,
    }

    with zipfile.ZipFile(fn) as z:
        data = {}
        lines = z.open('user_ratedmovies.dat').readlines()[1:]
        for line in lines:
            user, movie, rating = line.decode().split("\t")[:3]
            user = int(user)
            movie = int(movie)
            rating = float(rating)
            if rating >= 4:
                data.setdefault(user, set()).add(movie)
    
    data_as_list = [v for _, v in data.items()]

    attrs["n"] = len(data_as_list) - attrs["queries"]

    data, queries, groundtruth = find_interesting_queries(data_as_list, 
                        query_size=attrs["queries"], 
                        dist_threshold=attrs["dist_threshold"], 
                        distance=jaccard, test_size=attrs["n"],
                        num_neighbors=40)
    write_output(data, queries, groundtruth, out_fn, attrs)

def random_jaccard(out_fn, test_size=100, query_size=2):
    attrs = {
        "n": test_size,
        "queries": query_size,
        "dist_threshold": 0.3,
    }

    data = {i: set(random.choices(list(range(20)), k=10)) for i in range(test_size + query_size)}
    data, queries, groundtruth = find_interesting_queries(data, 
        query_size=query_size, dist_threshold=attrs["dist_threshold"], 
        distance=jaccard, test_size=attrs["queries"], num_neighbors=10)
    write_output(data, queries, groundtruth, out_fn, attrs)

def random_approximate(out_fn):
    import itertools
    l = list(range(30))
    nn = list(range(27))
    l1 = list(range(18))
    l2 = list(range(15, 30))

    data = [set(x) for x in list(itertools.combinations(l1, 17)) + 
            list(itertools.combinations(l1, 16)) + 
            list(itertools.combinations(l1, 15))]

    data.extend([set(nn), set(l1), set(l2)])

    query = [set(l)]

    attrs = {
        "n": len(data),
        "queries" : len(query),
        "dist_threshold": 0.3
    }

    write_output(data, query, bruteforce(data, query, jaccard), out_fn, attrs)




DATASETS = {
    'glove-100-angular': lambda out_fn: glove(out_fn, 100),
    'mnist-784-euclidean': mnist,
    'sift-128-euclidean': sift,
    'lastfm': lastfm,
    'movielens': movielens,
    'random-jaccard': lambda out_fn: random_jaccard(out_fn, 100, 10),
    'random-approximate': lambda out_fn: random_approximate(out_fn)
}

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        required=True)
    args = parser.parse_args()
    fn = get_dataset_fn(args.dataset)
    DATASETS[args.dataset](fn)
