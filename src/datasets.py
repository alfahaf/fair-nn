# This code is adapted from github.com/erikbern/ann-benchmarks.

import numpy
import os
import random
import sys
import pickle
from distance import l2, jaccard
from sklearn.model_selection import train_test_split

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
    return os.path.join('data', '%s.txt' % dataset)
           

def get_dataset(which):
    fn = get_dataset_fn(which)
    try:
        url = 'http://todo/%s.hdf5' % which
        download(url, hdf5_fn)
    except:
        print("Cannot download %s" % url)
        if which in DATASETS:
            print("Creating dataset locally")
            DATASETS[which](fn)
    with open(fn, 'rb') as f:
        data, queries, ground_truth = pickle.load(f)
    return data, queries, ground_truth 


# Everything below this line is related to creating datasets

def write_output(X, fn, r, dist=l2, queries=200):
    def bruteforce(X, Y, f):
        distances = [[] for _ in range(len(Y))] 
        for i, y in enumerate(Y):
            for j, x in enumerate(X):
                if f(x, y):
                    distances[i].append(j)
        return distances

    print('Splitting dataset')
    X, Y = train_test_split(X, test_size=queries, random_state=4)

    ground_truth = bruteforce(X, Y, lambda x, y: l2(x, y) <= r)
    # just get the ones where there are neighbors
    ground_truth_filtered = [(i, g) for i, g in enumerate(ground_truth) if len(g) > 0 ]
    
    print(len(ground_truth_filtered))

    ground_truth = []
    Ys = []

    for i, g in random.choices(ground_truth_filtered, k=100):
        ground_truth.append(g)
        Ys.append(Y[i])

    Y = numpy.array(Ys)
    

    print(f"Avg true neighbors: {sum([len(x) for x in ground_truth]) / queries}")
    print(f"No near neighbors: {sum([1 for x in ground_truth if len(x) == 0])}")

    with open(fn, 'wb') as f:
        pickle.dump([X, Y, ground_truth], f, pickle.HIGHEST_PROTOCOL)


def glove(out_fn, d):
    import zipfile

    url = 'http://nlp.stanford.edu/data/glove.twitter.27B.zip'
    fn = os.path.join('data', 'glove.twitter.27B.zip')
    download(url, fn)
    with zipfile.ZipFile(fn) as z:
        print('preparing %s' % out_fn)
        z_fn = 'glove.twitter.27B.%dd.txt' % d
        X = []
        for line in z.open(z_fn):
            v = [float(x) for x in line.strip().split()[1:]]
            X.append(numpy.array(v))
        write_output(numpy.array(random.choices(X, k=10100)), out_fn, 0.9)

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

    url = 'ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz'
    fn = os.path.join('data', 'sift.tar.tz')
    download(url, fn)
    with tarfile.open(fn, 'r:gz') as t:
        train = _get_irisa_matrix(t, 'sift/sift_base.fvecs')
        train, _ = train_test_split(train, train_size=10100, random_state=4)
        write_output(train, out_fn, 255)


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
    download(
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 'mnist-train.gz')  # noqa
    download(
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', 'mnist-test.gz')  # noqa
    train, _ = train_test_split(_load_mnist_vectors('mnist-train.gz'), train_size=10200)

    write_output(train, out_fn, 1500)

DATASETS = {
    'glove-100-angular': lambda out_fn: glove(out_fn, 100),
    'mnist-784-euclidean': mnist,
    'sift-128-euclidean': sift,
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
