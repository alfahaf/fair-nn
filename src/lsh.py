import numpy as np
import random
import time

class LSH:
    def preprocess(self, X):
        d, n = X.shape
        hvs = self._hash(X)
        self.tables = [{} for _ in range(self.L)]
        for i in range(n):
            for j in range(self.L):
                h = self._get_hash_value(hvs[i], j) 
                self.tables[j].setdefault(h, set()).add(i)

    def uniform_query(self, Y):
        hvs = self._hash(Y)
        results = []
        for q in hvs:
            buckets = [(i, self._get_hash_value(q, i)) for i in range(self.L)]
            table, bucket = random.choice(buckets)
            results.append(random.choice(list(self.tables[table][bucket])))
        return results

    def weighted_uniform_query(self, Y):
        hvs = self._hash(Y)
        bucket_sizes = []
        results = []
        for q in hvs:
            buckets = [(i, self._get_hash_value(q, i)) for i in range(self.L)]
            s = 0
            for table, bucket in buckets:
                s += len(self.tables[table][bucket])
            bucket_sizes.append(s)

        for _ in range(1000):
            for i, q in enumerate(hvs):
                buckets = [(i, self._get_hash_value(q, i)) for i in range(self.L)]
                i = random.randrange(bucket_sizes[i])
                s = 0
                for table, bucket in buckets:
                    s += len(self.tables[table][bucket])
                    if s > table:
                        results.append(random.choice(list(self.tables[table][bucket])))
                        break
        return results

    def opt(self, Y):
        hvs = self._hash(Y)
        results = []
        for q in hvs:
            buckets = [(i, self._get_hash_value(q, i)) for i in range(self.L)]
            elements = set()
            for table, bucket in buckets:
                elements = elements.union(self.tables[table][bucket])
            results.append(random.choice(list(elements)))
        return results

    def approx(self, Y, eps=0.2):
        pass

class MinHash():
    def __init__(self):
        # choose four random 8 bit tables
        self.t1 = [random.randint(0, 2**32 - 1) for _ in range(2**8)]
        self.t2 = [random.randint(0, 2**32 - 1) for _ in range(2**8)]
        self.t3 = [random.randint(0, 2**32 - 1) for _ in range(2**8)]
        self.t4 = [random.randint(0, 2**32 - 1) for _ in range(2**8)]

    def _intern_hash(self, x):
        return self.t1[(x >> 24) & 0xff] ^ self.t2[(x >> 16) & 0xff ] ^\
            self.t3[(x >> 8) & 0xff] ^ self.t4[x & 0xff]

    def _hash(self, X):
        return min([self._intern_hash(x) for x in X])

    def get_element(self, L):
        h = self.hash(L)
        for x in L:
            if self.intern_hash(x) == h:
                return x

class OneBitMinHash(LSH):
    def __init__(self, k, L, seed=3):
        self.k = k
        self.L = L
        self.hash_fcts = [[MinHash() for _ in range(k)] for _ in range(L)]

    def _hash(self, X):
        self.hvs = []
        for x in X:
            self.hvs.append([])
            for hash_fct in self.hash_fcts:
                h = 0
                for hf in hash_fct:
                    h += hf._hash(x) % 2
                    h *= 2
                self.hvs[-1].append(h) 
        return self.hvs

    def _get_hash_value(self, arr, idx):
        return arr[idx]


class E2LSH(LSH):
    def __init__(self, k, L, w, d, seed=3):
        np.random.seed(seed)
        random.seed(seed)
        self.A = np.random.normal(0.0, 1.0, (d, k * L))
        self.b = np.random.uniform(0.0, w, (1, k * L)) 
        self.w = w
        self.L = L
        self.k = k

    def _hash(self, X):
        X = np.transpose(X)
        hvs = np.matmul(X, self.A) 
        hvs += self.b
        hvs /= self.w
        return np.floor(hvs).astype(np.int32)

    def _get_hash_value(self, arr, idx):
        return tuple(arr[idx * self.k: (idx + 1) * self.k])


def l2(u, v):
    return np.linalg.norm(u, v)

def jaccard(u, v):
    pass

def test_minhash():
    n = 10000
    m = 100
    d = 10
    k = 4
    L = 10

    lsh = OneBitMinHash(k, L)
    X = [[random.choice(list(range(100))) for _ in range(d)] for _ in range(n)]
    Y = [[random.choice(list(range(100))) for _ in range(d)] for _ in range(m)]
    lsh.preprocess(X)
    s = time.time()
    lsh.weighted_uniform_query(Y)
    print(time.time() - s)

def test_euclidean():
    d = 10 
    n = 10000
    m = 100
    w = 4.0
    k = 2 
    L = 3
    lsh = E2LSH(k, L, w, d)
    X = np.random.normal(0.0, 1.0, (d, n))    
    lsh.preprocess(X)
    Y = np.random.normal(0.0, 1.0, (d, m))
    s = time.time()
    lsh.weighted_uniform_query(Y)
    print(time.time() - s)

if __name__ == "__main__":
    test_euclidean()

