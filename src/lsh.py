import numpy as np
import random
import time
import distance

class LSHBuilder:
    @staticmethod
    def build(d, r, k, L, lsh_params, validate=False):
        if lsh_params['type'] == 'e2lsh':
            return E2LSH(k, L, lsh_params['w'], d, r, validate)
        if lsh_params['type'] == 'onebitminhash':
            return OneBitMinHash(k, L, validate)


class LSH:
    def preprocess(self, X):
        self.X = X
        n, d = len(X), len(X[0]) 
        hvs = self._hash(X)
        self.tables = [{} for _ in range(self.L)]
        for i in range(n):
            for j in range(self.L):
                h = self._get_hash_value(hvs[i], j) 
                self.tables[j].setdefault(h, set()).add(i)

    def get_query_size(self, Y):
        hvs = self._hash(Y)
        results = {i: [] for i in range(len(hvs))}
        for j, q in enumerate(hvs):
            buckets = [(i, self._get_hash_value(q, i)) for i in range(self.L)]
            elements = set()
            for table, bucket in buckets:
                elements = elements.union(self.tables[table].get(bucket, set()))
            elements = set(x for x in elements if self.is_candidate_valid(Y[j], self.X[x]))
            results[j] = len(elements)
        return results

    def uniform_query(self, Y, runs=100):
        sizes = self.get_query_size(Y)
        hvs = self._hash(Y)
        results = {i: [] for i in range(len(hvs))}
        for j, q in enumerate(hvs):
            buckets = [(i, self._get_hash_value(q, i)) for i in range(self.L)]
            for _ in range(sizes[j] * runs):
                table, bucket = random.choice(buckets)
                elements = list(self.tables[table].get(bucket, [-1]))
                results[j].append(random.choice(elements))
        return results

    def weighted_uniform_query(self, Y, runs=100):
        sizes = self.get_query_size(Y)
        hvs = self._hash(Y)
        bucket_sizes = []
        results = {i: [] for i in range(len(hvs))}
        for q in hvs:
            buckets = [(i, self._get_hash_value(q, i)) for i in range(self.L)]
            s = 0
            for table, bucket in buckets:
                s += len(self.tables[table].get(bucket, []))
            bucket_sizes.append(s)

        for j, q in enumerate(hvs):
            if bucket_sizes[j] == 0:
                results[j].append(-1)
                continue
            buckets = [(i, self._get_hash_value(q, i)) for i in range(self.L)]
            for _ in range(sizes[j] * runs):
                i = random.randrange(bucket_sizes[j])
                s = 0
                for table, bucket in buckets:
                    s += len(self.tables[table].get(bucket, []))
                    if s > table:
                        results[j].append(random.choice(list(self.tables[table][bucket])))
                        break
        return results

    def opt(self, Y, runs=100):
        sizes = self.get_query_size(Y)
        hvs = self._hash(Y)
        results = {i: [] for i in range(len(hvs))}
        for j, q in enumerate(hvs):
            buckets = [(i, self._get_hash_value(q, i)) for i in range(self.L)]
            elements = set()
            for table, bucket in buckets:
                elements = elements.union(self.tables[table].get(bucket, []))
            elements = list(x for x in elements 
                if self.is_candidate_valid(Y[j], self.X[x]))
            if elements == []:
                elements = [-1]

            for _ in range(sizes[j] * runs):
                results[j].append(random.choice(elements))
        return results

    def approx(self, Y, eps=0.2):
        pass

    def is_candidate_valid(self, q, x):
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
    def __init__(self, k, L, r, validate=True, seed=3):
        self.k = k
        self.L = L
        self.r = r
        self.hash_fcts = [[MinHash() for _ in range(k)] for _ in range(L)]
        self.validate = validate

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

    def is_candidate_valid(self, q, x):
        return not self.validate or distance.jaccard(q, x) >= self.r

    def __str__(self):
        return f"OneBitMinHash(k={self.k}, L={self.L})"


class E2LSH(LSH):
    def __init__(self, k, L, w, d, r, validate=True, seed=3):
        np.random.seed(seed)
        random.seed(seed)
        self.A = np.random.normal(0.0, 1.0, (d, k * L))
        self.b = np.random.uniform(0.0, w, (1, k * L)) 
        self.w = w
        self.L = L
        self.k = k
        self.r = r
        self.validate = validate

    def _hash(self, X):
        #X = np.transpose(X)
        hvs = np.matmul(X, self.A) 
        hvs += self.b
        hvs /= self.w
        return np.floor(hvs).astype(np.int32)

    def _get_hash_value(self, arr, idx):
        return tuple(arr[idx * self.k: (idx + 1) * self.k])

    def is_candidate_valid(self, q, x):
        #print(distance.l2(q, x))
        return not self.validate or distance.l2(q, x) <= self.r

    def __str__(self):
        return f"E2LSH(k={self.k}, L={self.L}, w={self.w})"

    def __repr__(self):
        return f"k_{self.k}_L_{self.L}_w_{self.w}"

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
    test_minhash()

