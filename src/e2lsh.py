import numpy as np
import random


class LSH:
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

    def preprocess(self, X):
        hvs = self._hash(X)
        self.tables = [{} for _ in range(L)]
        for i in range(n):
            for j in range(self.L):
                h = self._get_hash_value(hvs[i], j) 
                self.tables[j].setdefault(h, set()).add(i)


def distance(u, v):
    return np.linalg.norm(u, v)


if __name__ == "__main__":
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
    print(lsh.opt(Y))
