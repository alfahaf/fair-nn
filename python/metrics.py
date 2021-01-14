import numpy 


def result_prob(results):
    probs = {}
    for q, res in results.items():
        probs[q] = []
        cnts = {}
        for r in res:
            if r != -1:
                cnts.setdefault(r, 0)
                cnts[r] += 1
        n = sum(cnts.values())
        p = [[r, cnts[r]/n] for r in cnts]
        probs[q] = p
    return probs


def total_variation_dist(g, res):
    all = numpy.zeros(len(g))
    for i in range(len(g)):
        n = int(1/g[i])
        r = numpy.array([p for _, p in res[i]] + [0] * (n - len(res[i])))
        all[i] = 0.5 * numpy.sum(numpy.abs(r - g[i]))
    return numpy.mean(all), all