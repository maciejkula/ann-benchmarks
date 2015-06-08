import numpy as np
import sklearn

import rptrees


class RP(object):

    def __init__(self, max_leaf_size, alpha):

        self.name = 'RP(max_leaf_size={}, alpha={})'.format(max_leaf_size,
                                                             alpha)
        self._max_leaf_size = max_leaf_size
        self._alpha = alpha

    def fit(self, X):

        rp = rptrees.RPTree(leaf_size=self._max_leaf_size)
        print 'fitting'
        rp.fit(X)

        print 'indexing'
        for i, x in enumerate(X):
            rp.index(i, x, self._alpha * 0)

        self._rp = rp
        self._X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')

        print 'finished'

    def query(self, v, n):

        indices = self._rp.query(self._X, v, alpha=self._alpha)

        sim = np.dot(self._X[indices], -v)

        return indices[np.argsort(sim)[:n]]
        # return indices[:n]
