from typing import List, Tuple, Optional, Union, Any
from itertools import product
from math import sqrt as msqrt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pyink as pu


def pair_distance(i1, i2):
    return msqrt((i1[0] - i2[0]) ** 2 + (i1[1] - i2[1]) ** 2)


class SOMSampler:
    """Populate a list with neuron positions"""

    def __init__(self, som: pu.SOM, N: int, method: str = "kmeans", **kwargs):
        self.som = som
        self.points: List = []
        self._sample(N, method=method, **kwargs)
        self.points.sort()

    @property
    def num_points(self) -> int:
        return len(self.points)

    def _sample(self, N: int, method: str = "random", **kwargs) -> None:
        methods = {"random": self.random_sampler, "kmeans": self.kmeans_sampler}
        fxn = methods.get(method)
        if fxn is None:
            raise KeyError(f"Method {method} not defined.")
        fxn(N, **kwargs)

    def random_neuron(self) -> Tuple[int, int]:
        w, h = self.som.som_shape[:2]
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        return (y, x)

    def random_sampler(self, N: int, min_dist: int = 2, max_attempts: int = 100):
        def valid(ind: Tuple[int, int]):
            if len(self.points) == 0:
                return True
            if self.min_distance(ind) > min_dist:
                return True
            return False

        def next_point() -> Tuple[int, int]:
            attempt = 0
            cont = True
            while cont:
                if attempt > max_attempts:
                    raise ValueError(
                        f"Could not find an index within {max_attempts} attempts"
                    )
                ind = self.random_neuron()
                cont = not valid(ind)
                attempt += 1
            return ind

        while len(self.points) < N:
            try:
                ind = next_point()
            except ValueError:
                print("Could not find any more valid points.")
                print(f"Filled {self.num_points} of the requested {N}.")
                break
            self.add_point(ind)

    def kmeans_sampler(self, N: int):
        """Use k-means clustering to optimize the position of N points on the SOM"""
        w, h = self.som.som_shape[:2]
        X = list(product(range(h), range(w)))
        km = KMeans(N).fit(X)
        centers = km.cluster_centers_
        for pi in centers:
            self.add_point(tuple(pi.astype(int)))

    def add_point(self, point: Tuple[int, int]):
        self.points.append(point)

    def min_distance(self, ind: Tuple[int, int]) -> float:
        dists = [pair_distance(ind, pi) for pi in self.points]
        return np.min(dists)

    def visualize(self):
        img = np.zeros(self.som.som_shape[:2])
        for p in self.points:
            img[p[0], p[1]] += 1
        plt.imshow(img)
