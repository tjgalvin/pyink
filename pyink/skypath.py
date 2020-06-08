
from typing import (
    List,
    Iterable,
    Set,
    Dict,
    Tuple,
    Optional,
    Union,
    Callable,
    Any,
)
import logging
import warnings
from itertools import combinations, permutations

import numpy as np
import astropy.units as u
from tqdm import tqdm
from astropy.coordinates import SkyCoord, Angle
import matplotlib.pyplot as plt

import pyink as pu

logger = logging.getLogger(__name__)

def shortest_path_between(
    separations: Dict[Tuple[int, int], Angle],
    start_end: Tuple[int, int],
    *args,
    max_iterations: int = 2000000,
    progress: bool = False,
    **kwargs,
) -> Tuple[Tuple[int, ...], Angle]:
    """Solve for the shortest path between all points described in the `separations` set. This 
    is not an efficent implementation of a searching algorithm, but is a simple brute force search. 
    When ten or more positions are specified the search will be slow. 

    Arguments:
        separations {Dict[Tuple[int,int], Angle]} -- Pair-wise separations between positions
        start_end {Tuple[int,int]} -- Indicies of the positions to start and end on

    Keyword Arguments:
        max_iterations {int} -- The maximum number of path evaluations to ensure that the search finishes in a timely manner. 
                                The result returned will (likely) be sub-optimal, but with enough iterations probably sufficent
                                for most uses (default :{int})
        progress {bool} -- Enable the `tqdm` progress bar (default: {2000000})

    Returns:
        Tuple[Tuple[int, ...], Angle] -- The shortest path specified by the indices and total distance
    """
    if len(set(start_end)) == 1:
        return (start_end, 0 * u.arcsecond)

    idxs = set([k for keys in separations.keys() for k in keys])
    path_idxs = idxs - set(start_end)

    if len(path_idxs) == 0:
        return (start_end, separations[start_end])

    # This will significantly speed up the (potentially big!) set of
    # paths to test. The resolution of the `units` types and access of
    # the `values` attribute add a very considerable amount of overhead. 
    # Adding these went from ~200 it/s to 100000 it/s.
    for k, v in separations.items():
        if isinstance(v, u.Quantity):
            separations[k] = v.to(u.deg).value

    paths = []
    for i, sub_path in tqdm(
        enumerate(permutations(path_idxs, len(path_idxs))), disable=not progress
    ):
        if i > max_iterations:
            warnings.warn(f"Maximum iterations of {max_iterations} reached while searching for the best path. ")
            break

        total_path = [start_end[0]] + list(sub_path) + [start_end[1]]
        paths.append(
            (
                tuple(total_path),
                np.sum([separations[s] for s in zip(total_path[:-1], total_path[1:])]),
            )
        )

    return min(paths, key=lambda d: d[1])


def maximum_distance(
    separations: Dict[Tuple[int, int], Angle], *args, **kwargs
) -> Tuple[Tuple[int, int], Angle]:
    """Returns the key and on-sky separation of the pair of positions with the largest separation

    Arguments:
        separations {Dict[Tuple[int, int], Angle]} -- Result separation set from `create_separation_set`

    Returns:
        Tuple[Tuple[int,int], Angle] -- The key and separation of the pair with the largest distance
    """
    return max([(k, v) for k, v in separations.items()], key=lambda k: k[1])


def create_separation_set(
    positions: SkyCoord, *args, idxs_items: Iterable[int] = None, **kwargs
) -> Dict[Tuple[int, int], Angle]:
    """Returns the distance between all pair-wise combinations of source positions. 

    Arguments:
        positions {SkyCoord} -- Positions to compute distances between

    Keyword Arguments:
        idxs_items {Iterable[int]} -- If not `None` these are the indicies each position corresponds to (i.e. in a catalogue). 
                            If `None`, just a position index into positions (default: {None})

    Returns:
        Dict[Tuple[int,int], Angle] -- On-sky angular separation between components. 
    """
    if idxs_items is None:
        idxs: Iterable[int] = np.arange(len(positions))
    else:
        idxs = idxs_items

    if len(positions) == 1:
        return {(idxs[0], idxs[0]): 0 * u.arcsecond}  # type: ignore

    pairs: Dict[Tuple[int, int], Angle] = {}

    for idx1, idx2 in combinations(idxs, 2):
        pos1 = positions[idx1]
        pos2 = positions[idx2]

        sep = pos1.separation(pos2)
        pairs[(idx1, idx2)] = sep
        pairs[(idx2, idx1)] = sep

    return pairs


class SkyPath:
    """Class to compute shortest path and maximum distance between a set of 
    points
    """

    def __init__(
        self, positions: SkyCoord, *args, idxs: Iterable[int] = None, **kwargs
    ):
        """Creates a new `SkyPath` object given a set of on-sky positions. `args` and `kwargs` are passed on through
        to the underlying functions. See `create_separation_set`, `maximum_distance` and `shortest_path_between`. 

        Arguments:
            positions {SkyCoord} -- Positions to compute path information for

        Keyword Arguments:
            idxs_items {Iterable[int]} -- If not `None` these are the indicies each position corresponds to (i.e. in a catalogue). 
                                          If `None`, just a position index into positions (default: {None})
        """
        if idxs is None:
            idxs = np.arange(len(positions))

        self.idxs = idxs
        self.positions = positions
        self.separations = create_separation_set(
            positions, *args, idxs_items=self.idxs, **kwargs
        )
        self.maximum_distance = maximum_distance(self.separations, *args, **kwargs)
        self.maximum_position_angle = self.maximum_distance_positions[0].position_angle(self.maximum_distance_positions[1])
        self.shortest_path = shortest_path_between(
            self.separations, self.maximum_distance[0], *args, **kwargs
        )
        if len(positions) <= 2:
            self.curliness = 1.0
        else:
            self.curliness = (
                self.shortest_path[1] / self.maximum_distance[1]
            ).decompose()

    @property
    def shortest_path_positions(self) -> SkyCoord:
        """Returns the on-sky positions in the shortest path order determined

        Returns:
            SkyCoord -- sorted on sky positions to produce the smallest path length
        """
        path = np.array(self.shortest_path[0])
        return self.positions[path]

    @property
    def maximum_distance_positions(self) -> SkyCoord:
        """Returns the pair of sources from `positions` with the largest angular separation

        Returns:
            SkyCoord -- pair of sky positions
        """
        max_dist = np.array(self.maximum_distance[0])
        return self.positions[max_dist]

    def plot_shortest_path(self, ax: plt.axes, *args, **kwargs):
        """Overlay the shortest path onto an existing axes, one with a world coordinate system included as the projection. All
        `*args` and `**kwargs` are passed onto `matplotlib.pyplot.scatter`.

        Arguments:
            ax {plt.axes} -- axes object to plot onto
        """
        path = self.shortest_path_positions
        ax.plot(path.ra, path.dec, *args, transform=ax.get_transform('world'), **kwargs)

    def plot_maximum_distance(self, ax: plt.axes, *args, **kwargs):
        """Overlay the maximum distance between two points onto an existing axes, one with a world coordinate system included as the projection. All
        `*args` and `**kwargs` are passed onto `matplotlib.pyplot.scatter`.

        Arguments:
            ax {plt.axes} -- axes object to plot onto
        """
        path = self.maximum_distance_positions
        ax.plot(path.ra, path.dec, *args, transform=ax.get_transform('world'), **kwargs)
