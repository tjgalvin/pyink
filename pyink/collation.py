"""Actions and helpers to build a graph to collate objects
"""

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
from enum import Enum, auto
from itertools import combinations, permutations

import numpy as np
import networkx as nx
import astropy.units as u
from tqdm import tqdm
from astropy.coordinates import SkyCoord, Angle

import pyink as pu

logger = logging.getLogger(__name__)


class Action(Enum):
    """Straight forward enumation for actions to perform on the graph
    """

    LINK = auto()
    UNLINK = auto()
    RESOLVE = auto()
    FLAG = auto()
    PASS = auto()
    NODE_ATTACH = auto()
    DATA_ATTACH = auto()
    TRUE_ATTACH = auto()
    FALSE_ATTACH = auto()
    ISOLATE = auto()


class LabelResolve(dict):
    """Helper class used to define actions that labels resolve to. A default Action
    may be specified for when a label is requested but a correspond Action is not
    defined
    """

    def __init__(self, *args, default: Action = Action.PASS, **kwargs):
        """Creates a new Dict resolving labels to certain Actions A default Action may
        be provided
        
        Keyword Arguments:
            default {Actions} -- Default Action to perform for unresolved label (default: {Actions.PASS})
        """
        self.default = default
        super().__init__(*args, **kwargs)

    def __getitem__(self, key: Any) -> Action:
        """Resolved the defined label to an action
        
        Arguments:
            key {str} -- Label to resolve
        
        Returns:
            Actions -- Action to perform
        """
        try:
            return super().__getitem__(key)
        except KeyError:
            return self.default

    def __repr__(self) -> str:
        """Pretty representation of class for printing
        
        Returns:
            str -- String description of instace
        """
        return f"{super().__repr__()}, Default action: {self.default.__repr__()}"


class Sorter:
    """Handler to control how objects and their filters are accessed by the greedy graph
    """

    MODES = ["best_matching_first", "largest_first", "area_ratio"]

    def __init__(
        self, som_set: pu.SOMSet, *args, mode: str = "best_matching_first", **kwargs
    ):
        """Creates the Sorter to provide source `Filters` in a specified order. Any 
        `args` or `kwargs` are passed through to the corresponding ordering tools. 

        Available `modes` are:
            best_matching_first -- Orders sources based on their similarity to their BMU, from best to worst
            largest_first -- Will sort neurons based on the spatial size of the filters, where size 
                             is the maximum distance between any two non-zero pixels. Sources will be 
                             presented in the same order that their best matching neuron is in the list. 
                             As a positional argument an `Annotator` object will have to be provided. 
                             This may also accept a keyword argument `channel` as an `int` to select which filter 
                             channel to use with a default of `0`. `sort_srcs` as a `bool` will sort 
                             the sources matching each neuron from best to worst with a default of `True`.
            area_ratio -- Will sort neurons based on the ratio of valid pixels between two filters. 
                          As a positional argument an `Annotator` object will have to be provided. `filter1`
                          and `filter2` set the channels to draw the filters from. `filter_includes` and `filter_excludes`
                          may be either an `int` or `Iterable[int]` that will build a mask based on the 
                          presence of those labels. If `filter_includes` is `None` all non-zero pixels are 
                          considered valid. `sort_srcs` as a `bool` will sort the sources matching each 
                          neuron from best to worst with a default of `True`. See `pu.valid_region`.

        Arguments:
            som_set {pu.SOMSet} -- Container holding the SOM, Mapping and Transform files of interest
        
        Keyword Arguments:
            mode {str} -- Sorting mode operation (default: {'best_matching_first'})
        """
        # MODES = ["best_matching_first", "largest_first", "area_ratio"]
        if mode not in self.MODES:
            raise NotImplementedError(
                f"Support order modes are {', '.join(self.MODES)}, received {mode}"
            )

        self.mode = mode
        self.som_set = som_set

        self.order: np.ndarray
        if mode == "best_matching_first":
            self.order = self._ed_order()
        elif mode == "largest_first":
            self.order = self._largest_order(*args, **kwargs)
        elif mode == "area_ratio":
            self.order = self._area_ratio(*args, **kwargs)

    def _ed_order(self) -> np.ndarray:
        """Creates an order from best matching to worst matching based on the similarity of an 
        image to its best matching neuron
        
        Returns:
            np.ndarray -- Indicies of sources that were best matching to worst matching
        """
        ed = self.mapper.bmu_ed()
        order = np.argsort(ed)

        return order

    def _largest_order(
        self,
        annotations: pu.Annotator,
        *args,
        channel: int = 0,
        sort_srcs: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Sorts the annotations by the size of the constructed filters, where size is determined
        by the largest separation between any two non-zero pixels. 

        Arguments:
            annotations {pu.Annotator} -- An annotated SOM described in an `Annotator` object

        Keyword Arguments:
            channel {int} -- The channel filter to operate over (default: {0})
            sort_srcs {bool} -- Sort the sources that match each BMU from best to worst (default: {True})

        Returns:
            np.ndarray -- source indicies belonging to the neurons in the sorted order
        """
        ants = [(k, v.filters[channel]) for k, v in annotations.results.items()]
        ants = sorted(
            ants,
            key=lambda x: pu.distances_between_valid_pixels(x[1] != 0)[0],
            reverse=True,
        )

        order = []
        for ant in ants:
            key = ant[0]
            src_idx = self.mapper.images_with_bmu(key)
            if sort_srcs:
                src_ed = self.mapper.bmu_ed()[src_idx]
                src_order = np.argsort(src_ed)
            else:
                src_order = np.arange(src_idx.shape[0])

            if len(src_order) > 0:
                order.extend(src_idx[src_order])

            order.extend(src_idx[src_order])

        return order

    def _area_ratio(
        self,
        annotations: pu.Annotator,
        *args,
        filter1=0,
        filter2=1,
        filter_includes=None,
        filter_excludes=None,
        sort_srcs: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Compute the order of the filters by the relative ratio between desired regions in each filter. 

        Arguments:
            annotations {pu.Annotator} -- An annotated SOM described in an `Annotator` object
            
        Keyword Arguments:
            filter1 {int} -- Desired filter 1 (default: {0})
            filter2 {int} -- Desired filter 2 (default: {1})
            filter_includes {Union[int, Iterable[int]]} -- Labels to include in the masked region (default: {None})
            filter_excludes {Union[int, Iterable[int]]} -- Labels to exclude from the region (default: {None})
            sort_srcs {bool} -- Sort the sources that match each BMU from best to worst (default: {True})

        Returns:
            np.ndarray -- source indicies in sorted order
        """
        ants = [
            (k, v.filters[filter1], v.filters[filter2])
            for k, v in annotations.results.items()
        ]
        ants = sorted(
            ants,
            key=lambda x: pu.area_ratio(
                x[1],
                x[2],
                filter_includes=filter_includes,
                filter_excludes=filter_excludes,
            ),
            reverse=True,
        )

        # TODO: Move this to a unified function to be used in common with
        # TODO: the same sorting in _largest_order
        order = []
        for ant in ants:
            key = ant[0]
            src_idx = self.mapper.images_with_bmu(key)
            if sort_srcs:
                src_ed = self.mapper.bmu_ed()[src_idx]
                src_order = np.argsort(src_ed)
            else:
                src_order = np.arange(src_idx.shape[0])

            if len(src_order) > 0:
                order.extend(src_idx[src_order])

        return order

    def __len__(self) -> int:
        """Returns the number of sources in the mapper file i.e. number of sources.
        
        Returns:
            int -- Number of sources in the mapper file
        """
        return self.mapper.data.shape[0]

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self) -> int:
        """Enables iteration in for the Sorted class, which will simply iterate
        over the items specified by the `order` attribute
        
        Returns:
            int -- Index corresponding to a source / image from mapping
        """
        try:
            item = self.order[self.idx]
        except IndexError:
            raise StopIteration()
        self.idx += 1

        return item

    def __getitem__(self, *args) -> np.ndarray:
        """Directly calls the __getitem__ of the stored `order`
        
        Returns:
            np.ndarray -- subset of `order` specified by the indexing
        """
        return self.order.__getitem__(*args)

    def __repr__(self) -> str:
        """Neat string representation of the sorter object

        Returns:
            str -- description string to print
        """
        return (
            f"`Sorter` with mode `{self.mode}` containing `{len(self.order)}` objects."
        )

    @property
    def som(self) -> pu.SOM:
        """Returns the `SOM` from the attached `SOMSet`

        Returns:
            pu.SOM -- SOM object describing a PINK SOM binary file
        """
        return self.som_set.som

    @property
    def mapper(self) -> pu.Mapping:
        """Returns the `mapping` from the attached som_set

        Returns:
            pu.Mapping -- Mapping object describing a PINK mapping binary file
        """
        return self.som_set.mapping

    @property
    def transform(self) -> pu.Transform:
        """Returns the `Transform` from the attached `SOMSet`

        Returns:
            pu.Transform -- Transform object describing a PINK transform binary file
        """
        return self.som_set.transform


# ---------------------------------


def greedy_graph(
    filters: pu.FilterSet,
    annotations: pu.Annotator,
    label_resolve: LabelResolve,
    sorter: Sorter,
    src_stats_fn: Callable = None,
    progress: bool = False,
) -> nx.MultiGraph:
    """Creates greedy graph based on configurables based on the exposed utility classes. 

    Arguments:
        filters {pu.FilterSet} -- Projected cookie-cutter filters containing the segmented catalogue sources
        annotations {pu.Annotator} -- The labeled filters with regions highlight location and type of features
        label_resolve {LabelResolve} -- Actions to perform when a label is encountered
        sorter {Sorter} -- Orders the construction of the greedy graph creation

    keyword Arguments:
            src_stats_fn {Callable} -- User provided function passed to `greedy_graph`. 
                                       It should atleast the `src_idx` returned by the 
                                       `Sorter` class, and should return a `dict` with 
                                       information to attach to the edge. No additional
                                       arguments are passed. If some are need a closure
                                       is recommended. (default: {None})
        progress {bool} -- Provide a `tqdm` progress bar (default: {False})

    Returns:
        nx.MultiGraph -- Link components and information as a `networkx` graph
    """
    labels = annotations.unique_labels()
    G = nx.MultiGraph()

    isolate = []

    for i, src_idx in tqdm(enumerate(sorter), disable=not progress):
        # Node already added
        if src_idx in G.nodes():
            continue

        src_filters = filters[src_idx]

        edge_data: Dict[Any, Any] = {"count": i, "src_idx": src_idx}
        if src_stats_fn is not None:
            edge_data.update(src_stats_fn(src_idx))

        node_link = []
        node_unlink = []

        for j, src_filter in enumerate(src_filters):
            for label in labels:
                # Some checks below to help `mypy` while typechecking.
                # TODO: Consider making the `src_idx` in CoordinateTransfom mandatory
                if src_filter.coords.src_idx is None:
                    raise ValueError(
                        "`src_idx` not provided for the `CoordinateTransformer`."
                    )

                action = label_resolve[label]
                mask = src_filter.coord_label_contains(label)

                if action == pu.Action.NODE_ATTACH:
                    G.add_nodes_from(
                        src_filter.coords.src_idx[mask], **{label: True}
                    )  # If node exists attribute is added

                elif action == pu.Action.DATA_ATTACH:
                    edge_data[label] = src_filter.coords.src_idx[mask]

                elif action == pu.Action.TRUE_ATTACH:
                    if np.sum(mask) > 0:
                        edge_data[label] = True

                elif action == pu.Action.FALSE_ATTACH:
                    if np.sum(mask) > 0:
                        edge_data[label] = False

                elif action == pu.Action.LINK:
                    node_link.extend(list(src_filter.coords.src_idx[mask]))

                elif action == pu.Action.UNLINK:
                    node_unlink.extend(list(src_filter.coords.src_idx[mask]))

                elif action == pu.Action.PASS:
                    pass

                elif action == pu.Action.ISOLATE:
                    isolate_idx = list(src_filter.coords.src_idx[mask])
                    isolate.extend(isolate_idx)
                    G.remove_edge(G.edges(isolate_idx))
                    G.add_nodes_from(isolate_idx, {"isolated": label})

                elif action == pu.Action.FLAG:
                    edge_data["Flagged"] = True

                elif action == pu.Action.RESOLVE:
                    edge_data["Resolve"] = True

        node_link = list(set(node_link) - set(isolate))
        node_unlink = list(set(node_unlink))

        for idx1, idx2 in combinations(node_link, 2):
            G.add_edge(idx1, idx2, **edge_data)

        G.add_edge(src_idx, src_idx, **edge_data)

        for idx1 in node_unlink:
            edges = G.edges(idx1)
            G.remove_edge(edges)

    return G


class Grouper:
    """Class that attempts to implement the `greedy graph`to collate sources together
    base on the projected cookie-cutter filters. 
    """

    def __init__(
        self,
        filters: pu.FilterSet,
        annotations: pu.Annotator,
        label_resolve: LabelResolve,
        sorter: Sorter,
        src_stats_fn: Callable = None,
        progress: bool = False,
    ):
        """Creates a new `Grouper` object that drives the creation of the greedy graph. 
        This will attempt to restrict its operation to the provided specialisation classes.

        Arguments:
            filters {pu.FilterSet} -- The projected cookie-cutter filters
            annotations {pu.Annotator} -- Provided annotated filters
            label_resolve {LabelResolve} -- Actions to perform for individual labels
            sorter {Sorter} -- Specifies the order which the filters are iterated over
        
        Keyword Arguments:
            src_stats_fn {Callable} -- User provided function passed to `greedy_graph`. 
                                       It should atleast the `src_idx` returned by the 
                                       `Sorter` class, and should return a `dict` with 
                                       information to attach to the edge. No additional
                                       arguments are passed. If some are need a closure
                                       is recommended. (default: {None})
            progress {bool} -- Provide a `tqdm` style progress bar (default: {False})
        """
        self.filters = filters
        self.annotations = annotations
        self.label_resolve = label_resolve
        self.sorter = sorter
        self.progress = progress
        self.src_stats_fn = src_stats_fn

        self.graph: nx.MultiGraph = self._generate_graph()

    def _generate_graph(self) -> nx.MultiGraph:
        """Creates the graph based on specified driver classes

        Returns:
            nx.MultiGraph -- greedy graph
        """
        return greedy_graph(
            self.filters,
            self.annotations,
            self.label_resolve,
            self.sorter,
            src_stats_fn=self.src_stats_fn,
            progress=self.progress,
        )
