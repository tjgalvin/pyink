"""Actions and helpers to build a graph to collate objects
"""

from typing import List, Set, Dict, Tuple, Optional, Union, Callable, TYPE_CHECKING
import logging
from enum import Enum, auto
from itertools import combinations

import numpy as np
import networkx as nx
from tqdm import tqdm

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

    def __getitem__(self, key: str) -> Action:
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

    def __init__(self, som_set: pu.SOMSet, mode: str = "best_matching_first"):
        """Creates the Sorter to provide source `Filters` in a specified order
        
        Arguments:
            som_set {pu.SOMSet} -- Container holding the SOM, Mapping and Transform files of interest
        
        Keyword Arguments:
            mode {str} -- Sorting mode operation (default: {'best_matching_first'})
        """
        MODES = ["best_matching_first"]
        if mode not in MODES:
            raise NotImplementedError(
                f"Support order modes are {', '.join(MODES)}, received {mode}"
            )

        self.mode = mode
        self.som_set = som_set

        self.order: np.ndarray
        if mode == "best_matching_first":
            self.order = self._ed_order()

    def _ed_order(self) -> np.ndarray:
        """Creates an order from best matching to worst matching based on the similarity of an 
        image to its best matching neuron
        
        Returns:
            np.ndarray -- Indicies of sources that were best matching to worst matching
        """
        ed = self.mapper.bmu_ed()
        order = np.argsort(ed)

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
    progress: bool = False,
) -> nx.MultiGraph:
    """Creates greedy graph based on configurables based on the exposed utility classes. 

    Arguments:
        filters {pu.FilterSet} -- Projected cookie-cutter filters containing the segmented catalogue sources
        annotations {pu.Annotator} -- The labeled filters with regions highlight location and type of features
        label_resolve {LabelResolve} -- Actions to perform when a label is encountered
        sorter {Sorter} -- Orders the construction of the greedy graph creation

    keyword Arguments:
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

        edge_data = {"count": i}
        node_link = []
        node_unlink = []

        for j, src_filter in enumerate(src_filters):
            for label in labels:
                action = label_resolve[label]
                mask = src_filter.coord_label_contains(label)

                if action == pu.Action.NODE_ATTACH:
                    G.add_nodes_from(
                        src_filter.coords.src_idx[mask], **{label: True}
                    )  # If node exists attribute is added

                elif action == pu.Action.DATA_ATTACH:
                    edge_data[label] = src_filter.coords.src_idx[mask]

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

        node_link = list(set(node_link) - set(isolate))
        node_unlink = list(set(node_unlink))

        for idx1, idx2 in combinations(node_link, 2):
            G.add_edge(idx1, idx2, **edge_data)

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
    ):
        """Creates a new `Grouper` object that drives the creation of the greedy graph. 
        This will attempt to restrict its operation to the provided specialisation classes.

        Arguments:
            filters {pu.FilterSet} -- The projected cookie-cutter filters
            annotations {pu.Annotator} -- Provided annotated filters
            label_resolve {LabelResolve} -- Actions to perform for individual labels
            sorter {Sorter} -- Specifies the order which the filters are iterated over
        """
        self.filters = filters
        self.annotations = annotations
        self.label_resolve = label_resolve
        self.sorter = sorter

        self.graph: nx.MultiGraph = self._generate_graph()

    def _generate_graph(self) -> nx.MultiGraph:
        """Creates the graph based on specified driver classes

        Returns:
            nx.MultiGraph -- greedy graph
        """
        return greedy_graph(
            self.filters, self.annotations, self.label_resolve, self.sorter
        )
