"""Actions and helpers to build a graph to collate objects
"""

from typing import List, Set, Dict, Tuple, Optional, Union, Callable, TYPE_CHECKING
import logging
from enum import Enum, auto

import numpy as np
import networkx as nx

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
    ATTACH = auto()
    IR_ATTACH = auto()
    CLASSIFICATION = auto()
    ISOLATE = auto()
    LABEL = auto()


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
