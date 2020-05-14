"""Actions and helpers to build a graph to collate objects
"""

from typing import List, Set, Dict, Tuple, Optional, Union, Callable, TYPE_CHECKING
import logging
from enum import Enum, auto

import numpy as np
import networkx as nx

import pyink as pu


class Action(Enum):
    """Straight forward enumation for actions to perform on the graph
    """

    LINK = auto()
    UNLINK = auto()
    RESOLVE = auto()
    FLAG = auto()
    PASS = auto()


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
