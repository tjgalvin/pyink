"""Simple example driving script to interactively annotate neurons
"""
import argparse
from typing import Tuple, Any

import platform
import logging

logger = logging.getLogger("pyink.annotation")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


plat = platform.system()
if plat == "Darwin":
    import matplotlib

    matplotlib.use("MacOSX")
    logger.debug(
        "Detected MacOSX platform. Configured matplotlib backend appropriately."
    )

import pyink as pu


def update_annotation(som: str, key: Tuple[Any, ...], results: str = None):
    """Load an existing annotation set and update a single neuron. To do this
    a annotation set has to be loaded. 

    TODO: Build in a custom path for the `results` argument

    Arguments:
        som {str} -- Path to SOM binary file
        key {Tuple[Any, ...]} -- Key of the `results` attribute to update
    
    Keyword Arguments:
        results {str} -- Path to existing results Annotator set. Default will atempt to automatically find one. (default: {None})
    """
    results_path = True if results is None else results

    annotator = pu.Annotator(som, results=results_path)
    annotator.annotate_neuron(key, update=True, labeling=True)

    results_path = None if results_path == True else results

    annotator.save_annotation_results(results_path)


def perform_annotation(som: str, save: bool = True, resume: bool = False):
    """A simple driver to perform annotation of a `SOM`

    Arguments:
        som {str} -- Path to desired SOM for annotation

    Keyword Arguments:
        save {bool} -- Save the annotations as they are being performed (default: {True})
        resume {bool} -- Continue the annotation process from the first un-annotated neuron (default: {False})
    """
    annotator = pu.Annotator(som, save=save)
    annotator.interactive_annotate(resume=resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Initiate the manual annotation of a SOM"
    )
    parser.add_argument("som", help="Path to the desired SOM to annotate")
    parser.add_argument(
        "-s",
        "--save",
        help="Automatically save the annotations using the default naming scheme (appending `.results.pkl). Default behaviour of the `Annotator` class is to save after each neuron has been annotated. ",
        default=True,
        action="store_false",
    )
    parser.add_argument(
        "-k",
        "--key",
        nargs="+",
        help="The key of the neuron to update. Following the scheme from `Annotator` this will be converted to a `tuple` with elements of type `int`.",
    )
    parser.add_argument(
        "-r",
        "--results",
        default=False,
        nargs=1,
        help="The path to a previously saved annotation set, only used when `--key` is supplied and is meant for when a non-standard naming scheme is used (see the `--save`). ",
    )
    parser.add_argument(
        "-c",
        "--resume",
        default=False,
        action="store_true",
        help="Continue the annotation process from the first un-annotated neuron (skip those already labeled)",
    )

    args = parser.parse_args()

    if args.key is None:
        perform_annotation(args.som, save=args.save, resume=args.resume)

    else:
        key = tuple(int(t) for t in args.key)
        results = args.results[0] if not args.results is False else None
        update_annotation(args.som, key, results=results)
