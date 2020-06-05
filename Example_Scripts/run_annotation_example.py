"""Simple example driving script to interactively annotate neurons
"""
import argparse
from typing import Tuple, Any

import platform
import logging

logger = logging.getLogger("pyink.annotator")
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


def perform_annotation(
    som: str, save: bool = True, resume: bool = False, results: str = None
):
    """A simple driver to perform annotation of a `SOM`

    Arguments:
        som {str} -- Path to desired SOM for annotation

    Keyword Arguments:
        save {bool} -- Save the annotations as they are being performed (default: {True})
        resume {bool} -- Continue the annotation process from the first un-annotated neuron (default: {False})
    """
    annotator = pu.Annotator(som, save=save, results=results)
    annotator.interactive_annotate(resume=resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Initiate the manual annotation of a SOM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("som", help="Path to the desired SOM to annotate")
    parser.add_argument(
        "-d",
        "--dont-save",
        help="Automatically save the annotations using the default naming scheme (appending `.results.pkl). Default behaviour of the `Annotator` class is to save after each neuron has been annotated. ",
        default=False,
        action="store_true",
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
        nargs="?",
        const=True,
        help="The path to a previously saved annotation set. If a path is provided attempt to load from it. If just the option flag is presented assume the desired file follows the default naming scheme (see the `--save`). Otherwise, do not attempt to load any existing results file. ",
    )
    parser.add_argument(
        "-c",
        "--resume",
        action="store_true",
        help="Continue the annotation process from the first un-annotated neuron (skip those already labeled)",
    )

    args = parser.parse_args()

    if args.key is None:
        if args.resume == True and args.results is False:
            args.results = True

        perform_annotation(
            args.som, results=args.results, save=not args.dont_save, resume=args.resume
        )

    else:
        key = tuple(int(t) for t in args.key)
        update_annotation(args.som, key, results=args.results)
