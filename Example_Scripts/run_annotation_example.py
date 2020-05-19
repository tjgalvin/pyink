"""Simple example driving script to interactively annotate neurons
"""
import argparse

import platform
import logging

logger = logging.getLogger("pyink.annotation")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


plat = platform.system()
if plat == "Darwin":
    import matplotlib

    matplotlib.use("MacOSX")
    logger.debug(
        "Detected MacOSX platform. Configured matplotlib backend appropriately."
    )

import pyink as pu


def perform_annotation(som: str, save: bool = True):
    """A simple driver to perform annotation of a `SOM`

    Arguments:
        som {str} -- Path to desired SOM for annotation

    Keyword Arguments:
        save {bool} -- Save the annotations as they are being performed (default: {True})
    """
    annotator = pu.Annotator(som, save=save)
    annotator.interactive_annotate()


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

    args = parser.parse_args()

    perform_annotation(args.som, save=args.save)
