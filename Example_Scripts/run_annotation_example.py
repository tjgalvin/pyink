"""Simple example driving script to interactively annotate neurons

TODO: Add a argparse interface to make script callable from command line
"""

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

path = pu.PathHelper("Annotation")

annotator = pu.Annotator("SOMs/SOM_B3Circular_h8_w8_emu.bin")

annotator.interactive_annotate()
annotator.save_annotation_results(path=f"{path}/ANNOTATION_B3Circular_h8_w8_emu.bin")

saved_annotations = pu.Annotator(
    "SOMs/SOM_B3Circular_h8_w8_emu.bin",
    results=f"{path}/ANNOTATION_B3Circular_h8_w8_emu.bin",
)
print(saved_annotations.results)
