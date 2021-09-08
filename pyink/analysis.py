"""A set of utility functions to help analyze the SOM, including
visualization tools for the preprocessed images and general 
measurements to assess the quality of the SOM's training.
"""


import os, sys
from collections import Counter
import argparse
from itertools import product, cycle
from typing import Callable, Iterator, Union, List, Set, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import pyink as pu


def trim_neuron(neuron, img_shape=None):
    """Trim the extra padding surrounding a neuron so that it matches
    the input image size.

    Args:
        neuron (Tuple[int, int]): The indices of the neuron to be trimmed.
        img_shape (Tuple[int, int], optional): The dimensions of the input 
            image. If None, the size will be calculated from the 
            neuron dimensions.

    Returns:
        np.ndarray: The trimmed neuron
    """
    som_shape = neuron.shape[0]
    if img_shape is None:
        img_shape = int(np.floor(som_shape / np.sqrt(2)))
    b1 = (som_shape - img_shape) // 2
    b2 = b1 + img_shape
    return neuron[b1:b2, b1:b2]


def plot_image(
    imbin: pu.ImageReader,
    idx: Union[None, int, list, np.ndarray] = None,
    df: pd.DataFrame = None,
    somset: pu.SOMSet = None,
    apply_transform: bool = False,
    transform_neuron: bool = False,
    fig: Union[None, plt.figure] = None,
    show_bmu: bool = False,
    show_index: bool = True,
    wcs: Union[None, WCS] = None,
    grid: bool = False,
    cmaps: Union[str, list] = ["viridis", "plasma", "inferno", "cividis"],
):
    """Plot an image from the image set.

    Args:
        imbin (pu.ImageReader): Image binary
        idx (int, optional): Index of the image to plot. Defaults to None.
        df (pandas.DataFrame, optional): Table with information on the sample. Defaults to None.
        somset (pu.SOMSet, optional): Container holding the SOM, mapping, and transform. Defaults to None.
        apply_transform (bool, optional): Option to transform the image to match the neuron. Defaults to False.
        fig (pyplot.Figure, optional): Preexisting Figure for plotting. Defaults to None.
        show_bmu (bool, optional): Display the best-matching neuron alongside the image. Defaults to False.
        wcs (astropy.wcs.WCS, optional): A WCS axis to apply to the image.
        grid (bool, optional): Show grid lines on the plot. Defaults to False.
        cmaps (str, Iterable, optional): A list of pyplot colormaps to iterate through. If the list is shorter than the number of channels, it will be cycled through.

    Raises:
        ValueError: Transform requested without the required information.
        ValueError: bmu requested with no SOMSet
    """
    if idx is None:
        # Choose an index randomly from either the df (which can
        # be trimmed to a smaller sample) or the pu.ImageReader.
        if df is not None:
            idx = np.random.choice(df.index)
        else:
            idx = np.random.randint(imbin.data.shape[0])
    img = imbin.data[idx]

    nchan = imbin.data.shape[1]
    img_shape = imbin.data.shape[2]

    if apply_transform or show_bmu:
        # Need bmu_idx, which requires either a df or somset
        if somset is not None:
            bmu_idx = somset.mapping.bmu(idx)
            tkey = somset.transform.data[(idx, *bmu_idx)]
        elif df is not None:
            bmu_idx = df.loc[idx]["bmu_tup"]
            tkey = df.loc[idx][["flip", "angle"]]
        else:
            raise ValueError("apply_transform requires either a df or somset")

    if apply_transform:
        img = pu.pink_spatial_transform(img, tkey)
        # img = np.array([inv_transform(c_img, tkey) for c_img in img])

    if fig is None:
        nrow = 2 if show_bmu else 1
        cl = False if wcs is not None else True
        fig, axes = plt.subplots(
            nrow,
            nchan,
            figsize=(nchan * 4, nrow * 4),
            squeeze=False,
            sharex=True,
            sharey=True,
            constrained_layout=cl,
            subplot_kw=dict(projection=wcs),
        )
    else:
        axes = fig.axes

    if isinstance(cmaps, str):
        cmaps = [cmaps]

    for ax, chan in zip(axes.flatten(), range(nchan)):
        cmaps_c = cycle(cmaps)
        ax.imshow(img[chan], cmap=next(cmaps_c))
        if grid:
            ax.grid()

    if show_index:
        axes.flatten()[0].set_title(f"index = {idx}")

    if show_bmu:
        cmaps_c = cycle(cmaps)
        if somset is None:
            raise ValueError("Cannot show the bmu with no somset provided")
        for chan in range(nchan):
            neuron_img = somset.som[bmu_idx][chan]
            if transform_neuron:
                neuron_img = pu.pink_spatial_transform(neuron_img, tkey, reverse=True)
            axes[1][chan].imshow(trim_neuron(neuron_img, img_shape), cmap=next(cmaps_c))
            if grid:
                ax.grid()
        axes[1, 0].set_title(f"Best-matching neuron: {bmu_idx}", fontsize=12)

    if df is not None:
        fig.suptitle(df.loc[idx]["Component_name"], fontsize=16)

