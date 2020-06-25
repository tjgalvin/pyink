"""Contains basic common utility functions that could foreseeably
be used outside of pyink
"""
from typing import List, Set, Dict, Tuple, Optional, Union, Any, Iterable
import logging
import os
import shutil
import warnings

import numpy as np
from scipy.ndimage import rotate
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


def _pink_spatial_transform(
    img: np.ndarray, transform: Tuple[np.int8, np.float32]
) -> np.ndarray:
    """Applying the PINK spatial transformation to an image
    
    Arguments:
        img {np.ndarray} -- Image to spatially transform
        transform {Tuple[np.int8, np.float32]} -- Spatial transformation specification
    
    Returns:
        np.ndarray -- Spatially transformed image
    """

    flip, angle = transform

    img = rotate(img, -np.rad2deg(angle), reshape=False)

    if flip == 1:
        img = img[::-1]

    return img


def _reverse_pink_spatial_transform(
    img: np.ndarray, transform: Tuple[np.int8, np.float32]
) -> np.ndarray:
    """Applying the PINK spatial transformation to an image, but performed in the reverse order. 
    This would be useful to align a neuron onto an image. 
    
    Arguments:
        img {np.ndarray} -- Image to spatially transform (should be a neuron)
        transform {Tuple[np.int8, np.float32]} -- Spatial transformation specification
    
    Returns:
        np.ndarray -- Spatially transformed image
    """

    flip, angle = transform

    if flip == 1:
        img = img[::-1]

    img = rotate(img, np.rad2deg(angle), reshape=False)

    return img


def pink_spatial_transform(
    img: np.ndarray, transform: Tuple[np.int8, np.float32], reverse: bool = False
) -> np.ndarray:
    """Applying the PINK spatial transformation to an image
    
    Arguments:
        img {np.ndarray} -- Image to spatially transform
        transform {Tuple[np.int8, np.float32]} -- Spatial transformation specification following the PINK standard
    
    Keyword Arguments:
        reverse {bool} -- Apply the spatial transform in reverse order (flip first and then rotate). Useful to align a neuron onto an input image. 

    Returns:
        np.ndarray -- Spatially transformed image
    """
    if reverse:
        transform_func = lambda i, t: _reverse_pink_spatial_transform(i, t)
    else:
        transform_func = lambda i, t: _pink_spatial_transform(i, t)

    if len(img.shape) == 3:
        return np.array([transform_func(c_img, transform) for c_img in img])
    elif len(img.shape) == 2:
        return transform_func(img, transform)
    else:
        raise ValueError(
            f"Image to transform must be either of shape (channel, height, width) or (height, width). Got image of shape {img.shape}"
        )


def compute_distances_between_valid_pixels(mask: np.ndarray) -> np.ndarray:
    """Given a mask, compute the distance between each pixel in a pair-wise fashion

    Arguments:
        mask {np.ndarray} -- Mask to compute distances between

    Returns:
        np.ndarray -- Matrix object of distances between pixels, see the return of `scipy.spatial.distance.cdist`
    """
    pos = np.argwhere(mask)

    return cdist(pos, pos)


def distances_between_valid_pixels(
    mask: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Given a mask, compute the distances between all valid pixels and return the
    maximum separation between any two pixels, which pixels these were, the actual
    distances between each pair-wise combination of valid pixels. 

    Arguments:
        mask {np.ndarray} -- a two-dimenaional boolean array

    Returns:
        Tuple[float, np.ndarray, np.ndarray] -- maximum separation between any two pixels,
                                                which pixels these were, and distance matrix
    """
    dist = compute_distances_between_valid_pixels(mask)
    try:
        max_dist = np.max(dist)
        max_pos = np.unravel_index(np.argmax(dist), dist.shape)

        return (max_dist, max_pos, dist)
    except ValueError:
        # if there are no valid pixels in the mask, nothing can be done.
        # return a negative distance.
        warnings.warn(
            "Empty mask detected from which no maximum distance can be computed. "
        )
        return (-1, np.ndarray([-1 for _ in dist.shape]), dist)


def valid_region(
    filter: np.ndarray,
    filter_includes: Union[int, Iterable[int]] = None,
    filter_excludes: Union[int, Iterable[int]] = None,
) -> np.ndarray:
    """Constructs a valid region based on whether labels are present or not. If `None` is specified
    for `fliter_includes` all non-zero pixels will be considered as valid

    Arguments:
        filter {np.ndarray} -- Filter to use as the base

    Keyword Arguments:
        filter_includes {Union[int, Iterable[int]]} -- Labels to include in the masked region (default: {None})
        filter_excludes {Union[int, Iterable[int]]} -- Labels to exclude from the region (default: {None})

    Returns:
        np.ndarray -- Boolean masked constructed following the specifications
    """
    mask = np.zeros_like(filter, dtype=np.bool)
    if filter_includes is None:
        mask = filter != 0
    else:
        if isinstance(filter_includes, int):
            filter_includes = [
                filter_includes,
            ]

        for fi in filter_includes:
            mask = mask | (filter % fi == 0)

    if filter_excludes is None:
        return mask

    else:
        if isinstance(filter_excludes, int):
            filter_excludes = [
                filter_excludes,
            ]

        for fe in filter_excludes:
            mask = mask & ~(filter % fe == 0)

        return mask


def area_ratio(
    filter1: np.ndarray,
    filter2: np.ndarray,
    filter_includes=None,
    filter_excludes=None,
    empty_check: bool = True,
) -> float:
    """Compute the order of the filters by the relative ratio between desired regions in each filter. 

    Arguments:
        filter1 {np.ndarray} -- Desired filter 1
        filter2 {np.ndarray} -- Desired filter 2
    
    Keyword Arguments:
        filter_includes {Union[int, Iterable[int]]} -- Labels to include in the masked region (default: {None})
        filter_excludes {Union[int, Iterable[int]]} -- Labels to exclude from the region (default: {None})
        empty_check {bool} -- If `filter2` has no valid pixels, the returned ratio will be `1` to prevent non-finite values from being returned (default: {True})

    Returns:
        float -- the ratio between the valid areas, computed as `filter1 / filter1`
    """
    mask1 = valid_region(
        filter1, filter_includes=filter_includes, filter_excludes=filter_excludes
    )
    mask2 = valid_region(
        filter2, filter_includes=filter_includes, filter_excludes=filter_excludes
    )

    mask1_sum = np.sum(mask1)
    mask2_sum = np.sum(mask2)

    if empty_check and mask2_sum == 0:
        mask2_sum = mask1_sum

    return mask1_sum / mask2_sum


class PathHelper:
    """Small utility helper to transparently create output folders. Idea being:

    path = PathHelpher('Example_Images')
    path.images1
    path.images2

    will produce a folder structure of:
    Example_Images/
    Example_Images/images1/
    Example_Images/images2/

    This was written with creating matplotlib figures in mind across different trials
    """

    def __init__(self, path: str, clobber: bool = False) -> None:
        """Creates the base path 
        
        Arguments:
            path {str} -- Base path to create 
        
        Keyword Arguments:
            clobber {bool} -- Will delete the base path and its contents if it exists (default: {False})
        """
        if clobber and os.path.exists(path):
            shutil.rmtree(path)

        if not os.path.exists(path):
            os.mkdir(path)
        self.path = path

    def __getattr__(self, name: str):
        """Will create a subdirectory of the specified base path
        
        Arguments:
            name {str} -- New sub-directory
        
        Returns:
            PathHelpher -- New PathHelper object with the base path of the sub-directory
        """
        name_path = f"{self.path}/{name}"

        if not os.path.exists(name_path) and "_ipython" not in name:
            os.mkdir(name_path)

        return self.__class__(name_path)

    def __repr__(self) -> str:
        """Neat string representation simply returning the base path
        
        Returns:
            str -- The base path
        """

        return self.path
