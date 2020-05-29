"""Contains basic common utility functions that could foreseeably
be used outside of pyink
"""
from typing import List, Set, Dict, Tuple, Optional, Union, Any
import logging
import os
import shutil

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


def pink_spatial_transform(
    img: np.ndarray, transform: Tuple[np.int8, np.float32]
) -> np.ndarray:
    """Applying the PINK spatial transformation to an image
    
    Arguments:
        img {np.ndarray} -- Image to spatially transform
        transform {Tuple[np.int8, np.float32]} -- Spatial transformation specification following the PINK standard
    
    Returns:
        np.ndarray -- Spatially transformed image
    """
    if len(img.shape) == 3:
        return np.array([_pink_spatial_transform(c_img, transform) for c_img in img])
    elif len(img.shape) == 2:
        return _pink_spatial_transform(img, transform)
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
    max_dist = np.max(dist)
    max_pos = np.unravel_index(np.argmax(dist), dist.shape)

    return (max_dist, max_pos, dist)


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
