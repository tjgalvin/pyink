"""Contains basic common utility functions that could foreseeably
be used outside of pyink
"""
from typing import List, Set, Dict, Tuple, Optional, Union, Any, TYPE_CHECKING
import logging

import numpy as np
from scipy.ndimage import rotate

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
