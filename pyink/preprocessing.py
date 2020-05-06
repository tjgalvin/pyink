"""Some basic methods that can be employed for preprocessing of images to PINK. This is not meant to be an exhaustive set. Merely here as a reference. 
"""

import logging
from typing import Tuple, List, Dict, Iterator

import numpy as np
from skimage.morphology import closing
from skimage.measure import label

logger = logging.getLogger(__name__)


def rms_estimate(
    data: np.ndarray,
    mode: str = "mad",
    clip_rounds: int = 2,
    bin_perc: float = 0.25,
    outlier_thres: float = 3.0,
) -> float:
    """Calculates to RMS of an image, primiarily for radio interferometric images. First outlying
    pixels will be flagged. To the remaining valid pixels a Guassian distribution is fitted to the
    pixel distribution histogram, with the standard deviation being return. 
    
    Arguments:
        data {np.ndarray} -- Data to estimate the noise level of
    
    Keyword Arguments:
        mode {str} -- Clipping mode used to flag outlying pixels, either made on the median absolute deviation (`mad`) or standard deviation (`std`) (default: {'mad'})
        clip_rounds {int} -- Number of times to perform the clipping of outlying pixels (default: {2})
        bin_perc {float} -- Bins need to have `bin_perc*MAX(BINS)` of counts to be included in the fitting procedure (default: {0.25})
        outlier_thres {float} -- Number of units of the adopted outlier statistic required for a item to be considered an outlier (default: {3})

    Raises:
        ValueError: Raised if a mode is specified but not supported
    
    Returns:
        float -- Estimated RMS of the supploed image
    """
    if bin_perc > 1.0:
        bin_perc /= 100.0

    if mode == "std":
        clipping_func = lambda data: np.std(data)

    elif mode == "mad":
        clipping_func = lambda data: np.median(np.abs(data - np.median(data)))

    else:
        raise ValueError(f"{mode} not supported as a clipping mode. ")

    for i in range(clip_rounds):
        data = data[np.abs(data) < outlier_thres * clipping_func(data)]

    # Attempts to ensure a sane number of bins to fit against
    mask_counts = 0
    loop = 1
    while mask_counts < 5 and loop < 5:
        counts, binedges = np.histogram(data, bins=50 * loop)
        binc = (binedges[:-1] + binedges[1:]) / 2

        mask = counts >= bin_perc * np.max(counts)
        mask_counts = np.sum(mask)
        loop += 1

    p = np.polyfit(binc[mask], np.log10(counts[mask] / np.max(counts)), 2)
    a, b, c = p

    x1 = (-b + np.sqrt(b ** 2 - 4.0 * a * (c - np.log10(0.5)))) / (2.0 * a)
    x2 = (-b - np.sqrt(b ** 2 - 4.0 * a * (c - np.log10(0.5)))) / (2.0 * a)
    fwhm = np.abs(x1 - x2)
    noise = fwhm / 2.355

    return noise


def minmax(data: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Apply MinMax normalistion on a dataset. If `mask` is supplied, derive the statistics from the valid pixels   
    
    
    Arguments:
        data {np.ndarray} -- Data to normalise
    
    Keyword Arguments:
        mask {np.ndarray} -- Boolean mask array to select meaningful data (default: {None})
    
    Returns:
        np.ndarray -- Scaled data
    """
    if mask is not None:
        assert (
            data.shape == mask.shape
        ), "Data and Mask arrays must be of the same shape. "
        data = data[mask]
    else:
        mask = np.ones_like(data, dtype=np.bool)

    scaled = (data - np.nanmin(data[mask])) / (
        np.nanmax(data[mask]) - np.nanmin(data[mask])
    )

    return scaled


def square_mask(data: np.ndarray, size: int = None, scale: float = None) -> np.ndarray:
    """Return a boolean array mask with the inner region marked as valid
    
    Arguments:
        data {np.ndarray} -- Data to create a corresponding mask for

    Keyword Arguments:    
        size {int} -- Size of the inner valid region (default: {None})
        scale {float} -- Compute the size of the valid region in proportion to the data shape (default: {None})

    Returns:
        np.ndarray -- Square boolean array mask
    """
    assert (
        size is not None and scale is not None
    ), "Can only set a region based on `scale` or `size`, not both"
    img_size = np.array(data.shape)
    cen = img_size // 2

    if scale is not None:
        dsize = (img_size / scale / 2).astype(np.int)
    elif size is not None:
        dsize = (size // 2, size // 2)
    else:
        raise ValueError("Either `size` or `scale` has to be set")

    h, w = data.shape[:2]
    Y, X = np.ogrid[:h, :w]
    mask = (
        (Y > (cen[0] - dsize[0]))
        & (Y < (cen[0] + dsize[0]))
        & (X > (cen[1] - dsize[1]))
        & (X < (cen[1] + dsize[1]))
    )

    return mask


def circular_mask(
    data: np.ndarray, radius: int = None, scale: float = None
) -> np.ndarray:
    """Create a circular boolean mask for a sepecified image. The radius of the valid inner region will
    be set explicitly with radius or computed based on `scale`ing the image size
    
    Arguments:
        data {np.ndarray} -- Data to create a corresponding boolean mask for
    
    Keyword Arguments:
        radius {int} -- Radius of the valid inner circular region (default: {None})
        scale {float} -- Compute the radius of the valid circular region as `scale*image_size` (default: {None})
    
    Returns:
        np.ndarray -- Boolean array with a circular valid region at the center
    """
    assert (
        radius is not None and scale is not None
    ), "Only `radius` or `scale` can be set, not both"

    if scale is not None:
        radius = data.shape[-1] / scale
    elif radius is None:
        raise ValueError("Both `radius` and `scale` are unset. ")

    cen = np.array(data.shape) // 2

    h, w = data.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cen[0]) ** 2 + (Y - cen[1]) ** 2)

    mask = dist <= radius

    return mask


def island_segmentation(
    data: np.ndarray,
    threshold: float,
    return_background: bool = False,
    minimum_island_size: int = 5,
) -> Iterator[np.ndarray]:
    """Yield a set of masks that denote unique islands in a image after a threshold operation
    has been applied. 
    
    Arguments:
        data {np.ndarray} -- Data to produce a set of islands of
        threshold {float} -- Threshold level to create the set of islands of

    Keyword Arguments:
        return_background {bool} -- Return the zero-index background region determined by scikit-img (default: {False}) 
        minimum_island_size {int} -- The minimum number of pixels required for an island to be returned. Useful to avoid noise peaks creeping through (default: {5})

    Returns:
        Iterator[np.ndarray] -- Set of island masks
    """
    mask = closing(data > threshold)

    img_labels, no_labels = label(mask, return_num=True)

    for i in range(no_labels):
        if i == 0 and return_background is False:
            continue
        if np.sum(mask) <= minimum_island_size:
            continue
        yield img_labels == i
