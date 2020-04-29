
from typing import List, Set, Dict, Tuple, Optional, Union, Any, TYPE_CHECKING
import logging

import numpy as np
from scipy.ndimage import rotate
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel

from pyink.annotator import Annotation

logger = logging.getLogger(__name__)

class CoordinateTransformer:
    """Helper class to manage transforming Coordinate positions from 
    the sky-reference frame to a neuron-reference frame. This corresponds
    to applying the PINK produced transform to coordinates. 

    At the moment this is strictly a forward feed process i.e. operates
    strictly from sky positions -> angular offsets -> pixels -> BMU frame.
    In the future this should be expanded to be agnostic and capable of
    operating in any direction. 
    """

    def __init__(
        self,
        center_coord: SkyCoord,
        sky_coords: SkyCoord,
        transform: Tuple[int, float],
        pixel_scale: Union[None, Angle, Tuple[Angle, Angle]] = u.arcsecond,
        wcs: Union[None, WCS] = None,
        img_size: Union[None, Tuple[int, int]] = None,
    ) -> None:
        """Create a new instance of the CoordinateTransformer. Turns positions within
        the sky-reference frame to the neuron-reference frame
        
        It is probably best practise to ensure each image has a reference pixel within
        the FoV to avoid and image distortions from projection effects. If each image has
        a local coordinate-reference position then supplying a pixel_scale should be sufficent. 
        If the coordinate-reference position is sufficently far away then a WCS object
        should be supplied. 

        Arguments:
            center_coord {SkyCoord} -- RA/Dec center position to rotate around
            sky_coord {SkyCoord} -- RA/Dec coordinates to transform
            transform {Tuple[int, float]} -- PINK provided spatial transform
        
        Keyword Arguments:
            pixel_scale {Union[None,Angle, Tuple[Angle, Angle]]} -- PINK scale of the neuron (default: {1*u.arcsecond})
            wcs {Union[None, WCS]} -- WCS object that corresponds to an image of interest (default: {None})
        """
        self.center_coord = center_coord
        self.pixel_scale = pixel_scale
        self.transform = transform
        self.wcs = wcs

        self.coords: Dict[str, np.ndarray] = {}
        self.coords["sky"] = sky_coords
        self.coords["offsets-angular"] = self.__spherical_offsets()

        if self.wcs is not None:
            self.coords["offsets-pixel"] = self.__wcs_skycoord_to_pixels()
        elif self.pixel_scale is not None:
            self.coords["offsets-pixel"] = self.__delta_sky_to_pixels()
        else:
            raise ValueError(
                "Missing WCS and pixel scale information. At least one has to be provided. "
            )

        self.coords["offsets-neuron"] = self.__spatial_transform_points()

    def __spherical_offsets(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the spherical offsets between the centre point and
        the provided neighbours. If requested account for the pixel scale

        Returns:
            {Tuple[np.ndarray, np.ndarray]} -- The delta-offsets of transform_coords
            relative to the center_coord. If self.pixel_scale is None or `apply_pixel_scale`
            is False thet are returned as Angles, otherwise return in a pixel refernce frame
        """
        return self.center_coord.spherical_offsets_to(self.coords["sky"])

    def __delta_sky_to_pixels(self) -> Tuple[np.ndarray, np.ndarray]:
        """Transform the spherical offsets from the sky-reference frame to a
        pixel-reference frame. 
        
        Returns:
            Tuple[np.ndarray, np.ndarray] -- Coordinates in a pixel-reference frame
        """
        offsets = self.coords["offsets-angular"]

        if self.pixel_scale is None:
            return offsets

        pixel_scale = (
            self.pixel_scale
            if isinstance(self.pixel_scale, tuple)
            else (self.pixel_scale, self.pixel_scale)
        )
        pixel_scale = tuple([u.pixel_scale(ps / u.pixel) for ps in pixel_scale])

        # RA increases right-to-left. The opposite of an array
        offsets = (
            -offsets[0].to(u.pixel, pixel_scale[0]),
            offsets[1].to(u.pixel, pixel_scale[1]),
        )

        return offsets

    def __wcs_skycoord_to_pixels(self) -> Tuple[np.ndarray, np.ndarray]:
        """Use the wcs.utils skycoord_to_pixel to derive the pixel positions
        
        Returns:
            Tuple[np.ndarray, np.ndarray] -- Pixels coordinate postions of nearby sources
        """
        pix = skycoord_to_pixel(self.coords["sky"], self.wcs) * u.pixel
        center_pix = skycoord_to_pixel(self.center_coord, self.wcs) * u.pixel

        pix = [p - c for p, c in zip(pix, center_pix)]

        return pix

    def __spatial_transform_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the PINK spatial transformation to pixel coordinates. Internally the 
        angle is negated so that rotation occurs in the same clockwise direction as PINK. 
        
        Returns:
            Tuple[np.ndarray, np.ndarray] -- Spatially transformed points
        """
        flip, angle = self.transform

        # Rotation matrix rotates anti-clockwise
        angle = -angle

        rotation = np.array(
            ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
        )

        coords = np.array(self.coords["offsets-pixel"])
        transform_coords = (coords.T @ rotation).T

        if flip == 1:
            transform_coords[1] = -transform_coords[1]

        return (transform_coords[0] * u.pixel, transform_coords[1] * u.pixel)


class Filter:
    """Object to maintain the results of a cookie-cutter type projection of 
    spatially transformed coordinates and their evaluation against their BMU filter
    """

    def __init__(
        self, coords: CoordinateTransformer, neuron: Annotation, channel: int = 0
    ):
        """Creates a new filter instance to project spatially transformed coordinates onto an annotated neuron
        
        Arguments:
            coords {CoordinateTransformer} -- Set of spatially transformed coordinates to project and evaluate
            neuron {Annotation} -- Previously annotated neuron with saved filters
        
        Keyword Arguments:
            channel {int} -- Which filter channel to project coordinates onto (default: {0})
        """

        self.coords = coords
        self.neuron = neuron
        self.channel = channel

        self._evaluate()

    def _evaluate(self):
        """Perform the projection of spatially transformed positions onto  a neuron

        TODO: Consider moving the logic to a separate function for access outside of this class?
        """
        filter = self.neuron.filters[self.channel]
        size = np.array(filter.shape)
        center = size / 2

        ra_pix, dec_pix = self.coords.coords["offset-neuron"]

        # Recall C-style ordering of two-dimensional arrays. Shouldn't matter
        # too much as PINK enforces equal image dimensions for x/y axes
        ra_pix = ra_pix.value + center[1]
        dec_pix = dec_pix.value + center[0]

        # TODO: Consider desired behaviour when it comes to rounding pixel coordinates
        self.labels = filter[dec_pix.astype(int), ra_pix.astype(int)]