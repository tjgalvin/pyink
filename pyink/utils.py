"""Contains basic common utility functions that could foreseeably
be used outside of pyink
"""
from typing import List, Set, Dict, Tuple, Optional, Union, Any, TYPE_CHECKING

import numpy as np
from scipy.ndimage import rotate
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle


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
    ) -> None:
        """Create a new instance of the CoordinateTransformer. Turns positions within
        the sky-reference frame to the neuron-reference frame
        
        Arguments:
            center_coord {SkyCoord} -- RA/Dec center position to rotate around
            sky_coord {SkyCoord} -- RA/Dec coordinates to transform
            transform {Tuple[int, float]} -- PINK provided spatial transform
        
        Keyword Arguments:
            pixel_scale {Union[None,Angle, Tuple[Angle, Angle]]} -- PINK scale of the neuron (default: {1*u.arcsecond})
        """
        self.center_coord = center_coord
        self.pixel_scale = pixel_scale
        self.transform = transform

        self.coords: Dict[str, np.ndarray] = {}
        self.coords["sky"] = sky_coords
        self.coords["offsets-angular"] = self.__spherical_offsets()

        # TODO: Transform the angular offsets before they become pixel offsets

        if self.pixel_scale is not None:
            self.coords["offsets-pixel"] = self.__delta_sky_to_pixels()
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

        TODO: Consider incorporating a proper WCS skycoord_to_pixel scheme? 
        
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

    def __spatial_transform_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the PINK spatial transformation to pixel coordinates.
        
        Returns:
            Tuple[np.ndarray, np.ndarray] -- Spatially transformed points
        """
        flip, angle = self.transform

        # TODO: This in principal should not be needed, but it seems to
        # TODO: be necessary to align the points onto the neuron. The
        # TODO: reason why this is required needs to be identified.
        # TODO: From what I can tell in simple test code the rotation matrix
        # TODO: below rotates clockwise - the same way PINK does.
        angle = -angle

        rotation = np.array(
            ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
        )

        coords = np.array(self.coords["offsets-pixel"])
        transform_coords = (coords.T @ rotation).T

        if flip == 1:
            transform_coords[1] = -transform_coords[1]

        return (transform_coords[0], transform_coords[1])
