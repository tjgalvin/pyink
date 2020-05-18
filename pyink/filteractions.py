from typing import List, Set, Dict, Tuple, Optional, Union, Any, Iterable
from concurrent.futures import ProcessPoolExecutor
import logging

import tqdm
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, Angle, search_around_sky
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel

from pyink.annotator import Annotation, Annotator
from pyink import SOMSet

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
        src_idx: np.ndarray = None,
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
            src_idx {np.ndarray[int]} -- Integer IDX (or similar) to uniquely identify objects from the source catalogue (default: {None})
        """
        self.center_coord = center_coord
        self.pixel_scale = pixel_scale
        self.transform = transform
        self.wcs = wcs
        self.src_idx = src_idx

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
        self,
        coords: CoordinateTransformer,
        neuron: Annotation,
        channel: int = 0,
        plot: bool = False,
        data: Any = None,
    ):
        """Creates a new filter instance to project spatially transformed coordinates onto an annotated neuron

        Arguments:
            coords {CoordinateTransformer} -- Set of spatially transformed coordinates to project and evaluate
            neuron {Annotation} -- Previously annotated neuron with saved filters
        
        Keyword Arguments:
            channel {int} -- Which filter channel to project coordinates onto (default: {0})
            plot {bool} -- Produce a plot of the filter and overlaid components (default: {False})
            data {Any} -- user specified that that may be carried forwarded to the collation process (default: {None})
        """

        self.coords = coords
        self.neuron = neuron
        self.channel = channel
        self.data = data

        self.ra_pix: np.ndarray = None
        self.dec_pix: np.ndarray = None

        self._evaluate()

        if plot:
            self.plot()

    def _evaluate(self):
        """Perform the projection of spatially transformed positions onto  a neuron

        TODO: Consider moving the logic to a separate function for access outside of this class?
        """
        filter = self.neuron.filters[self.channel]
        center = np.array(filter.shape) / 2

        ra_pix, dec_pix = self.coords.coords["offsets-neuron"]

        # Recall C-style ordering of two-dimensional arrays. Shouldn't matter
        # too much as PINK enforces equal image dimensions for x/y axes
        self.ra_pix = ra_pix.value + center[1]
        self.dec_pix = dec_pix.value + center[0]

        # TODO: Consider desired behaviour when it comes to rounding pixel coordinates
        # TODO: Resolve the index value to the labels
        self.product_labels = filter[self.dec_pix.astype(int), self.ra_pix.astype(int)]
        self.coord_labels = list(map(self.neuron.resolve_label, self.product_labels))

    def plot(
        self, figure: plt.Figure = None, axes: plt.Axes = None
    ) -> Union[plt.Figure, plt.Axes]:
        """Produce a basic diagnostic plot to examine where coordinates fall one a neuron
        
        Keyword Arguments:
            figure {plt.Figure} -- Figure object to plot onto. Will be created if None (default: {None})
            axes {plt.Axes} -- Axes object to plot onto. Will be created if None (default: {None})
        
        Returns:
            fig {plt.Figure} -- Figure object used for plotting
            ax {plt.Axes} -- Axes object used for plotting
        """
        if figure is None and axes is None:
            fig = plt.figure()
        if axes is None:
            ax = fig.add_subplot(111)

        ax.imshow(self.neuron.filters[self.channel])

        if self.product_labels is None:
            ax.plot(self.ra_pix, self.dec_pix, "ro")
        else:
            for u in np.unique(self.product_labels):
                try:
                    label = (
                        "Unassigned"
                        if u == 0
                        else ", ".join(self.neuron.resolve_label(u))
                    )
                except:
                    logger.debug(f"Label resolution failed")
                    label = u

                mask = u == self.product_labels
                ax.scatter(self.ra_pix[mask], self.dec_pix[mask], label=label)
                ax.legend()

        return fig, ax

    def coord_label_contains(self, label_val: Union[str, int]) -> np.ndarray:
        """Examines the labels corresponding to the pixel of the filter each coordinate fell in to
        see if a label value is a component. 
        
        Arguments:
            label_val {Union[str,int]} -- Target labels to comapare to
        
        Returns:
            np.ndarray -- Bool array corresponding to each coordinate contain the target label
        """
        if isinstance(label_val, str):
            return np.array(list(map(lambda x: label_val in x, self.coord_labels)))
        elif isinstance(label_val, int):
            return self.product_labels % label_val == 0
        else:
            raise ValueError(
                f"label_val may be either the string label or numeric value, received type {type(label_val)}"
            )


class FilterSet:
    """Object to manage the Filters of many sources across channels
    """

    def __init__(
        self,
        base_catalogue: SkyCoord,
        match_catalogues: Tuple[SkyCoord, ...],
        annotation: Annotator,
        som_set: SOMSet,
        cpu_cores: int = None,
        seplimit: Angle = 1 * u.arcminute,
        progress: bool = False,
        **ct_kwargs,
    ):
        """Create a set of Filters that describe the projection of sources onto their neurons. Other keyword-arguments are passed to `CoordinateTransformer`
        
        Arguments:
            base_catalogue {SkyCoord} -- The central positions of each image that corresponds to the Mapping and Transform of the currect `som_set`
            match_catalogues {Tuple[SkyCoord,...]} -- Catalogues of positions to project the neurons through. Each catalogue is assumed to correspond to a single channel
            annotation {Annotator} -- Previously defined annotated SOM
            som_set {pu.SOMSet} -- Reference to the SOM, Mapping and Transform binary files for the projection
        
        Keyword Arguments:
            cpu_cores {int} -- The number of CPU cores to use while projecting the filters. The default is to use one (and avoid the ProcessPoolExecutor) (default: {None})
            seplimit {Angle} -- Matching area for the `search_around_sky` matcher (defaul: {1*astropy.units.arcminute})
            progress {bool} -- Enable the `tqdm` progress bar updates (default: {False})
        """
        self.base_catalogue = base_catalogue
        self.annotation = annotation
        self.som_set = som_set
        self.cpu_cores = cpu_cores
        self.seplimit = seplimit
        self.progress = progress
        self.ct_kwargs = ct_kwargs

        assert isinstance(
            match_catalogues, tuple
        ), f"Expect tuple of SkyCoord catalogues, even if only of length 1. Received object of type {type(match_catalogues)}"
        self.match_catalogues = match_catalogues
        self.sky_matches = [
            search_around_sky(self.base_catalogue, mc, self.seplimit)
            for mc in self.match_catalogues
        ]

        self.filters = self.project()

    def cookie_cutter(self, channel: int, src_idx: int) -> Filter:
        """Creates filter for a given subject source and channel
        
        Arguments:
            channel {int} -- Desired channel (i.e. index of the desired `match_catalogue`)
            src_idx {int} -- Index of the subject source to project coordinates through
        
        Returns:
            Filter -- Filter instance collating the coordinate transformations and filter projections together
        """
        sky_matches = self.sky_matches[channel]
        sky_catalogue = self.match_catalogues[channel]

        center_pos = self.base_catalogue[src_idx]
        src_mask = sky_matches[0] == src_idx
        src_matches = sky_matches[1][src_mask]

        bmu = self.som_set.mapping.bmu(idx=src_idx)
        transform_key = (src_idx, *bmu)
        transform = self.som_set.transform.data[transform_key]

        spatial_transform = CoordinateTransformer(
            center_pos,
            sky_catalogue[src_matches],
            transform,
            src_idx=src_matches,
            **self.ct_kwargs,
        )
        coord_filter = Filter(
            spatial_transform, self.annotation.results[tuple(bmu)], channel=channel
        )

        return coord_filter

    def project(self):
        """Apply the cookie-cutter projection onto sources
        """
        len_base_cata = len(self.base_catalogue)
        srcs = np.arange(len_base_cata)
        channels = len(self.match_catalogues)

        filters = []
        for c in range(channels):

            def map_lambda(src_idx):
                # Unable to see a way of passing arguments that aren't iterable to `map`
                # Lets just do a closure around them
                print(src_idx)
                return self.cookie_cutter(c, src_idx)

            if not isinstance(self.cpu_cores, int):
                filters.append(
                    list(
                        tqdm.tqdm(
                            map(map_lambda, srcs),
                            disable=not self.progress,
                            total=len_base_cata,
                        )
                    )
                )
            else:
                raise NotImplementedError(
                    "Current implementation of multiple CPU cores is not working. Remove `cpu_cores` argument."
                )

                # When the below is run it just hangs. Example code that is not in a class suggests this use of list(tqdm(executor.map))
                # should work. Suspect the problem is the serialisation of the class instance and the corresponding catalogues?
                # Suggestion would be to make `cookie_cutter` an exposed function that accepts the `annotation` and `som_set` as
                # arguments (i.e. remove all `self.` in the function) and ammend the `map_lambda` to use this and packed the necessary
                # attributes as arguements in a closure
                with ProcessPoolExecutor(max_workers=self.cpu_cores) as executor:
                    f = list(
                        tqdm.tqdm(
                            executor.map(
                                map_lambda,
                                srcs,
                                chunksize=4096,  # len(srcs) // self.cpu_cores,
                            ),
                            disable=not self.progress,
                            total=len_base_cata,
                        )
                    )
                filters.append(f)

        return filters
