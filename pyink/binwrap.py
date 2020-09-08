"""A set of classes to interact with PINK binary mapping files
and help manage the corresponding utilities required when transforming
coordinates
"""

import struct as st
import os as os
import sys as sys
import logging
import pickle
from itertools import product
from functools import lru_cache
from typing import (
    List,
    Set,
    Dict,
    Tuple,
    Optional,
    Union,
    Any,
    Generator,
    Iterable,
    Sequence,
)

import numpy as np
from matplotlib import pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle

import pyink as pu

logger = logging.getLogger(__name__)
REC_SUFFIX = ".records.pkl"


class ImageWriter:
    """Helper to create a image binary file
    """

    def __init__(
        self,
        binary_path: str,
        data_layout: int,
        data_header: Tuple[int, ...],
        comment: str = None,
        clobber: bool = False,
    ):
        """Creates a new ImageWriter object. 

        Note that it would be best practise to ensure that images are reprojected to a local reference pixel. 
        This currently is not performed by this classs. 
        
        Arguments:
            binary_path {str} -- Output path of the image binary file
            data_layout {int} -- Cartesian or hexagonal layout following the PINK specification
            data_header {Tuple[int, ...]} -- Image dimensions
        
        Keyword Arguments:
            comment {str} -- A comment to place in the file header (default: {None})
            clobber {bool} -- Overwrite binary if it already exists (default: {False})
        
        Raises:
            ValueError: Raised if clobber is not True and the output image file exists
        
        """
        if data_layout != 0:
            raise NotImplemented(
                f"Only cartesian layout modes (mode `0`) are currently supported. Supplied mode was {data_layout}"
            )

        if not clobber and os.path.exists(binary_path):
            raise ValueError(f"Path Exists: {binary_path}")

        self.fd = open(binary_path, "wb")
        if comment is not None:
            # NOTE: Not entirely convinced the header works expected. To be tested.
            # TODO: Consider getting `records` recorded in the header. Would require
            # TODO: padding the header once `records` has been established so as to no
            # TODO: overwrite the image data
            self.fd.write(bytes(f"# {comment}", "ascii"))

            if "# END OF HEADER\n" not in comment:
                self.fd.write(b"# END OF HEADER\n")

        self.header_start = self.fd.tell()
        self.header_end = None
        self.data_layout = data_layout
        self.no_dimensions = len(data_header)
        self.data_header = data_header
        self.count = 0
        self.binary_path = binary_path
        self.records: List[Any] = []

        self.create_header()

    def __enter__(self):
        """Beginning of the context manager
        
        Returns:
            ImageWriter -- Returning the ImageWriter instance
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit of the context manager. Will update the header and close the file descriptor
        """
        self.close()

    def create_header(self):
        """Internal function used to create a header to the PINK image file
        """

        self.fd.seek(self.header_start)
        self.fd.write(st.pack("i", 2))  # version number of PINK
        self.fd.write(st.pack("i", 0))  # Seems to be some delimiter?
        self.fd.write(st.pack("i", 0))  # `0' corresponds to float32

        self.fd.write(st.pack("i", int(self.count)))  # The number of images
        self.fd.write(st.pack("i", int(self.data_layout)))  # square vs hex
        self.fd.write(
            st.pack("i", int(self.no_dimensions))
        )  # number of dimensions in image
        for i in self.data_header:
            self.fd.write(st.pack("i", int(i)))

        self.header_end = self.fd.tell()

    def update_count(self):
        """Update the header field of file corresponding to the number of images
        """
        curr = self.fd.tell()

        # 2 * 32 bits, start of number of images field
        self.fd.seek(self.header_start + 3 * 4)
        self.fd.write(st.pack("i", int(self.count)))

        self.fd.seek(curr)

    def close(self):
        """Closes the file descriptor after updating the file header. 
        """
        self.update_count()
        self.fd.close()
        self.save_records()

    def add(
        self, img: np.ndarray, nonfinite_check: bool = True, attributes: Any = None
    ):
        """Appends a new image to the end of the image file and updates the file header
        
        Arguments:
            img {np.ndarray} -- Image array of shance (z, y, x) to the file. If z = 1 it may be ignored. 
        
        Keyword Arguments:
            nonfinite_check {bool} -- Ensure that only finite values are in images, which would otherwise cause PINK to behave in strange ways (default: {True})
            attributes {Any} -- Information to store in the `records` attribute for each file, which will be serialised alongside the binary file(default: {None})

        Raises:
            ValueError: Raised if not all pixels have finite values
        """
        assert self.data_header == img.shape, ValueError("Shape do not match")

        if nonfinite_check and np.sum(~np.isfinite(img)) > 0:
            raise ValueError(
                "Non-finite values have been detected in the input image. This will cause PINK to fail. "
            )

        # TODO: Support data-type encoding. This is planned in PINK but not implemented.
        img.astype("f").tofile(self.fd)

        if attributes is not None:
            self.records.append(attributes)

        self.count += 1
        self.update_count()

    def save_records(self, path: str = None) -> None:
        """Will pickle the records attribute if it is being used
        
        Keyword Arguments:
            path {str} -- Output path to save records. If undefined it is based on the path set (default: {None})
        """
        if path is None:
            path = f"{self.binary_path}{REC_SUFFIX}"

        if len(self.records) == self.count:
            with open(path, "wb") as out:
                logger.debug(f"Writing {path}")
                pickle.dump(self.records, out)


def header_offset(path: str) -> int:
    """Determine the offset required to ignore the header information
    of a PINK binary. The header format spec: lines with a '#' start are
    ignored until a '# END OF HEADER' is found. Only valid at beginning of
    file and 

    Arguments:
        path {str} -- Path to the pink binary file
    
    Returns:
        {int} -- Byte offset position of where the comment at the beginning of
                 the header ends
    """
    with open(path, "rb") as fd:
        for line in fd:
            if not line.startswith(b"#"):
                return 0
            elif line == b"# END OF HEADER":
                return fd.tell()

    raise ValueError("Malformed PINK header: Unbroken Comment Section")


def resolve_data_type(dtype: Union[int, type]) -> Union[type, int]:
    """Resolves a requested `int` value to its correspond data type expected by PINK
    
    Arguments:
        header {Union[int, type]} -- If `int` resolve to the corresponding data type set by PINK. Otherwise
                                    assume it is a type to resolve to its corresponding int. 
    
    Returns:
        {type} -- Resolved datatype
    
    """
    types = {
        0: np.float32,
        1: np.float64,
        2: np.int8,
        3: np.int16,
        4: np.int32,
        5: np.int64,
        6: np.uint8,
        7: np.uint16,
        8: np.uint32,
        9: np.uint64,
    }

    try:
        return types[dtype]  # type: ignore
    except KeyError:
        inverse_types = {v: k for k, v in types.items()}

        return inverse_types[dtype]


class ImageReader:
    """Object to help manage the interaction of a PINK image binary file
    """

    def __init__(self, path: str, record_path: str = None) -> None:
        """Establish a new ImageReader class around a PINK image binary file
        
        Arguments:
            path {str} -- Path to the PINK image binary
        
        Keyword Arguements:
            record_path {str} -- Path to a serialised list to accompany the image binary (default: {None})

        Raises:
            ValueError: Raised when PINK image binary does not exists
        """

        if not os.path.exists(path):
            raise ValueError(f"{path} does not exist")
        if record_path is None:
            record_path = f"{path}{REC_SUFFIX}"
        if os.path.exists(record_path):
            self.record_path = record_path
            with open(self.record_path, "rb") as rec:
                self.records = pickle.load(rec)

        self.path = path
        self.header_start = header_offset(self.path)

        self.__read_header()

        data_shape = (self.header[3], *self.img_shape)
        self.data = np.memmap(
            self.path,
            offset=self.data_start,
            dtype=self.dtype,
            order="C",
            mode="r",
            shape=data_shape,
        )

    def __read_header(self) -> None:
        """Process the file header associated with the PINK binary formate of an image binary. 
        """
        with open(self.path, "rb") as of:
            of.seek(self.header_start)

            header = st.unpack("i" * 6, of.read(4 * 6))
            ver, file_type, dtype, no_imgs, projection, img_rank = header

            img_shape = st.unpack("i" * img_rank, of.read(4 * img_rank))

            header = (ver, file_type, dtype, no_imgs, projection, img_rank, img_shape)

            self.header = header
            self.data_start = of.tell()
            self.dtype = resolve_data_type(dtype)

    @property
    def img_rank(self) -> int:
        """The number of image dimensions as described by the file header
        
        Returns:
            int -- the number of image dimensions of a single source image
        """
        return self.header[5]

    @property
    def img_shape(self) -> Tuple[Any, ...]:
        """The shape of a single source image. Will guarantee (channel, height, width) layout. 
        
        Returns:
            Tuple[int, int, int] -- The dimenions of a single source image
        """
        if self.img_rank == 2:
            return (*self.header[6], 1)
        return self.header[6]

    def transform(self, idx: int, transform: Tuple[np.int8, np.float32]) -> np.ndarray:
        """Apply a specified spatial transformation onto an image
        
        Arguments:
            idx {int} -- the source image index to transform
            transform {Tuple[np.int8, np.float32]} -- PINK spatial transformation specification
        
        Returns:
            np.ndarray -- Transformed image
        """
        if not np.issubdtype(type(idx), np.integer):
            raise TypeError("Currently only an integer index is supported. ")

        src_img = self.data[idx].copy()

        return pu.pink_spatial_transform(src_img, transform)

    def transform_images(
        self, idxs: Sequence[int], transforms: Sequence[Tuple[int, float]]
    ) -> np.ndarray:
        """Return a set of transformed images given a set of indicies and PINK transform
        
        Arguments:
            idxs {Sequence[int]} -- Index of images to return
            transforms {Sequence[Tuple[int, float]]} -- PINK transform function of each image
        
        Returns:
            np.ndarray -- Set of transformed images
        """
        assert len(idxs) == len(transforms), ValueError(
            f"Lengths of idxs and transforms are not equal"
        )

        return np.array(
            [self.transform(idx, transform) for idx, transform in zip(idxs, transforms)]
        )


class SOM:
    """A wrapper around the SOM binary files produced by PINK. Only supports 
    Cartesian SOM lattice layouts at the moment. For SOMs trained in three-dimensions
    the class will reshape the structure to be (X, YZ) as a matter of convenience. 
    
    TODO: Build in support for the Hexagonal SOM lattice layout
    """

    def __init__(self, path: str):
        """Create a wrapped around a PINK SOM binary file
        
        Arguments:
            path {str} -- Path to the PINK SOM binary file
        """
        self.path = path
        self.offset = header_offset(self.path)

        self.__read_header()
        self.__read_data()

    def __str__(self):
        """Neat string output for the SOM
        """
        return self.path

    def __iter__(self):
        """Yield the unique coordinates for each neuron
        """
        som_shape = (self.som_shape[0], self.som_shape[1] * self.som_shape[2])
        coords = product(*[np.arange(i) for i in som_shape])
        for c in coords:
            yield tuple(c)

    def __getitem__(self, key):
        """Accesor for neurons on the SOM. Note that the 
        depth dimension of the SOM lattice has been appended to the height
        dimension
        
        Arguments:
            key {tuple} -- Typical numeric index items to access the (C,X,Y) neuron.   
            slices are supported
        
        Raises:
            TypeError: Type error for elements of the key. Needs to be int or slice
        
        Returns:
            numpy.ndarray -- Channel/s of neuron/s
        """

        # SOM has been reshaped so channels always at front. This
        # initial slice ensure all channels are selected
        idxs = [slice(None, None, None)]

        # if not isinstance(key, tuple):
        #     key = (key,)

        if not hasattr(key, "__iter__"):
            key = (key,)

        for c, k in enumerate(key):
            dkey = self.neuron_shape[c + 1]

            if np.issubdtype(type(k), np.integer):
                idxs.append(slice(k * dkey, (k + 1) * dkey))
            elif isinstance(k, slice):
                kstart = k.start * dkey if k.start is not None else None
                kstop = k.stop * dkey if k.stop is not None else None
                kstep = k.step * dkey if k.step is not None else None
                idxs.append(slice(kstart, kstop, kstep))
            elif k is None:
                idxs.append(slice(None, None, None))
            else:
                raise TypeError(f"{k} is not an int or slice")

        return self.data[tuple(idxs)]

    @property
    def som_rank(self) -> int:
        """Return the number of SOM dimensions on the lattice
        
        Returns:
            int -- number of dimensions to the SOM lattice
        """
        return self.header[4]

    @property
    def som_shape(self) -> Tuple[Any, ...]:
        """The size of each dimension on fhe SOM lattice. This will be guaranteed to be three dimensions (width, height, depth)
        
        Returns:
            Tuple[int, int, int] -- The size of each dimension of the SOM lattice as described in the header
        """
        if self.som_rank == 2:
            return (*self.header[5], 1)
        return self.header[5]

    @property
    def neuron_rank(self) -> int:
        """Number of dimensions to each neuron
        
        Returns:
            int -- The number of dimensions to each neuron as described by the header
        """
        return self.header[7]

    @property
    def neuron_shape(self) -> Tuple[Any, ...]:
        """The size of each dimension for a neuron. This is guaranteed to be three dimensions. 
        
        Returns:
            Tuple[int, int, int] -- The size of each neuron described by the header
        """
        if self.neuron_rank == 2:
            return (1, *self.header[8])
        return self.header[8]

    @property
    def neuron_size(self) -> int:
        """The total number of pixels across all dimensions for a single neuron
        
        Returns:
            int -- The total number of pixels across all dimensions for a single neuron
        """
        return np.prod(self.neuron_shape)

    def __read_data(self):
        som_shape = self.som_shape
        neuron_shape = self.neuron_shape
        data_shape = som_shape + neuron_shape

        # For a cartesian SOM layout with neurons being cartesians PINK
        # will write out the SOM with the following array structure
        # X Y Z c h w
        # where X, Y, Z describe the structure of the lattice and
        # c h w describe the channels, width and height of the neurons.
        # I am not totally convinced the the w / h are the correct order
        # and I can't explicitly test because PINK enforces a quadrilateral
        # layout. This will matter when it comes to the transformations

        self.data = np.memmap(
            self.path,
            dtype=self.dtype,
            order="C",
            mode="r",
            offset=self.data_start,
            shape=data_shape,
        )

        self.data = np.moveaxis(self.data, 3, 0)

        self.data = self.data.reshape(
            neuron_shape[0],
            som_shape[0],
            som_shape[1] * som_shape[2],
            *neuron_shape[1:],
        )

        self.data = np.swapaxes(self.data, 2, 3)

        self.data = self.data.reshape(
            neuron_shape[0],
            som_shape[0] * neuron_shape[1],
            som_shape[1] * som_shape[2] * neuron_shape[2],
        )

    def __read_header(self):
        """Function to parse the header of the PINK SOM file
        """
        with open(self.path, "rb") as of:
            of.seek(self.offset)

            header = st.unpack("i" * 5, of.read(4 * 5))
            ver, file_type, dtype, som_layout, som_rank = header

            assert som_rank < 100, f"som_rank of {som_rank}, likely a malformed header?"

            som_shape = st.unpack("i" * som_rank, of.read(4 * som_rank))

            neuron_layout, neuron_rank = st.unpack("i" * 2, of.read(4 * 2))

            assert (
                neuron_rank < 100
            ), f"neuron_rank of {neuron_rank}, likely a malformed header?"

            neuron_shape = st.unpack("i" * neuron_rank, of.read(4 * neuron_rank))

            header = (
                ver,
                file_type,
                dtype,
                som_layout,
                som_rank,
                som_shape,
                neuron_layout,
                neuron_rank,
                neuron_shape,
            )

            logger.debug(
                f"data_type {dtype} resolves to type: {resolve_data_type(dtype)}"
            )

            self.header = header
            self.data_start = of.tell()
            self.dtype = resolve_data_type(dtype)

    def plot_neuron(
        self, neuron: Tuple[int, int] = (0, 0), fig: Union[None, plt.Figure] = None,
    ):
        """Plot a single neuron of the SOM across all channels"""
        shape = self.som_shape[:2]
        img_shape = self.neuron_shape[-2:]
        nchan = self.neuron_shape[0]

        # Define a reasonable size for the image
        if fig is None:
            base_size = [5, 5]
            base_size[np.argmin(shape)] = (
                base_size[np.argmax(shape)] * np.min(shape) / np.max(shape)
            )
            base_size[0] *= nchan
            fig, axes = plt.subplots(
                1,
                nchan,
                sharex=True,
                sharey=True,
                figsize=tuple(base_size),
                constrained_layout=True,
            )

        for chan, ax in enumerate(fig.axes):
            ny, nx = neuron
            ax.imshow(
                self.data[
                    chan,
                    ny * img_shape[0] : (ny + 1) * img_shape[0],
                    nx * img_shape[1] : (nx + 1) * img_shape[1],
                ]
            )

    def explore(self, start: Tuple[int, int] = (0, 0)):
        """Plotting routine to explore the SOM in manageable and easily navigable chunks.
        Could be absorbed into SOM.plot_neuron
        """
        self.plot_neuron(neuron=start)

        def press(event):
            print("press", event.key)
            sys.stdout.flush()
            if event.key == "up":
                if neuron[0] != 0:
                    neuron[0] -= 1
            if event.key == "down":
                if neuron[0] != self.som_shape[0] - 1:
                    neuron[0] += 1
            if event.key == "left":
                if neuron[1] != 0:
                    neuron[1] -= 1
            if event.key == "right":
                if neuron[1] != self.som_shape[1] - 1:
                    neuron[1] += 1
            print(f"Current neuron: {neuron}")

            xlim = fig.axes[0].set_xlim()
            ylim = fig.axes[0].set_ylim()
            self.plot_neuron(neuron=tuple(neuron), fig=fig)
            fig.canvas.draw()
            fig.axes[0].set_xlim(xlim)
            fig.axes[0].set_ylim(ylim)

        neuron = list(start)
        fig = plt.gcf()
        fig.canvas.mpl_connect("key_press_event", press)


class Mapping:
    """Wrapper around a PINK Map binary file. Includes some helper functions. Note that
    this only works for Cartesian SOM lattice layouts. For SOMs trained in three-dimensions
    the class will reshape the structure to be (X, YZ) as a matter of convenience. 
    """

    def __init__(self, path: str):
        """Create a new instance to manage a PINK mapping binary file. Currently
        only works for the Cartesian layout scheme.

        TODO: Expand to handle the hexagonal SOM lattice scheme
        
        Arguments:
            path {str} -- Path to the PINK mapping binary file
        """
        self.path = path
        self.offset = header_offset(self.path)

        self.data_start = None
        self.__read_header()

        data_shape = (
            self.header[3],
            self.som_shape[0],
            self.som_shape[1] * self.som_shape[2],  # Get rid of depth
        )
        self.data = np.memmap(
            self.path,
            dtype=self.dtype,
            order="C",
            mode="r",
            offset=self.data_start,
            shape=data_shape,
        )

    def __iter__(self) -> Generator:
        """Produce a index iterating over the input image axis
        """
        for i in range(self.header[3]):
            yield i

    def __read_header(self) -> None:
        """Process the initial header information of the Mapping binary file
        """
        with open(self.path, "rb") as of:
            of.seek(self.offset)

            header = st.unpack("i" * 6, of.read(4 * 6))
            ver, file_type, dtype, no_imgs, som_layout, som_rank = header

            assert som_rank < 100, f"som_rank of {som_rank}, likely a malformed header?"

            som_shape = st.unpack("i" * som_rank, of.read(4 * som_rank))

            self.data_start = of.tell()
            self.header = (
                ver,
                file_type,
                dtype,
                no_imgs,
                som_layout,
                som_rank,
                som_shape,
            )
            self.dtype = resolve_data_type(dtype)

    def __read_data_indicies(self, idx: Union[int, np.ndarray]) -> np.ndarray:
        """A helper function to provide a common interface to accessing data

        Keyword Arguments:
            idxs {Union[int, np.ndarray]} -- Desired indicies to obtain data from )default: {None})

        Returns:
            np.ndarray -- Returned slices for the prodived indicies
        """
        if np.issubdtype(type(idx), np.integer):
            idx = np.array([idx])

        return self.data if idx is None else self.data[idx]

    @property
    def som_rank(self) -> int:
        """The number of dimensions of the SOM lattice
        
        Returns:
            int -- the number of dimensions to the SOM lattice
        """
        return self.header[5]

    @property
    def som_shape(self) -> Tuple:
        """Return the structure of the SOM. This will pad it out to a
        three-element tuple if required. 
        
        Returns:
            tuple -- A three-element tuple describing the SOM lattice structure
        """
        if self.som_rank == 2:
            return (*self.header[6], 1)
        return self.header[6]

    @property
    def srcrange(self) -> np.ndarray:
        """Creates an array of indices corresponding to the source image axis
        
        Returns:
            np.ndarray[int] -- Indicies to access the source images
        """
        return np.arange(self.data.shape[0]).astype(np.int)

    @lru_cache(maxsize=16)
    def bmu(
        self,
        idx: Union[int, np.ndarray] = None,
        squeeze: bool = True,
        return_idx: bool = False,
        return_tuples: bool = False,
    ) -> np.ndarray:
        """Return the BMU coordinate for each source image. This corresponds to
        the coordinate of the neuron on the SOM lattice with the smalled ED to
        a source image
        
        Keyword Arguments:
            idx {Union[int,np.ndarray]} -- The index / indicies to look at. Will default return all (default: {None})
            squeeze {bool} -- Remove empty axes from the return np.ndarray (default: {True})
            return_idx {bool} -- Include the source index/indices as part of the returned structure (default: {True})
            return_tuples {bool} -- Return as a list of tuples (default: {False})

        Returns:
            np.ndarray -- Indices to the BMU on the SOM lattice of each source image
        """
        data = self.__read_data_indicies(idx)

        bmu = np.array(
            np.unravel_index(
                np.argmin(np.reshape(data, (data.shape[0], -1)), axis=1),
                data.shape[1:],
            )
        ).T

        if return_idx:
            idx = self.srcrange if idx is None else idx
            bmu = np.column_stack((idx, bmu))

        bmu = np.squeeze(bmu) if squeeze else bmu

        if return_tuples:
            bmu = [tuple(i) for i in bmu]

        return bmu

    def bmu_counts(self) -> np.ndarray:
        """Return counts of how often a neuron was the best matching unit
        
        Returns:
            np.ndarray -- Counts of how often neuron was the BMU
        """
        som_shape = self.data.shape[1:]
        counts = np.zeros(som_shape)
        coords = product(*[np.arange(i) for i in som_shape])

        bmu_keys = self.bmu(return_idx=True, squeeze=True)
        bz, by, bx = bmu_keys.T

        for c in coords:
            counts[c] = np.sum((by == c[0]) & (bx == c[1]))

        return np.squeeze(counts)

    @lru_cache(maxsize=16)
    def bmu_ed(self, idx: Union[int, np.ndarray] = None) -> np.ndarray:
        """Returns the similarity measure of the BMU for each source. The BMU will have the smallest
        similarity measure statistic for each image, so it is straight forward to search for. 
        
        Keyword Arguments:
            idx {Union[int, np.ndarray]} -- Indices of the images to pull information from

        Returns:
            np.ndarray -- The similarity measure statistic of each image to its BMU
        """
        data = self.__read_data_indicies(idx)

        ed = data.reshape(data.shape[0], -1).min(axis=1)

        return ed

    def images_with_bmu(self, key: Tuple[int, ...]) -> np.ndarray:
        """Return the indicies of images that have the `key` as their BMU

        Arguments:
            key {Tuple[int, ...]} -- Key of the neuron to search for

        Returns:
            np.ndarray -- Source indicies that have `key` as their BMU
        """
        bmu = self.bmu()

        # Create a mask for each axis
        mask = np.array([k == bmu[:, i] for i, k in enumerate(key)])

        # A match is when it is True across al dimensions for a single index
        mask = np.all(mask, axis=0)

        return np.array(np.squeeze(np.argwhere(mask)), ndmin=1)


class Transform:
    """Class to interact with PINK transformation files
    """

    def __init__(self, path: str):
        """Transform intialiser
        
        Arguments:
            path {str} -- path to the PINK transform file
        """
        self.path = path
        self.offset = header_offset(self.path)

        self.data_start = None
        self.__read_header()
        self.dtype = np.dtype([("flip", np.int8), ("angle", np.float32)])

        data_shape = (
            self.header[2],
            self.som_shape[0],
            self.som_shape[1] * self.som_shape[2],  # Get rid of depth
        )

        self.data = np.memmap(
            self.path,
            dtype=self.dtype,
            order="C",
            mode="r",
            offset=self.data_start,
            shape=data_shape,
        )

    def __iter__(self):
        """Produce a key iterating over the image axis
        """
        for i in range(self.header[3]):
            yield i

    def __read_header(self):
        """Parse the PINK Transformation file header
        """
        with open(self.path, "rb") as of:
            of.seek(self.offset)

            header = st.unpack("i" * 5, of.read(4 * 5))
            ver, file_type, no_imgs, som_layout, som_rank = header

            assert som_rank < 100, f"som_rank of {som_rank}, likely a malformed header?"

            som_shape = st.unpack("i" * som_rank, of.read(4 * som_rank))

            self.data_start = of.tell()
            self.header = (ver, file_type, no_imgs, som_layout, som_rank, som_shape)

    @property
    def som_rank(self) -> int:
        """The number of dimensions (height, width, depth) of the SOM lattice
        
        Returns:
            int -- the number of dimensions of the SOM lattice
        """
        return self.header[4]

    @property
    def som_shape(self) -> Tuple[Any, ...]:
        """The dimensions of the SOM lattice layout. Note that by default the depth is appended to the height dimension in the data array. 
        
        Returns:
            Tuple[int, int, int] -- The size of each dimension of the SOM layout as described in the header
        """

        if self.som_rank == 2:
            return (*self.header[5], 1)
        return self.header[5]


class SOMSet:
    """Container to hold a SOM, Mapping and Transform class instances
    """

    def __init__(
        self,
        som: Union[SOM, str],
        mapping: Union[Mapping, str],
        transform: Union[Transform, str],
    ):
        """Creates a container to hold a related SOM, Mapping and Transform set together
        
        Arguments:
            som {Union[SOM, str]} -- The desired SOM binary file. If `str` attempt to create a SOM instance
            mapping {Union[Mapping, str]} -- The desired Mapping binary file. If `str` attempt to create a Mapping instance
            transform {Union[Transform, str]} -- The desired Transform binary file. If `str` attempt to create a Transform instance
        """
        self.som = SOM(som) if isinstance(som, str) else som
        self.mapping = Mapping(mapping) if isinstance(mapping, str) else mapping
        self.transform = (
            Transform(transform) if isinstance(transform, str) else transform
        )
