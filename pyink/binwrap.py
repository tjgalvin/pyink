"""A set of classes to interact with PINK binary mapping files
and help manage the corresponding utilities required when transforming
coordinates
"""

import struct as st
import os as os
import sys as sys
from itertools import product
from typing import List, Set, Dict, Tuple, Optional, Union, Any, Iterable, Sequence
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle

from .utils import pink_spatial_transform


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
            self.fd.write(bytes(f"# {comment}", "ascii"))

            if "# END OF HEADER\n" not in comment:
                self.fd.write(b"# END OF HEADER\n")

        self.header_start = self.fd.tell()
        self.header_end = None
        self.data_layout = data_layout
        self.no_dimensions = len(data_header)
        self.data_header = data_header
        self.count = 0

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
        self.update_count()
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

    def add(self, img: np.ndarray, nonfinite_check: bool = True):
        """Appends a new image to the end of the image file and updates the file header
        
        Arguments:
            img {np.ndarray} -- Image array of shance (z, y, x) to the file. If z = 1 it may be ignored. 
        
        Keyword Arguments:
            nonfinite_check {bool} -- Ensure that only finite values are in images, which would otherwise cause PINK to behave in strange ways (default: {True})
        
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

        self.count += 1
        self.update_count()


def header_offset(path):
    """Determine the offset required to ignore the header information
    of a PINK binary. The header format spec: lines with a '#' start are
    ignored until a '# END OF HEADER' is found. Only valid at beginning of
    file and 

    Arguments:
        path {str} -- Path to the pink binary file
    """
    with open(path, "rb") as fd:
        for line in fd:
            if not line.startswith(b"#"):
                return 0
            elif line == b"# END OF HEADER":
                return fd.tell()


def resolve_data_type(dtype):
    """Stub function to act as a lookup for the datatypes within binaries
    
    TODO: Expand this to be loaded as the appropriate datatypes in the binary classes
    TODO: Produce the reverse so the ImageWriter acts appropriately

    Arguments:
        header {tuple} -- Pink file header
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

    return types[dtype]


class ImageReader:
    """Object to help manage the interaction of a PINK image binary file
    """

    def __init__(self, path: str) -> None:
        """Establish a new ImageReader class around a PINK image binary file
        
        Arguments:
            path {str} -- Path to the PINK image binary
        
        Raises:
            ValueError: Raised when PINK image binary does not exists
        """

        if not os.path.exists(path):
            raise ValueError(f"{path} does not exist")

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

        return pink_spatial_transform(src_img, transform)

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
        with open(self.path, "rb") as of:
            of.seek(self.offset)

            header = st.unpack("i" * 5, of.read(4 * 5))
            ver, file_type, dtype, som_layout, som_rank = header

            som_shape = st.unpack("i" * som_rank, of.read(4 * som_rank))

            neuron_layout, neuron_rank = st.unpack("i" * 2, of.read(4 * 2))
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

            self.header = header
            self.data_start = of.tell()
            self.dtype = resolve_data_type(dtype)


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

    def __iter__(self):
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

    def bmu(
        self,
        idx=None,
        squeeze=True,
        return_idx: bool = False,
        return_tuples: bool = False,
    ) -> np.ndarray:
        """Return the BMU coordinate for each source image. This corresponds to
        the coordinate of the neuron on the SOM lattice with the smalled ED to
        a source image
        
        Keyword Arguments:
            idx {Union[int,np.ndarray[int]]} -- The index / indicies to look at. Will default return all (default: {None})
            squeeze {bool} -- Remove empty axes from the return np.ndarray (default: {True})
            return_idx {bool} -- Include the source index/indices as part of the returned structure (default: {True})
            return_tuples {bool} -- Return as a list of tuples (default: {False})

        Returns:
            np.ndarray -- Indices to the BMU on the SOM lattice of each source image
        """
        if np.issubdtype(type(idx), np.integer):
            idx = np.array([idx])

        data = self.data if idx is None else self.data[idx]

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
