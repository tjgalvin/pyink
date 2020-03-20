import numpy as np
import struct as st
import os as os
import sys as sys
from itertools import product


class ImageWriter:
    """Helper to create a image binary file
    """

    def __init__(
        self, binary_path, data_layout, data_header, comment=None, clobber=False
    ):
        if not clobber and os.path.exists(binary_path):
            raise ValueError(f"Path Exists: {binary_path}")

        self.fd = open(binary_path, "wb")
        if comment is not None:
            self.fd.write(f"# {comment}")

            # Ensure there is a header deliminter
            if "# END OF HEADER\n" not in comment:
                self.fd.write("# END OF HEADER\n")

        self.header_start = self.fd.tell()
        self.header_end = None
        self.data_layout = data_layout
        self.no_dimensions = len(data_header)
        self.data_header = data_header
        self.count = 0

        self.create_header()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.update_count()
        self.close()

    def create_header(self):

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
        curr = self.fd.tell()

        # 2 * 32 bits, integer is 32 bits
        self.fd.seek(self.header_start + 3 * 4)
        self.fd.write(st.pack("i", int(self.count)))

        self.fd.seek(curr)

    def close(self):
        self.update_count()
        self.fd.close()

    def add(self, d, update_count=True):
        assert self.data_header == d.shape, ValueError("Shape do not match")

        d.astype("f").tofile(self.fd)

        if update_count:
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
    def __init__(self, path):
        if not os.path.exists(path):
            raise ValueError(f"{path} does not exist")

        self.path = path
        self.header_start = header_offset(self.path)

        self.read_header()

        data_shape = (self.header[3], *self.img_shape)
        self.data = np.memmap(
            self.path,
            offset=self.data_start,
            dtype=self.dtype,
            order="C",
            shape=data_shape,
        )

    def read_header(self):
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
    def img_rank(self):
        return self.header[5]

    @property
    def img_shape(self):
        if self.img_rank == 2:
            return (*self.header[6], 1)
        return self.header[6]


class SOM:
    def __init__(self, path):
        self.path = path
        self.offset = header_offset(self.path)

        # Initiated to None, but set in read_header. Here as reminder
        self.header = None
        self.data_start = None
        self.read_header()
        self.read_data()

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
    def som_rank(self):
        return self.header[4]

    @property
    def som_shape(self):
        if self.som_rank == 2:
            return (*self.header[5], 1)
        return self.header[5]

    @property
    def neuron_rank(self):
        return self.header[7]

    @property
    def neuron_shape(self):
        if self.neuron_rank == 2:
            return (1, *self.header[8])
        return self.header[8]

    @property
    def neuron_size(self):
        return np.prod(self.neuron_shape)

    def read_data(self):
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

        self.data = np.swapaxes(self.data, 1, 2)

    def read_header(self):
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
    def __init__(self, path):
        self.path = path
        self.offset = header_offset(self.path)

        # Initiated to None, but set in read_header. Here as reminder
        self.header = None
        self.data_start = None
        self.read_header()

        data_shape = (
            self.header[3],
            self.som_shape[0],
            self.som_shape[1] * self.som_shape[2],  # Get rid of depth
        )
        self.data = np.memmap(
            self.path,
            dtype=self.dtype,
            order="C",
            offset=self.data_start,
            shape=data_shape,
        )

    def __iter__(self):
        """Produce a key iterating over the image axis
        """
        for i in range(self.header[3]):
            yield i

    def read_header(self):
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
    def som_rank(self):
        return self.header[5]

    @property
    def som_shape(self):
        if self.som_rank == 2:
            return (*self.header[6], 1)
        return self.header[6]

    def bmu(self, idx=None, squeeze=True, order="C"):
        """Identify the position of the BMU for each image
        """
        if np.issubdtype(type(idx), np.integer):
            idx = [idx]
        data = self.data if idx is None else self.data[idx]

        bmu = np.array(
            np.unravel_index(
                np.argmin(np.reshape(data, (data.shape[0], -1), order=order), axis=1),
                data.shape[1:],
            )
        )

        if squeeze:
            return np.squeeze(bmu.T)
        else:
            return bmu.T


class Transform:
    def __init__(self, path):
        self.path = path
        self.offset = header_offset(self.path)

        # Initiated to None, but set in read_header. Here as reminder
        self.header = None
        self.data_start = None
        self.read_header()
        self.dtype = np.dtype([("flip", np.int8), ("angle", np.float64)])

        print(self.header)
        print(self.som_shape)

        data_shape = (
            self.header[2],
            self.som_shape[0],
            self.som_shape[1] * self.som_shape[2],  # Get rid of depth
        )
        self.data = np.memmap(
            self.path,
            dtype=self.dtype,
            order="C",
            offset=self.data_start,
            shape=data_shape,
        )

    def __iter__(self):
        """Produce a key iterating over the image axis
        """
        for i in range(self.header[3]):
            yield i

    def read_header(self):
        with open(self.path, "rb") as of:
            of.seek(self.offset)

            header = st.unpack("i" * 5, of.read(4 * 5))
            ver, file_type, no_imgs, som_layout, som_rank = header

            som_shape = st.unpack("i" * som_rank, of.read(4 * som_rank))

            self.data_start = of.tell()
            self.header = (ver, file_type, no_imgs, som_layout, som_rank, som_shape)

    @property
    def som_rank(self):
        return self.header[4]

    @property
    def som_shape(self):
        if self.som_rank == 2:
            return (*self.header[5], 1)
        return self.header[5]

    def bmu(self, idx=None):
        """Identify the position of the BMU for each image
        """
        data = self.data if idx is None else self.data[idx]

        bmu = np.array(
            np.unravel_index(
                np.argmin(data.reshape(data.shape[0], -1), axis=1), data.shape[1:]
            )
        )

        return bmu.T
