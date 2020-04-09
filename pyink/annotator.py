"""Classes to assist in the annotation of SOM neurons
"""
from typing import List, Set, Dict, Tuple, Optional, Union, TYPE_CHECKING
from collections import defaultdict
import pickle
import logging

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from skimage.segmentation import flood

import pyink.utils as pu

marker_style = ["ro", "g*", "yv"]
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class Annotation:
    """Class to retain annotation information applied on neurons
    """

    def __init__(self, neuron: np.ndarray):
        """Create an annotation instance to manage neurion
        
        Arguments:
            neuron {np.ndarray} -- Image of the neuron of the SOM
        """
        self.neuron: np.ndarray = neuron
        self._init_containters()

    def _init_containters(self):
        """Initialise the clicks and filter attributes. This
        is provided as a self-contained method for widgets
        """
        self.clicks: Dict[int, list] = defaultdict(list)

        self.filters: Dict[int, np.ndarray] = {
            k: np.zeros_like(self.neuron[k]).astype(np.bool)
            for k in np.arange(self.neuron.shape[0])
        }

    def evaluate_points(self, data: np.ndarray, index: int) -> np.ndarray:
        """Evaluate whether a set of data points fall withing a filter
        
        Arguments:
            data {np.ndarray} -- Transformed data points
            index {int} -- Index of filter to compare
        
        Returns:
            np.ndarray -- Boolean array of whether data points fall within a region
        """
        pass

    def transform_evaluate_points(
        self, data: np.ndarray, transform: Tuple[int, float], index: int
    ) -> np.ndarray:
        """Transform a set of datapoints and evaluate whether they fall within a filter
        
        Arguments:
            data {np.ndarray} -- data points representing sources in a pixel frame of reference
            transform {tuple[int, float]} -- transformation to apply to datapoints
            index {int} -- index of the filter to compare
        
        Returns:
            np.ndarray -- Boolean array of whether data points fall within a region
        """
        pass


# -------------------------------------
def overlay_clicks(results: Annotation, ax: plt.Axes, index: int = None):
    """Plot the markers from an Annotation instance onto the mask axes
    
    Arguments:
        results {Annotation} -- Container of current click information
        mask_ax {matplotlib.Axes} -- Instance of the axes panel

    Keyword Arguments:
        index {int} -- Index of the clicks to overlap
    """
    channels = [index,] if index is not None else results.clicks

    for chan in channels:
        marker = marker_style[chan]
        points = results.clicks[chan]

        for p in points:
            ax.plot(p[1], p[0], marker, ms=12)


def seed_fill_img(neuron_img: np.ndarray, clicks: List[tuple]):
    """Using specified seed points, create a filter with a flood fill style
    
    Arguments:
        neuron_mask {np.ndarray} -- Image of the neuron. Single channel, so shape is (y, x)
        clicks {List[tuple]} -- All clicks in the form (y, x, sigma)

    Raises:
        TypeError: Raised when neuron_img is not two-dimensional (only a single channel )
    
    Returns:
        np.ndarray -- Combined mask of clicked regions
    """
    if not len(neuron_img.shape) == 2:
        raise TypeError(
            f"neuron_img has to be two-dimensional, passed shape is {neuron_img.shape}"
        )
    master_mask = np.zeros_like(neuron_img).astype(np.bool)

    for click in clicks:
        tmp_mask = neuron_img > click[2] * neuron_img.std()
        # at the time of writing flood needs floats to be float64
        tmp_mask = flood(
            tmp_mask.astype(np.float64), (int(click[0] + 0.5), int(click[1] + 0.5))
        )
        master_mask = master_mask | tmp_mask

    return master_mask


def sigma_update_mask(results: Annotation, callback: "Callback") -> np.ndarray:
    """Update the mask region
    
    Arguments:
        results {Annotation} -- Annotation for current neurion
        callback {[type]} -- matplotlib structure for communitication
    
    Returns:
        np.ndarray -- masked array
    """
    index = callback.last_index
    logger.info(f"Last index retained is {index}")
    logger.debug(f"Result clicks {index} before: {results.clicks[index]}")

    if len(results.clicks[index]) == 0:
        logger.debug("No clicks to flood with. Returning empty mask. ")

        return np.zeros_like(results.neuron[index])

    results.clicks[index][-1][-1] = callback.sigma

    logger.debug(f"Result neuron {index} type: {type(results.neuron[index])}")
    logger.debug(f"Result clicks {index} type: {type(results.clicks[index])}")
    logger.debug(f"Result clicks {index}: {results.clicks[index]}")
    img_mask = seed_fill_img(results.neuron[index], results.clicks[index])
    results.filters[index] = img_mask

    return img_mask


def delete_markers(ax: plt.Axes):
    """Remove any overlaid markers from an Axes
    
    Arguments:
        ax {plt.Axes} -- Axes object to remove markers from
    """
    for line in ax.lines:
        line.set_marker(None)


def make_fig1_callbacks(
    callback: "Callback",
    results: Annotation,
    fig1: plt.Figure,
    axes: Union[plt.Axes, np.ndarray],
    mask_axes: Union[plt.Axes, np.ndarray],
):
    """Create the function handlers to pass over to the plt backends
    
    Arguments:
        callback {Callback} -- Class to have hard references for communication between backend and code
        results {Annotation} -- Store annotationg information for each of the neurons
        fig1 {plt.Figure} -- Instance to plt figure to plot to
        axes {plt.Axes} -- List of active axes objectd on figure. 
        mask_axes {plt.Axes} -- List of active axes objectd on figure for the masks

    Returns:
        callables -- Return the functions to handle figure key press and button press events
    """

    neuron = results.neuron

    def fig1_press(event: matplotlib.backend_bases.KeyEvent):
        """Capture the keyboard pressing a button
        
        Arguments:
            event {matplotlib.backend_bases.KeyEvent} -- Keyboard item pressed
        """
        index = np.argwhere(axes.flat == event.inaxes)[0, 0]
        mask_ax = mask_axes[index]

        logger.debug(f"Event type: {type(event)}")

        if event.key == "n":
            logger.info("Moving to next neuron")
            callback.next_move = "next"
            plt.close(fig1)

        if event.key == "b":
            logger.info("Moving back to previous neuron")
            callback.next_move = "back"
            plt.close(fig1)

        elif event.key == "c":
            logger.warn("Clearing clicks")
            results._init_containters()

            mask_im = results.filters[0]  # Will always be at least 1 neuron

            mask_ax.clear()  # Clears axes limits
            mask_ax.imshow(mask_im)

            overlay_clicks(results, mask_ax)

            for ax in axes:
                for line in ax.lines:
                    line.set_marker(None)

            fig1.canvas.draw_idle()

        elif event.key == "d":
            logger.info("Removing last click")

            if len(results.clicks[index]) > 0:
                results.clicks[index].pop(-1)

                delete_markers(axes.flat[index])
                overlay_clicks(results, axes.flat[index], index=index)

                img_mask = sigma_update_mask(results, callback)

                mask_ax.imshow(img_mask)
                delete_markers(mask_ax)
                overlay_clicks(results, mask_ax, index=index)

                fig1.canvas.draw_idle()

        elif event.key == "q":
            logger.info("Exiting...")
            callback.next_move = "quit"
            plt.close(fig1)

        elif event.key == "u":
            callback.live_update = not callback.live_update
            logger.info(f"Live update set to {callback.live_update}")

        elif event.key in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            results.type = event.key

            fig1.suptitle(f"{results.key} - Label: {results.type}")
            fig1.canvas.draw_idle()

        elif event.key == "down":
            callback.sigma = callback.sigma - 0.5
            logger.info(f"Sigma moved to {callback.sigma}")

            if len(results.clicks) > 0 and callback.live_update:
                img_mask = sigma_update_mask(results, callback)

                mask_ax.imshow(img_mask)
                overlay_clicks(results, mask_ax)

                fig1.canvas.draw_idle()

        elif event.key == "up":
            callback.sigma = callback.sigma + 0.5
            logger.info(f"Sigma moved to {callback.sigma}")

            if len(results.clicks) > 0 and callback.live_update:
                img_mask = sigma_update_mask(results, callback)

                mask_ax.imshow(img_mask)
                overlay_clicks(results, mask_ax)

                fig1.canvas.draw_idle()

    def fig1_button(event: matplotlib.backend_bases.MouseEvent):
        """Capture the mouse button press
        
        Arguments:
            event {matplotlib.backend_bases.Evenet} -- Item for mouse button press
        """
        logger.debug(f"Event fired: {event}")
        logger.debug(f"Event type: {type(event)}")

        index = np.argwhere(axes.flat == event.inaxes)[0, 0]
        mask_ax = mask_axes[index]
        callback.last_index = index

        if event.button != 1:
            return

        if fig1.canvas.manager.toolbar.mode != "":
            logger.warn(f"Toolbar mode is {fig1.canvas.manager.toolbar.mode}")
            return

        if event.xdata != None and event.ydata != None and event.inaxes != mask_ax:
            results.clicks[index].append([event.ydata, event.xdata, callback.sigma])

            logger.info(f"Click position: {int(event.ydata+0.5), int(event.xdata+0.5)}")
            logger.info(
                f"Channel {index} now {len(results.clicks[index])} clicks recored"
            )

            img_mask = seed_fill_img(results.neuron[index], results.clicks[index])
            results.filters[index] = img_mask

            mask_ax.imshow(img_mask)
            overlay_clicks(results, mask_ax)

            for ax in axes.flat:
                if ax != mask_ax:
                    ax.plot(event.xdata, event.ydata, "go", ms=12)

        fig1.canvas.draw_idle()

    def lasso_onselect(verts: np.ndarray):
        """Event handler for the lasso widget
        
        Arguments:
            verts {np.ndarray} -- Point data supplied by matplotlib
        """
        if fig1.canvas.manager.toolbar.mode != "":
            logger.warn(f"Toolbar mode is {fig1.canvas.manager.toolbar.mode}")
            return

        index = callback.last_index
        neuron = results.neuron[index]
        mask_ax = mask_axes[index]

        xx, yy = np.meshgrid(np.arange(neuron.shape[0]), np.arange(neuron.shape[1]))
        pix = np.vstack((xx.flatten(), yy.flatten())).T

        # Select elements in original array bounded by selector path:
        path = Path(verts)
        indicies = path.contains_points(pix, radius=1)
        logger.debug(f"Indicies shape: {indicies.shape}")
        logger.debug(f"Number of filters: {len(results.filters)}")

        mask = np.zeros_like(results.filters[index])

        logger.debug(f"Mask shape: {mask.shape}")
        logger.debug(f"Pix shape: {pix.shape}")
        mask[pix[:, 1], pix[:, 0]] = indicies

        results.filters[index] = mask | results.filters[index]

        mask_ax.imshow(results.filters[index])

        fig1.canvas.draw_idle()

    return fig1_press, fig1_button, lasso_onselect


class Callback:
    """Helper class to retain items from the matplotlib callbacks. Only a 
    weak reference is retained, so is modified to create a new copy (say an int)
    the reference is lost/garbage collected once matplotlib finishes
    """

    def __init__(self):
        self.next_move: str = None
        self.sigma: float = 2
        self.last_index: int = 0  # Always atleast one axes
        self.live_update: bool = True
        self.currect_axes: int = None


class Annotator:
    """Class to drive the interactive annotation of SOM neurons
    """

    def __init__(
        self,
        som: Union[str, pu.SOM],
        results: Dict[tuple, Annotation] = None,
        save: Union[bool, str, None] = True,
    ):
        """An object to help manage and annotate a SOM and its neurons
        
        Arguments:
            som {Union[str, pu.SOM]} -- SOM file to annotate as a PINK binary
        
        Keyword Arguments:
            results {Dict[tuple, Annotation]} -- Mapping for neurons and their corresponding Annotations (default: {None})
            save {Union[bool, str, None]} -- Action to save (default: {True})
        
        Returns:
            [type] -- [description]
        """
        self.som = pu.SOM(som) if isinstance(som, str) else som
        self.results: Dict[tuple, Annotation] = {} if results is None else results
        self.save: Union[bool, str, None] = False

        logger.info(f"Loaded SOM {som}...")
        if save == True:
            self.save = f"{self.som.path}.annotation"
        elif isinstance(save, str):
            self.save = save

    def annotate_neuron(
        self, key: tuple, return_callback: bool = False, cmap: str = "bwr"
    ):
        """Perform the annotation for a specified neuron
        
        Arguments:
            key {tuple} -- Index of the neuron within the SOM
        
        Keyword Arguments:
            return_callback {bool} -- return matplotlib action information (default: {False})
            cmap {str} -- colour map style (default: {'bwr'})
        
        Returns:
            [Callback, Annotation] -- matplotlib action information, neuron annotation
        """
        neuron = self.som[key]

        ant = self.results[key] if key in self.results.keys() else Annotation(neuron)

        no_chans = neuron.shape[0]

        fig1_callback = Callback()

        fig1, (axes, mask_axes) = plt.subplots(
            2, no_chans, sharex=True, sharey=True, squeeze=False
        )

        fig1_key, fig1_button, lasso_select = make_fig1_callbacks(
            fig1_callback, ant, fig1, axes, mask_axes
        )
        fig1.canvas.mpl_connect("key_press_event", fig1_key)
        fig1.canvas.mpl_connect("button_press_event", fig1_button)

        lassos = [LassoSelector(ax, lasso_select, button=3) for ax in axes]

        mask_ax = axes.flat[-1]

        for i, (n, ax) in enumerate(zip(neuron, axes.flat)):
            ax.imshow(np.sqrt(n), cmap=cmap)
            overlay_clicks(ant, ax, index=i)

        for i, mask_ax in enumerate(mask_axes):
            mask_ax.imshow(ant.filters[i])
            overlay_clicks(ant, mask_ax, index=i)

        plt.show()

        if return_callback:
            return fig1_callback, ant
        else:
            return ant

    def interactive_annotate(self):
        """Interate over neurons in the som and annotate
        """

        neurons: List[tuple] = [k for k in self.som]

        idx: int = 0
        while idx < len(neurons):

            key: tuple = neurons[idx]
            callback, ant = self.annotate_neuron(key, return_callback=True)

            if self.save is not None:
                self.save_annotations(path=self.save)

            if callback.next_move == "next":
                idx += 1
                self.results[key] = ant

            elif callback.next_move == "back":
                self.results[key] = ant
                if idx >= 1:
                    idx -= 1
                else:
                    logger.warn("Can't move back. Position zero.")

            elif callback.next_move == "quit":
                break

    def save_annotations(self, path: str = None):
        """Save the Annotator instance as a pickle file
        
        Keyword Arguments:
            path {str} -- Output path to write to (default: {None})
        """
        if path is None:
            path = f"{self.som.path}.annotations"

        with open(path, "wb") as out_file:
            logger.info(f"Saving to {path}")
            pickle.dump(self.results, out_file)

    @classmethod
    def load(cls, som: Union[str, pu.SOM], path: str):
        """Load a existing pickled Annotator object
        
        Arguments:
            path {str} -- Path to unpickle
        """
        with open(path, "rb") as infile:
            results = pickle.load(infile)

        return cls(som, results=results)
