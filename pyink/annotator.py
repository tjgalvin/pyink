"""Classes to assist in the annotation of SOM neurons
"""
from typing import List, Set, Dict, Tuple, Optional, Union, Callable, TYPE_CHECKING
from collections import defaultdict
import pickle
import logging

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector, CheckButtons, TextBox
from matplotlib.path import Path
from skimage.segmentation import flood

import pyink as pu

marker_style = ["ro", "g*", "yv"]

logger = logging.getLogger(__name__)

ANT_SUFFIX = "results.pkl"
# Used to record unqiue combinations of regions
PRIMES = {
    0: 2,
    1: 3,
    2: 5,
    3: 7,
    4: 11,
    5: 13,
    6: 17,
    7: 19,
    8: 23,
    9: 29,
    10: 31,
    11: 37,
    12: 41,
    13: 43,
    14: 47,
    15: 53,
    16: 59,
    17: 61,
    18: 67,
    19: 71,
}


class Annotation:
    """Class to retain annotation information applied on neurons
    """

    labels: List[Tuple[str, int]] = []

    def __repr__(self) -> str:
        """Neat string representation
        
        Returns:
            str -- description for printing
        """
        return f"{self.neuron.shape}"

    def __init__(self, neuron: np.ndarray):
        """Create an annotation instance to manage neurion
        
        Arguments:
            neuron {np.ndarray} -- Image of the neuron of the SOM
        """
        self.neuron: np.ndarray = neuron
        self._init_containters()

    def _init_containters(self):
        """Initialise the clicks and filter attributes. This
        is provided as a self-contained method for widgets to call
        """
        self.clicks: Dict[int, list] = defaultdict(list)

        self.filters: Dict[int, np.ndarray] = {
            k: np.zeros_like(self.neuron[k]).astype(np.bool)
            for k in np.arange(self.neuron.shape[0])
        }

        # recording lasso regions separately just in case in post-processing
        # it is required, something maybe recalculated after
        self.lasso_filters: Dict[int, list] = defaultdict(list)

    def evaluate_points(self, data: np.ndarray, index: int) -> np.ndarray:
        """Evaluate whether a set of data points fall withing a filter
        
        Arguments:
            data {np.ndarray} -- Transformed data points
            index {int} -- Index of filter to compare
        
        Returns:
            np.ndarray -- Boolean array of whether data points fall within a region
        """
        raise NotImplementedError(
            "This is a stub method that may be expanded at a later date."
        )

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
        raise NotImplementedError(
            "This is a stub method that may be expanded at a later date."
        )

    def resolve_label(self, label_value: int) -> Tuple[str, ...]:
        """Given a label integer value, work out the corresponding labels that were activated by the user. 
        
        Arguments:
            label_value {int} -- Desired label value to look for
        
        Returns:
            Tuple[str, ...] -- The string labels corresponding to that label
        """
        primes = [i for i in PRIMES.values() if label_value % i == 0]

        # Reverse the labels so a prime number resolves to a string
        lookup: Dict[int, str] = {}
        for l in self.labels:
            lookup[l[1]] = l[0]

        return tuple([lookup[p] for p in primes])


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
            ax.plot(p[1], p[0], marker, ms=5)


def calculate_region_value(callback: "Callback") -> int:
    """Given a set of checkboxes and their activation state, calculate a unique
    value to store in the filter pixels
    
    Arguments:
        callback {Callback} -- Structure to communicate with the matplotlib backend
    
    Returns:
        int -- Unique value to store as region. If no checkboxes axes of checkboxes activate return 1. 
    """
    if callback.checkbox is None:
        logger.debug(f"No checkbox axes detected. Returning default vaue of 1. ")
        return 1

    logger.debug(f"Checkbox.get_status(): {callback.checkbox.get_status()}")

    states = [
        PRIMES[i]
        for i, state in enumerate(callback.checkbox.get_status())
        if state == True
    ]

    logger.debug(f"Number of active boxes {len(states)}")

    if len(states) > 0:
        val = np.prod(states).astype(int)
        logger.debug(f"Returning product {val}. ")
        return val
    else:
        logger.debug(f"No active boxes detected. Return default value of 1. ")
        return 1


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
    master_mask = np.zeros_like(neuron_img).astype(np.int)

    for click in clicks:
        tmp_mask = neuron_img > click[3] * neuron_img.std()
        # at the time of writing flood needs floats to be float64
        tmp_mask = flood(
            tmp_mask.astype(np.float64), (int(click[0] + 0.5), int(click[1] + 0.5))
        )
        master_mask[tmp_mask] = click[2]

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
    button_axes: Union[None, np.ndarray] = None,
):
    """Create the function handlers to pass over to the plt backends
    
    Arguments:
        callback {Callback} -- Class to have hard references for communication between backend and code
        results {Annotation} -- Store annotationg information for each of the neurons
        fig1 {plt.Figure} -- Instance to plt figure to plot to
        axes {plt.Axes} -- List of active axes objectd on figure. 
        mask_axes {plt.Axes} -- List of active axes objectd on figure for the masks
        button_axes {Union[None, np.ndarray]} -- If labeling is enabled, this will be the checkbox and textbox axes. None otherwise (default: {None})

    Returns:
        callables -- Return the functions to handle figure key press and button press events
    """
    if button_axes is not None:
        logger.debug(f"button_axes is not empty.")

    neuron = results.neuron

    def fig1_press(event: matplotlib.backend_bases.KeyEvent):
        """Capture the keyboard pressing a button
        
        Arguments:
            event {matplotlib.backend_bases.KeyEvent} -- Keyboard item pressed
        """
        if button_axes is not None and event.inaxes in button_axes:
            logger.debug("Key press captured in button_axes. Discarding. ")
            return

        if event.inaxes in axes:
            index = np.argwhere(axes.flat == event.inaxes)[0, 0]
        elif event.inaxes in mask_axes:
            index = np.argwhere(mask_axes.flat == event.inaxes)[0, 0]
        else:
            logger.debug("Event not in a meaningful axes window. ")
            return

        mask_ax = mask_axes[index]

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

        if not event.inaxes in axes[:-1]:
            logger.debug(
                f"Click captured but not in image neuron subplot. Discarding. "
            )
            return

        index = np.argwhere(axes.flat == event.inaxes)[0, 0]
        logger.debug(f"Click Index is {index}, {type(index)}")
        if button_axes is not None and axes[index] in button_axes:
            logger.debug(f"Click event in CheckBox or TextBox axes")
            return

        mask_ax = mask_axes[index]
        callback.last_index = index

        if event.button != 1:
            return

        if fig1.canvas.manager.toolbar.mode != "":
            logger.warn(f"Toolbar mode is {fig1.canvas.manager.toolbar.mode}")
            return

        if event.xdata != None and event.ydata != None and event.inaxes != mask_ax:
            results.clicks[index].append(
                [
                    event.ydata,
                    event.xdata,
                    calculate_region_value(callback),
                    callback.sigma,
                ]
            )
            logger.debug(f"Recorded click information: {results.clicks[index][-1]}")

            logger.info(f"Click position: {int(event.ydata+0.5), int(event.xdata+0.5)}")
            logger.info(
                f"Channel {index} now {len(results.clicks[index])} clicks recored"
            )

            img_mask = seed_fill_img(results.neuron[index], results.clicks[index])

            for i, lasso_img in enumerate(results.lasso_filters[index]):
                logger.debug(f"Adding lasso image filter to img_mask")
                img_mask += lasso_img

            results.filters[index] = img_mask
            logger.debug(
                f"img_mask statistics: Min {np.min(img_mask)}, Max {np.max(img_mask)}"
            )

            mask_ax.imshow(img_mask)
            overlay_clicks(results, mask_ax)

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

        mask = results.filters[index]
        lasso_region = np.zeros_like(mask)

        logger.debug(f"Mask shape: {mask.shape}")
        logger.debug(f"Pix shape: {pix.shape}")
        mask[pix[indicies, 1], pix[indicies, 0]] = calculate_region_value(callback)
        lasso_region[pix[:, 1], pix[:, 0]] = indicies

        results.filters[index] = mask
        results.lasso_filters[index].append(lasso_region)

        mask_ax.imshow(results.filters[index])

        fig1.canvas.draw_idle()

    return fig1_press, fig1_button, lasso_onselect


def make_box_callbacks(
    callback: "Callback",
    results: Annotation,
    fig1: plt.Figure,
    button_axes: Tuple[plt.Axes, plt.Axes],
) -> Tuple[Callable, Callable]:
    """Make the callbacks that will be provided to the checkbox and textbox
    
    Arguments:
        callback {Callback} -- Handler to manage communication between matplotlib and widgets
        results {Annotation} -- Neuron annotation structure
        fig1 {plt.Figure} -- figure canvas
        button_axes {Tuple[plt.Axes, plt.Axes]} -- References to the button axes
        checkbox {matplotlib.widgets.CheckButtons} -- checkbox widget return by matplotlib
        checkbox {matplotlib.widgets.TextBox} -- textbox widget return by matplotlib

    Returns:
        Tuple[Callable, Callable] -- callback functions for checkbox and textbox widgets
    """

    def textbox_submit(text: str):
        """Action for when new label is added
        
        Arguments:
            text {str} -- New label entered by user
        """
        logger.debug(f"TextBox submit event captured. Submitted text {text}")
        if text == "":
            logger.warn(f"Textbox submission is empty. Ignoring. ")
            return
        if text in [l[0] for l in results.labels]:
            logger.warn(f"{text} already added as a label. Ignoring. ")
            return
        if len(PRIMES.keys()) == len(results.labels):
            logger.info(
                f"There are already {len(results.labels)}. Not adding any more. "
            )
            return

        results.labels.append((text, PRIMES[len(results.labels)]))
        callback.textbox.text = ""

        callback.checkbox.ax.clear()
        callback.checkbox = CheckButtons(
            callback.checkbox.ax, [l[0] for l in results.labels]
        )
        callback.checkbox.on_clicked(checkbox_activations)

        callback.textbox.ax.clear()
        callback.textbox = TextBox(callback.textbox.ax, "")
        callback.textbox.on_submit(textbox_submit)

        fig1.canvas.draw_idle()

    def checkbox_activations(label: str):
        """Callback function for when a label is selected
        
        Arguments:
            label {str} -- The label selected by user
        """
        logger.debug(f"Checkbox label has been clicked, {label}")
        callback.checkbox_activate = callback.checkbox.get_status()

    return textbox_submit, checkbox_activations


class Callback:
    """Helper class to retain items from the matplotlib callbacks. Only a 
    weak reference is retained, so is modified to create a new copy (say an int)
    the reference is lost/garbage collected once matplotlib finishes
    """

    # TODO: Consider making checkbox activations persistent between neurons somhow
    # TODO: Suspect will have to pass around get_status type messages?
    checkbox: Union[matplotlib.widgets.CheckButtons] = None
    textbox: Union[matplotlib.widgets.TextBox] = None

    def __init__(self):
        self.next_move: str = None
        self.sigma: float = 2
        self.last_index: int = 0  # Always atleast one axes
        self.live_update: bool = True
        self.currect_axes: int = None
        self.checkbox: Union[matplotlib.widgets.CheckButtons] = None
        self.textbox: Union[matplotlib.widgets.TextBox] = None
        self.checkbox_activate: Tuple = None


class Annotator:
    """Class to drive the interactive annotation of SOM neurons
    
    TODO: Add a `update_annotated_neuron` function? Although `annotate_neuron`
          can be called specifically, is this usable if a single neuron is 
          requested to be updated easily?
    """

    def __repr__(self) -> str:
        """Neat string representation of an Annotator instance
        
        Returns:
            str -- string description of the Annotator class
        """
        s: str = f"Annotated SOM: {self.som.path} \n"

        try:
            no_ant: int = len(self.results.keys())
            s += f"Annotated {no_ant} neurons rescorded \n"
            s += f"Keys of results: {self.results.keys()}\n"
        except:
            s += "No saved annotation\n"

        return s

    def __init__(
        self,
        som: Union[str, pu.SOM],
        results: Union[str, Dict[tuple, Annotation]] = None,
        save: Union[bool, str, None] = True,
    ):
        """An object to help manage and annotate a SOM and its neurons
        
        Arguments:
            som {Union[str, pu.SOM]} -- SOM file to annotate as a PINK binary
        
        Keyword Arguments:
            results {Union[str, Dict[tuple, Annotation]]} -- A path to a pickled Annotator object, an existing Dict of appropriate mappings for neurons and their corresponding Annotations. If None a new Dict is created. (default: {None})
            save {Union[bool, str, None]} -- Action to save (default: {True})
        
        Returns:
            [type] -- [description]
        """
        self.som = pu.SOM(som) if isinstance(som, str) else som
        logger.info(f"Loaded SOM {som}...")

        if results is None:
            self.results: Dict[tuple, Annotation] = {}
        elif isinstance(results, str):
            with open(results, "rb") as infile:
                self.results = pickle.load(infile)
        elif isinstance(results, dict):
            self.results = results
        else:
            raise ValueError(
                f"Expected either a path of a pickled Annotator or a Dict are accepted, got {type(results)}"
            )

        self.save: Union[bool, str, None] = False
        if save == True:
            self.save = f"{self.som.path}.annotation"
        elif isinstance(save, str):
            self.save = save

    def annotate_neuron(
        self,
        key: tuple,
        return_callback: bool = False,
        cmap: str = "bwr",
        labeling: bool = False,
    ):
        """Perform the annotation for a specified neuron
        
        TODO: Add options to enable / disable certain features like
        lines, colour scales etc. 

        Arguments:
            key {tuple} -- Index of the neuron within the SOM
        
        Keyword Arguments:
            return_callback {bool} -- return matplotlib action information (default: {False})
            cmap {str} -- colour map style (default: {'bwr'})
            labeling {bool} -- enabling the interactive creation and assignment of labels (default: {False})

        Returns:
            [Callback, Annotation] -- matplotlib action information, neuron annotation
        """
        neuron = self.som[key]
        logger.debug(f"Loaded {key} neuron, of shape {neuron.shape}")

        ant = self.results[key] if key in self.results.keys() else Annotation(neuron)

        no_chans = neuron.shape[0]

        fig1_callback = Callback()

        no_label = 1 if labeling else 0

        fig1, (axes, mask_axes) = plt.subplots(
            2,
            no_chans + no_label,
            figsize=(5 * no_chans, 8),
            sharex=False,
            sharey=False,
            squeeze=False,
        )

        # Manually link the axes
        ax_base = axes[0]
        ax_base.get_shared_x_axes().join(*axes[:-no_label], *mask_axes[:-no_label])
        ax_base.get_shared_y_axes().join(*axes[:-no_label], *mask_axes[:-no_label])
        ax_base.set(title=f"{key}")

        logger.debug(f"Axes shape: {axes.shape}")
        logger.debug(f"Mask_axes shape: {mask_axes.shape}")

        button_axes = None
        if labeling:
            button_axes = np.array((axes[-1], mask_axes[-1]))
            logger.debug(f"Creating button_axes.")
            fig1_callback.checkbox = CheckButtons(
                button_axes[0], [l[0] for l in ant.labels], None
            )
            fig1_callback.textbox = TextBox(button_axes[1], "")

            textbox_submit, checkbox_change = make_box_callbacks(
                fig1_callback, ant, fig1, button_axes,
            )
            fig1_callback.textbox.on_submit(textbox_submit)
            fig1_callback.checkbox.on_clicked(checkbox_change)

        fig1_key, fig1_button, lasso_select = make_fig1_callbacks(
            fig1_callback, ant, fig1, axes, mask_axes, button_axes=button_axes
        )
        fig1.canvas.mpl_connect("key_press_event", fig1_key)
        fig1.canvas.mpl_connect("button_press_event", fig1_button)

        lassos = [LassoSelector(ax, lasso_select, button=3) for ax in axes[:-no_label]]

        for i, (n, ax) in enumerate(zip(neuron, axes.flat)):
            logger.debug(f"Loading neuron image channel {i+1} of {neuron.shape[0]}")
            ax.imshow(n, cmap=cmap)
            overlay_clicks(ant, ax, index=i)
            ax.axvline(n.shape[0] / 2, color="black", ls="--", alpha=0.5)
            ax.axhline(n.shape[1] / 2, color="black", ls="--", alpha=0.5)
            ax.grid(which="major", axis="both", color="white", alpha=0.4)

        for i, _ in enumerate(ant.filters.keys()):
            logger.debug(
                f"Loading neuron filter channel {i+1} of {len(ant.filters.keys())}"
            )
            mask_ax = mask_axes[i]
            mask_ax.imshow(ant.filters[i])
            overlay_clicks(ant, mask_ax, index=i)

        plt.show()

        if return_callback:
            return fig1_callback, ant
        else:
            return ant

    def interactive_annotate(self):
        """Interate over the neurons in a SOM and call annotate_neuron. This method
        manages progressing over all neurons upon the SOM surface and saving their results. 
        Provided certain user prompts, certain actions are taken. 
        """

        neurons: List[tuple] = [k for k in self.som]

        idx: int = 0
        while idx < len(neurons):

            key: tuple = neurons[idx]
            callback, ant = self.annotate_neuron(
                key, return_callback=True, labeling=True
            )

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

    def save_annotation_results(self, path: str = None):
        """Save the Annotator results, which is a Dict, as a pickle file. For the moment
        only the result structure is saved, and not the entire Annotator instance. This is 
        to avoid potentially issues of pickling a memory mapped file (which is the underlying
        interface around all of the PINK binary file classes).
        
        Keyword Arguments:
            path {str} -- Output path to write to (default: {None})
        """
        if path is None:
            path = f"{self.som.path}.{ANT_SUFFIX}"

        with open(path, "wb") as out_file:
            logger.info(f"Saving to {path}")
            pickle.dump(self.results, out_file)
