# -*- coding: utf-8 -*-

"""
Versatile module for environmental data processing and analysis, providing 
functions for image handling, grid generation, and color mapping. This module 
includes functionalities to decompose images into RGB values, define grids, 
load images, map RGB values to real-world data, and process legend images.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..api.types import Tuple, Optional, List, Dict, Callable
from ..decorators import EnsureFileExists 
from ..tools.depsutils import ensure_pkg 
from ..tools.validator import validate_positive_integer 

__all__= [
     'decompose_image_to_rgb',
     'define_grid',
     'load_image',
     'make_grid_data',
     'process_legend',
     'process_legend2',
     'rgb_to_z'
    ]

@EnsureFileExists
def load_image(image_path: str, grayscale: bool = False) -> np.ndarray:
    """
    Load an image from the specified path.

    Parameters
    ----------
    image_path : str
        Path to the image file. The path should be a string that points 
        to a valid image file format (e.g., PNG, JPEG).

    grayscale : bool, optional
        Whether to convert the image to grayscale. Default is `False`. 
        If set to `True`, the image will be converted to grayscale by 
        averaging the RGB values.

    Returns
    -------
    np.ndarray
        Loaded image as a numpy array. If `grayscale` is `True`, the 
        returned array will be two-dimensional. Otherwise, it will be 
        three-dimensional with RGB channels.

    Notes
    -----
    The function reads the image from the specified `image_path` using 
    `matplotlib.pyplot.imread`. If the `grayscale` parameter is `True`, 
    the image is converted to grayscale by computing the mean of the 
    RGB values along the third axis:
    
    .. math::
        I_{gray} = \frac{1}{3} (R + G + B)
    
    where :math:`I_{gray}` is the intensity of the grayscale image, and 
    :math:`R`, :math:`G`, and :math:`B` are the red, green, and blue 
    channels, respectively.

    Examples
    --------
    >>> from gofast.tools.baseutils import load_image
    >>> img = load_image("path/to/image.png")
    >>> img_gray = load_image("path/to/image.png", grayscale=True)

    See Also
    --------
    matplotlib.pyplot.imread : Function to read an image from a file 
    into an array.
    
    numpy.mean : Function to compute the arithmetic mean along the 
    specified axis.

    References
    ----------
    .. [1] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. 
       Computing in Science & Engineering, 9(3), 90-95.
    """
    try:
        image = plt.imread(image_path)
        if grayscale:
            image = np.mean(image, axis=2)  # Convert to grayscale
        return image
    except Exception as e:
        raise IOError(f"Error loading image: {e}")

def define_grid(
    image: np.ndarray,
    grid_size: int, 
    padding: Optional[int] = 0, 
    return_extents: bool = False
    ) -> Tuple[int, int, Optional[Tuple[int, int, int, int]]]:
    """
    Define the grid dimensions based on the image size and grid cell size.

    Parameters
    ----------
    image : np.ndarray
        Loaded image as a numpy array. The array should be at least two-dimensional 
        (height, width) or three-dimensional (height, width, channels).

    grid_size : int
        Size of the grid cells in pixels. This should be a positive integer.

    padding : Optional[int], optional
        Optional padding to add around the image. Default is 0. The padding is 
        added equally to all sides of the image before dividing it into grid cells.

    return_extents : bool, optional
        Whether to return the extents of the padded image. Default is `False`. 
        If `True`, the function will return a tuple of extents 
        (xmin, ymin, xmax, ymax).

    Returns
    -------
    num_cells_x : int
        Number of grid cells in the x direction (width).

    num_cells_y : int
        Number of grid cells in the y direction (height).

    extents : Optional[Tuple[int, int, int, int]]
        The extents of the padded image as a tuple (xmin, ymin, xmax, ymax) if 
        `return_extents` is `True`. Otherwise, `None`.

    Notes
    -----
    The function calculates the number of grid cells by dividing the image 
    dimensions by the specified `grid_size`. If `padding` is specified, the 
    dimensions are increased accordingly before division. The extents are 
    returned if `return_extents` is `True`.

    Examples
    --------
    >>> from gofast.tools.baseutils import define_grid
    >>> image = np.random.rand(500, 500, 3)  # Example image
    >>> num_cells_x, num_cells_y = define_grid(image, 20)
    >>> num_cells_x, num_cells_y, extents = define_grid(
    ...       image, 20, padding=10, return_extents=True)

    See Also
    --------
    numpy.pad : Function to pad an array.

    References
    ----------
    .. [1] Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., 
       Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., 
       Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del R'ıo, J. F., Wiebe, M., 
       Peterson, P., Gérard-Marchant, P., Sheppard, K., Reddy, T., Weckesser, W., Abbasi, H., 
       Gohlke, C., & Oliphant, T. E. (2020). Array programming with NumPy.
       Nature, 585(7825), 357-362.
    """
    grid_size= validate_positive_integer (grid_size, )
    
    try:
        image_height, image_width = image.shape[:2]

        if padding:
            image_height += 2 * padding
            image_width += 2 * padding

        num_cells_x = image_width // grid_size
        num_cells_y = image_height // grid_size

        extents = None
        if return_extents:
            extents = (0 - padding, 0 - padding, image_width + padding,
                       image_height + padding)

        return num_cells_x, num_cells_y, extents
    except Exception as e:
        raise ValueError(f"Error defining grid: {e}")

@ensure_pkg("cv2", extra="'cv2' package is needed for processing legend.")
@EnsureFileExists
def process_legend(
    image_path: str, 
    min_value: float, max_value: float, 
    num_sections: Optional[int] = None
    ) -> Dict[Tuple[int, int, int], float]:
    """
    Process a legend image to map RGB values to corresponding real values.

    Parameters
    ----------
    image_path : str
        Path to the legend image file. The path should be a string that 
        points to a valid image file format (e.g., PNG, JPEG).

    min_value : float
        Minimum value represented in the legend. This should be a float 
        indicating the lowest value that the legend corresponds to.

    max_value : float
        Maximum value represented in the legend. This should be a float 
        indicating the highest value that the legend corresponds to.

    num_sections : Optional[int], optional
        Number of sections to divide the legend into. If `None`, the 
        function will automatically divide the legend based on the width 
        of the legend image. Default is `None`.

    Returns
    -------
    legend_mapping : Dict[Tuple[int, int, int], float]
        Dictionary mapping RGB tuples to real values. The keys are tuples 
        of three integers representing the RGB values, and the values are 
        the corresponding real values.

    Notes
    -----
    The function reads the legend image and extracts the RGB values. It 
    then divides the legend into sections either based on the specified 
    number of sections or automatically if `num_sections` is not provided.

    The mapping from RGB to real values is performed using linear 
    interpolation between the `min_value` and `max_value`:

    .. math::
        V_{real} = V_{min} + i \\cdot \\frac{V_{max} - V_{min}}{N - 1}

    where :math:`V_{real}` is the real value, :math:`V_{min}` is the 
    minimum value, :math:`V_{max}` is the maximum value, :math:`i` is the 
    section index, and :math:`N` is the number of sections.

    Examples
    --------
    >>> from gofast.geo.envmod import process_legend
    >>> legend_mapping = process_legend("path/to/legend_image.png", -360, 60,
                                        num_sections=10)
    >>> print(legend_mapping)

    See Also
    --------
    numpy.ndarray : For handling arrays of pixel values.
    cv2.imread : Function to read an image file into an array.
    cv2.cvtColor : Function to convert an image from one color space to another.

    References
    ----------
    .. [1] Clark, C. D., & Wilson, C. R. (2011). Color in scientific 
           visualization. Computers & Geosciences, 37(5), 648-655.
    """
    import cv2
    # Load the legend image
    legend_img = cv2.imread(image_path)
    legend_img = cv2.cvtColor(legend_img, cv2.COLOR_BGR2RGB)
    
    # Determine the number of sections
    if num_sections is None:
        num_sections = legend_img.shape[1]
    
    # Create the mapping dictionary
    legend_mapping = {}
    for i in range(num_sections):
        x = int(i * legend_img.shape[1] / num_sections)
        rgb_value = tuple(legend_img[0, x])
        real_value = min_value + i * (max_value - min_value) / (num_sections - 1)
        legend_mapping[rgb_value] = real_value
    
    return legend_mapping

@ensure_pkg("PIL", extra="'PIL' is needed for processing legend.")
@EnsureFileExists
def process_legend2(
    image_path: str, 
    min_value: float, 
    max_value: float, 
    num_sections: Optional[int] = None
    ) -> List[Tuple[Tuple[int, int, int], float]]:
    """
    Process the legend image to map RGB values to corresponding real values.

    Parameters
    ----------
    image_path : str
        Path to the legend image file. The path should be a string that 
        points to a valid image file format (e.g., PNG, JPEG).

    min_value : float
        The minimum value represented by the legend. This should be a float 
        indicating the lowest value that the legend corresponds to.

    max_value : float
        The maximum value represented by the legend. This should be a float 
        indicating the highest value that the legend corresponds to.

    num_sections : Optional[int], optional
        The number of sections to divide the legend into. If `None`, the 
        function will automatically divide the legend based on unique colors.

    Returns
    -------
    List[Tuple[Tuple[int, int, int], float]]
        A list of tuples where each tuple contains an RGB value and its 
        corresponding real value.

    Notes
    -----
    The function reads the legend image and extracts the RGB values. It then 
    divides the legend into sections either based on the specified number of 
    sections or automatically if `num_sections` is not provided.

    The mapping from RGB to real values is performed using linear interpolation 
    between the `min_value` and `max_value`:

    .. math::
        V_{real} = V_{min} + i \\cdot \\frac{V_{max} - V_{min}}{N - 1}

    where :math:`V_{real}` is the real value, :math:`V_{min}` is the minimum 
    value, :math:`V_{max}` is the maximum value, :math:`i` is the section 
    index, and :math:`N` is the number of sections.

    Examples
    --------
    >>> from gofast.geo.envmod import process_legend2
    >>> legend_mapping = process_legend2("path/to/legend_image.png", -360, 60,
                                         num_sections=10)
    >>> print(legend_mapping)

    See Also
    --------
    numpy.ndarray : For handling arrays of pixel values.
    PIL.Image : For processing image files.

    References
    ----------
    .. [1] Clark, C. D., & Wilson, C. R. (2011). Color in scientific 
           visualization. Computers & Geosciences, 37(5), 648-655.
    """
    from PIL import Image
    try: 
        # Load the legend image
        legend_image = Image.open(image_path)
        legend_image = legend_image.convert("RGB")
        legend_pixels = np.array(legend_image)
    
        # Flatten the image to 1D array for easier processing
        flat_pixels = legend_pixels.reshape(-1, 3)
    
        # Get unique RGB values
        unique_colors = np.unique(flat_pixels, axis=0)
    
        # If num_sections is not provided, use the unique colors count
        if num_sections is None:
            num_sections = len(unique_colors)
    
        # Calculate value range for each section
        value_range = np.linspace(min_value, max_value, num_sections)
    
        # Generate the mapping
        color_mapping = []
        for idx, color in enumerate(unique_colors):
            if idx < num_sections:
                value = value_range[idx]
            else:
                # Assign last range if more unique colors than sections
                value = value_range[-1]  
            color_mapping.append((tuple(color), value))
    
        return color_mapping
    
    except Exception as e:
        raise ValueError(f"Error processing legend image: {e}"
                         " Or use `process_legend` instead.")

def make_grid_data(
    image: np.ndarray, 
    grid_size: int, 
    num_cells_x: int, 
    num_cells_y: int, 
    rgb_to_z_func: Callable[[Tuple[int, int, int]], float], 
    save: bool = False, 
    output_path: Optional[str] = None
 ) -> pd.DataFrame:
    """
    Create a DataFrame with grid cell coordinates and Z values.

    Parameters
    ----------
    image : np.ndarray
        Loaded image as a numpy array. The array should have at least two 
        dimensions (height, width) for grayscale images, and three dimensions 
        (height, width, channels) for RGB images.

    grid_size : int
        Size of the grid cells in pixels. This should be a positive integer.

    num_cells_x : int
        Number of cells in the x direction. This should be calculated based 
        on the width of the image and the `grid_size`.

    num_cells_y : int
        Number of cells in the y direction. This should be calculated based 
        on the height of the image and the `grid_size`.

    rgb_to_z_func : Callable[[Tuple[int, int, int]], float]
        Function to map RGB values to Z values. This function should accept 
        an RGB tuple (three integers) and return a float representing the 
        corresponding Z value.

    save : bool, optional
        Whether to save the DataFrame to a CSV file. Default is `False`.

    output_path : Optional[str], optional
        Path to save the CSV file if `save` is `True`. This should be a 
        string representing the desired file path. Default is `None`.

    Returns
    -------
    pd.DataFrame
        DataFrame containing X, Y, and Z values for each grid cell.

    Notes
    -----
    The function iterates through the grid, extracts RGB values from the center
    of each cell, maps them to Z values using `rgb_to_z_func`, and stores the
    results in a DataFrame.

    The grid coordinates (X, Y) are calculated as:

    .. math::
        X = j \cdot \text{grid\_size}
    
        Y = i \cdot \text{grid\_size}

    where :math:`i` and :math:`j` are the row and column indices of the grid cells,
    respectively.

    The RGB value is extracted from the center of each grid cell and converted to a 
    Z value using the `rgb_to_z_func` function.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.geo.envmod import make_grid_data
    >>> def rgb_to_z(rgb):
    ...     return sum(rgb) / 3  # Example function
    >>> image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    >>> df = make_grid_data(image, 10, 10, 10, rgb_to_z, save=True,
                            output_path="output.csv")
    >>> print(df)

    See Also
    --------
    pandas.DataFrame : For handling tabular data.
    numpy.ndarray : For handling arrays of pixel values.

    References
    ----------
    .. [1] McKinney, W. (2010). Data structures for statistical computing in Python. 
           In Proceedings of the 9th Python in Science Conference (Vol. 445, pp. 51-56).
    """
    grid_data = []

    try:
        for i in range(num_cells_y):
            for j in range(num_cells_x):
                x = j * grid_size
                y = i * grid_size

                # Ensure we don't go out of bounds
                if (y + grid_size // 2) >= image.shape[0] or (
                        x + grid_size // 2) >= image.shape[1]:
                    continue

                # Get the RGB value at the center of the cell
                rgb_value = tuple(image[y + grid_size // 2, x + grid_size // 2])

                # Map the RGB value to Z value
                z_value = rgb_to_z_func(rgb_value)

                # Add the data to the list
                grid_data.append([x, y, z_value])

        # Convert to DataFrame
        grid_df = pd.DataFrame(grid_data, columns=["X", "Y", "Z"])

        # Save to CSV if required
        if save:
            if output_path is None:
                raise ValueError("Output path must be specified if save is True.")
            grid_df.to_csv(output_path, index=False)

        return grid_df

    except Exception as e:
        raise RuntimeError(f"Error creating grid dataframe: {e}")

def rgb_to_z(
        rgb_value: Tuple[int, int, int], 
        legend: Dict[Tuple[int, int, int], float], 
        interpolate: bool = True,
        interpolation_method: Optional[Callable[[float, float, float], float]] = None,
        default_value: Optional[float] = None
    ) -> Optional[float]:
    """
    Map an RGB value to a Z value based on a legend.

    Parameters
    ----------
    rgb_value : Tuple[int, int, int]
        RGB color value as a tuple of three integers representing the red, 
        green, and blue channels respectively. Each value should be in the 
        range 0 to 255.

    legend : Dict[Tuple[int, int, int], float]
        Dictionary mapping RGB color tuples to real values. The keys are 
        tuples of three integers representing RGB values, and the values 
        are the corresponding real values.

    interpolate : bool, optional
        Whether to interpolate between legend values if the exact RGB value 
        is not found. Default is `True`.

    interpolation_method : Optional[Callable[[float, float, float], float]], optional
        Function to perform interpolation between legend values. If `None`, 
        a default linear interpolation method is used. Default is `None`.

    default_value : Optional[float], optional
        Default value to return if the RGB value is not found in the legend 
        and interpolation is disabled. Default is `None`.

    Returns
    -------
    Optional[float]
        The corresponding real value from the legend. If the RGB value is not 
        found in the legend and interpolation is disabled, it returns the 
        `default_value`.

    Notes
    -----
    This function looks up the input `rgb_value` in the provided `legend` 
    dictionary to find the corresponding real value. If the RGB value is not 
    present in the legend and `interpolate` is `True`, the function uses the 
    specified `interpolation_method` to estimate the Z value. If no 
    interpolation method is provided, a default linear interpolation is used:

    .. math::
        Z = Z_1 + \\frac{Z_2 - Z_1}{RGB_2 - RGB_1} (RGB - RGB_1)

    where :math:`Z` is the interpolated Z value, :math:`Z_1` and :math:`Z_2` 
    are the known Z values, and :math:`RGB_1` and :math:`RGB_2` are the known 
    RGB values.

    Examples
    --------
    >>> from gofast.geo.envmod import rgb_to_z 
    >>> legend = {
    ...     (255, 0, 0): 0.0,     # Red
    ...     (0, 255, 0): 120.0,   # Green
    ...     (0, 0, 255): 240.0    # Blue
    ... }
    >>> rgb_value = (128, 128, 0)
    >>> z_value = rgb_to_z(rgb_value, legend, interpolate=False, default_value=-1)
    >>> print(z_value)
    -1

    See Also
    --------
    numpy.interp : For performing linear interpolation in numpy.

    References
    ----------
    .. [1] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. 
           Computing in Science & Engineering, 9(3), 90-95.
    """
    def linear_interpolation(rgb_1, rgb_2, z_1, z_2, rgb):
        """Default linear interpolation method."""
        return z_1 + (z_2 - z_1) * ((rgb - rgb_1) / (rgb_2 - rgb_1))

    try:
        # Check if the RGB value is directly in the legend
        if rgb_value in legend:
            return legend[rgb_value]
        
        # If interpolation is enabled, attempt to interpolate
        if interpolate:
            sorted_legend = sorted(legend.items(), key=lambda x: x[1])
            for i in range(len(sorted_legend) - 1):
                rgb_1, z_1 = sorted_legend[i]
                rgb_2, z_2 = sorted_legend[i + 1]
                if rgb_1 <= rgb_value <= rgb_2:
                    if interpolation_method:
                        return interpolation_method(rgb_1, rgb_2, z_1, z_2, rgb_value)
                    else:
                        return linear_interpolation(rgb_1, rgb_2, z_1, z_2, rgb_value)

        # Return default value if RGB value not found and interpolation is disabled
        return default_value
    except Exception as e:
        raise ValueError(f"Error in mapping RGB to real value: {e}")

@ensure_pkg("PIL", extra = "'PIL' is needed for decomposing image to RGB.")
@EnsureFileExists
def decompose_image_to_rgb(
        image_path: str, 
        return_mode: str = 'array', 
        resize: Optional[Tuple[int, int]] = None, 
        normalize: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Decompose an image into its RGB values.

    Parameters
    ----------
    image_path : str
        Path to the image file. The path should be a string that points 
        to a valid image file format (e.g., PNG, JPEG).

    return_mode : str, optional
        The format to return the RGB values. Options are 'array' for a single 
        numpy array of shape (height, width, 3) or 'split' for three separate 
        arrays for R, G, and B. Default is 'array'.

    resize : Optional[Tuple[int, int]], optional
        If provided, resize the image to the specified (width, height) before 
        decomposing. Default is `None`.

    normalize : bool, optional
        Whether to normalize the RGB values to the range [0, 1]. Default is `False`.

    Returns
    -------
    Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
        If `return_mode` is 'array', returns a single numpy array of shape 
        (height, width, 3) containing RGB values. If `return_mode` is 'split', 
        returns three separate numpy arrays for the R, G, and B channels.

    Notes
    -----
    The function reads the image from the specified `image_path`, optionally resizes 
    it, and decomposes it into its RGB components. The RGB values can be returned 
    as a single array or as separate arrays for each channel.

    Examples
    --------
    >>> from gofast.geo.envmod import decompose_image_to_rgb
    >>> rgb_array = decompose_image_to_rgb("path/to/image.png", return_mode='array')
    >>> r, g, b = decompose_image_to_rgb("path/to/image.png", return_mode='split')

    See Also
    --------
    numpy.ndarray : For handling arrays of pixel values.
    PIL.Image : For processing image files.

    References
    ----------
    .. [1] Clark, C. D., & Wilson, C. R. (2011). Color in scientific 
           visualization. Computers & Geosciences, 37(5), 648-655.
    .. [2] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. 
           Computing in Science & Engineering, 9(3), 90-95.
    """
    from PIL import Image
    
    try:
        # Load the image
        image = Image.open(image_path)
        
        # Resize the image if specified
        if resize is not None:
            image = image.resize(resize, Image.ANTIALIAS)
        
        # Convert the image to RGB
        image = image.convert('RGB')
        
        # Convert image to numpy array
        rgb_array = np.array(image)
        
        # Normalize the RGB values if specified
        if normalize:
            rgb_array = rgb_array / 255.0

        if return_mode == 'array':
            return rgb_array
        elif return_mode == 'split':
            r, g, b = rgb_array[:,:,0], rgb_array[:,:,1], rgb_array[:,:,2]
            return r, g, b
        else:
            raise ValueError("Invalid return_mode. Choose 'array' or 'split'.")
    
    except Exception as e:
        raise RuntimeError(f"Error decomposing image: {e}")




