# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""Provides a variety of image processing transformers, including resizing, 
normalization, color manipulation, and feature extraction for advanced 
image analysis tasks."""

from __future__ import division, annotations  
import os
import numpy as np 

import matplotlib.pyplot as plt  
from sklearn.base import BaseEstimator,TransformerMixin 

from ..tools._dependency import import_optional_dependency 

EMSG = (
        "`scikit-image` is needed for this transformer. Note `skimage`is "
        "the shorthand of `scikit-image`."
        )

__docformat__='restructuredtext'

__all__= [
    "ImageResizer", "ImageNormalizer", "ImageToGrayscale", "ImageBatchLoader",
    "ImageAugmenter", "ImageChannelSelector", "ImageFeatureExtractor",
    "ImageEdgeDetector", "ImageHistogramEqualizer", "ImagePCAColorAugmenter",
    ]

class ImageResizer(BaseEstimator, TransformerMixin):
    """
    Resize images to a specified size.

    Parameters
    ----------
    output_size : tuple of int
        The desired output size as (width, height).

    Examples
    --------
    >>> from skimage.transform import resize
    >>> resizer = ImageResizer(output_size=(128, 128))
    >>> image = np.random.rand(256, 256, 3)
    >>> resized_image = resizer.transform(image)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable 
        parameters and does nothing during fitting.

    transform(X)
        Resize input images to the specified output size.

    Notes
    -----
    ImageResizer is a transformer that resizes input images to a specified 
    output size. It is commonly used for standardizing image dimensions before
    further processing or analysis.

    """
    
    def __init__(self, output_size):
        """
        Initialize the ImageResizer transformer.

        Parameters
        ----------
        output_size : tuple of int
            The desired output size as (width, height).

        Returns
        -------
        None

        """
        self.output_size = output_size
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        import_optional_dependency("skimage", extra=EMSG)
        return self
    
    def transform(self, X):
        """
        Resize input images to the specified output size.

        Parameters
        ----------
        X : ndarray, shape (height, width, channels)
            Input image(s) to be resized.

        Returns
        -------
        resized_images : ndarray, shape (output_height, output_width, channels)
            Resized image(s) with the specified output size.

        """
        from skimage.transform import resize
        return resize(X, self.output_size, anti_aliasing=True)

class ImageNormalizer(BaseEstimator, TransformerMixin):
    """
    Normalize images by scaling pixel values to the range [0, 1].

    Parameters
    ----------
    None

    Examples
    --------
    >>> normalizer = ImageNormalizer()
    >>> image = np.random.randint(0, 255, (256, 256, 3), 
                                  dtype=np.uint8)
    >>> normalized_image = normalizer.transform(image)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no 
        trainable parameters and does nothing during fitting.

    transform(X)
        Normalize input images by scaling pixel values to the range [0, 1].

    Notes
    -----
    ImageNormalizer is a transformer that scales pixel values of input images 
    to the range [0, 1]. This is a common preprocessing step for images 
    before feeding them into machine learning models.

    """
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X):
        """
        Normalize input images by scaling pixel values to the range [0, 1].

        Parameters
        ----------
        X : ndarray, shape (height, width, channels)
            Input image(s) to be normalized.

        Returns
        -------
        normalized_images : ndarray, shape (height, width, channels)
            Normalized image(s) with pixel values scaled to the range [0, 1].

        """
        return X / 255.0


class ImageToGrayscale(BaseEstimator, TransformerMixin):
    """
    Convert images to grayscale.

    Parameters
    ----------
    keep_dims : bool, default=False
        If True, keeps the third dimension as 1 (e.g., (height, width, 1)).

    Examples
    --------
    >>> from skimage.color import rgb2gray
    >>> converter = ImageToGrayscale(keep_dims=True)
    >>> image = np.random.rand(256, 256, 3)
    >>> grayscale_image = converter.transform(image)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable 
        parameters and does nothing during fitting.

    transform(X)
        Convert input color images to grayscale.

    Notes
    -----
    ImageToGrayscale is a transformer that converts color images to grayscale.
    Grayscale images have only one channel, while color images typically 
    have three (red, green, and blue). This transformer allows you to 
    control whether the grayscale image should have a single channel or 
    retain a third dimension with a value of 1.

    """
    def __init__(self, keep_dims=False):
        """
        Initialize an ImageToGrayscale transformer.

        Parameters
        ----------
        keep_dims : bool, default=False
            If True, keeps the third dimension as 1 (e.g., (height, width, 1)).

        Returns
        -------
        ImageToGrayscale
            Returns an instance of the ImageToGrayscale transformer.

        """
        self.keep_dims = keep_dims
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        import_optional_dependency ('skimage', extra = EMSG )
        return self
    
    def transform(self, X):
        """
        Convert input color images to grayscale.

        Parameters
        ----------
        X : ndarray, shape (height, width, channels)
            Input color image(s) to be converted to grayscale.

        Returns
        -------
        grayscale_images : ndarray, shape (height, width, 1) or (height, width)
            Grayscale image(s) after conversion. The output can have a 
            single channel or retain the third dimension with a value of 1, 
            depending on the `keep_dims` parameter.

        """
        from skimage.color import rgb2gray
        grayscale = rgb2gray(X)
        if self.keep_dims:
            grayscale = grayscale[:, :, np.newaxis]
        return grayscale

class ImageAugmenter(BaseEstimator, TransformerMixin):
    """
    Apply random transformations to images for augmentation.

    Parameters
    ----------
    augmentation_funcs : list of callable
        A list of functions that apply transformations to images.

    Examples
    --------
    >>> from imgaug import augmenters as iaa
    >>> augmenter = ImageAugmenter(augmentation_funcs=[
        iaa.Fliplr(0.5), iaa.GaussianBlur(sigma=(0, 3.0))])
    >>> image = np.random.rand(256, 256, 3)
    >>> augmented_image = augmenter.transform(image)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable
        parameters and does nothing during fitting.

    transform(X)
        Apply random transformations to input images.

    Notes
    -----
    ImageAugmenter is a transformer that applies random augmentations to 
    input images. Data augmentation is commonly used in computer vision 
    tasks to increase the diversity of training data, which can lead to 
    improved model generalization.

    The `augmentation_funcs` parameter allows you to specify a list of 
    callable functions that apply various transformations to the input 
    images. These functions can include operations like flipping, rotating,
    blurring, and more.

    """
    def __init__(self, augmentation_funcs):
        """
        Initialize an ImageAugmenter.

        Parameters
        ----------
        augmentation_funcs : list of callable
            A list of functions that apply transformations to images.

        Returns
        -------
        ImageAugmenter
            Returns an instance of the ImageAugmenter.

        """
        self.augmentation_funcs = augmentation_funcs
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X):
        """
        Apply random transformations to input images.

        Parameters
        ----------
        X : ndarray, shape (height, width, channels)
            Input image(s) to which random augmentations will be applied.

        Returns
        -------
        augmented_images : ndarray, shape (height, width, channels)
            Augmented image(s) after applying random transformations.

        """
        for func in self.augmentation_funcs:
            X = func(images=X)
        return X


class ImageChannelSelector(BaseEstimator, TransformerMixin):
    """
    Select specific channels from images.

    Parameters
    ----------
    channels : list of int
        The indices of the channels to select.

    Examples
    --------
    # Selects the first two channels.
    >>> selector = ImageChannelSelector(channels=[0, 1])  
    >>> image = np.random.rand(256, 256, 3)
    >>> selected_channels = selector.transform(image)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable 
        parameters and does nothing during fitting.

    transform(X)
        Select specific channels from the input images.

    Notes
    -----
    ImageChannelSelector is a transformer that allows you to select specific
    color channels from input images. In many computer vision tasks, you may 
    only be interested in certain color channels of an image 
    (e.g., grayscale images, or selecting the red and green channels for analysis).

    The `channels` parameter allows you to specify which channels to select 
    from the input images. You can provide a list of channel indices to be
    retained.

    """
    def __init__(self, channels):
        """
        Initialize an ImageChannelSelector.

        Parameters
        ----------
        channels : list of int
            The indices of the channels to select.

        Returns
        -------
        ImageChannelSelector
            Returns an instance of the ImageChannelSelector.

        """
        self.channels = channels
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X):
        """
        Select specific channels from the input images.

        Parameters
        ----------
        X : ndarray, shape (height, width, channels)
            Input image(s) from which channels will be selected.

        Returns
        -------
        selected_channels : ndarray, shape (height, width, n_selected_channels)
            Input image(s) with specific channels selected.

        """
        return X[:, :, self.channels]

class ImageFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    A transformer for extracting features from images using a specified 
    pre-trained model. It supports preprocessing input images, resizing,
    and applying global average pooling to the output features.

    Parameters
    ----------
    model : keras.Model, optional
        A pre-trained model from `tensorflow.keras.applications` or custom 
        model for feature extraction. If `None`, VGG16 model with 'imagenet' 
        weights is used by default.
        
    preprocess_input : callable, optional
        A function to preprocess input images before feeding them into the 
        model. It should take an image array as input and return a similarly 
        shaped array. This is model specific; for example, use 
        `tensorflow.keras.applications.vgg16.preprocess_input` for VGG16.
        
    apply_global_pooling : bool, default False
        If `True`, applies global average pooling to the output of the model, 
        converting the output from a 4D tensor to a 2D tensor with shape 
        `(n_samples, n_features)`. Useful for flattening the features.
        
    target_size : tuple of int, optional
        The target size for resizing input images before feature extraction, 
        specified as `(height, width)`. If not specified, images are not 
        resized. This is necessary if the model expects a specific input size.
        
    weights : str, default 'imagenet'
        The weights to be loaded into the model. Only applicable if `model` 
        is `None`. Common options are 'imagenet' or None (random initialization).
        
    include_top : bool, default False
        Whether to include the fully-connected layer at the top of the model.
        Only applicable if `model` is `None`. Typically `False` for feature 
        extraction.

    Notes
    -----
    This transformer is designed for extracting high-level features from 
    images, which can be used in various machine learning pipelines, such as 
    classification, clustering, or as input to other models. It is especially 
    useful in transfer learning scenarios where pre-trained models are 
    leveraged to benefit from knowledge gained on large benchmark datasets.

    Examples
    --------
    >>> from tensorflow.keras.applications import VGG16
    >>> from tensorflow.keras.applications.vgg16 import preprocess_input 
    >>> from tensorflow.keras.models import Model
    >>> from gofast.transformers import ImageFeatureExtractor
    >>> base_model = VGG16(weights='imagenet', include_top=False)
    >>> model = Model(inputs=base_model.input, 
    ...               outputs=base_model.get_layer('block3_pool').output)
    >>> extractor = ImageFeatureExtractor(model=model, 
    ...                                    preprocess_input=preprocess_input,
    ...                                    apply_global_pooling=True, 
    ...                                    target_size=(224, 224))
    >>> image = np.random.rand(224, 224, 3)
    >>> features = extractor.transform(image)
    >>> features.shape
    (1, 256)

    The example above demonstrates creating an `ImageFeatureExtractor` with a 
    VGG16 model truncated at the 'block3_pool' layer. The input images are 
    preprocessed and resized to (224, 224), which is the expected input size 
    for VGG16, and global average pooling is applied to flatten the output 
    features.
    """
    def __init__(
        self, 
        model=None, 
        preprocess_input=None, 
        apply_global_pooling=False, 
        target_size=None, 
        weights='imagenet', 
        include_top=False
        ):
 
        self.model = model
        self.preprocess_input = preprocess_input
        self.apply_global_pooling = apply_global_pooling
        self.target_size = target_size
        self.weights = weights
        self.include_top = include_top
        
    def _ensure_model(self):
        """
        Ensures that a model is available for feature extraction. If no model 
        has been provided during instantiation, it initializes a default model 
        (VGG16) with specified weights and the option to include the top layers.
        This method also applies global average pooling to the model's output if 
        requested.
    
        This is a private method called internally by the class.
        """
        if self.model is None:
            from tensorflow.keras.models import Model
            from tensorflow.keras.applications import VGG16
            from tensorflow.keras.layers import GlobalAveragePooling2D

            base_model = VGG16(weights=self.weights, include_top=self.include_top)
            output = base_model.output
            if self.apply_global_pooling:
                output = GlobalAveragePooling2D()(output)# Apply Global Average Pooling
            self.model = Model(inputs=base_model.input, outputs=output)

    def fit(self, X, y=None):
        """
        Prepares the model for transforming images. This method checks and 
        initializes the model if it has not been provided or set up yet. Since 
        this transformer does not learn from the data, `X` and `y` are ignored, 
        but are included in the signature for compatibility with the scikit-learn 
        transformer API.
    
        Parameters
        ----------
        X : Ignored
            Not used, present for API consistency by convention.
        y : Ignored, optional
            Not used, present for API consistency by convention.
    
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        import_optional_dependency('tensorflow')
        self._ensure_model()
    
        return self
    
    def _preprocess(self, X):
        """
        Applies preprocessing steps to the input images. This includes resizing 
        the images to a target size (if specified) and applying a preprocessing 
        function (if provided). 
    
        This is a private method used to prepare images before they are passed 
        to the model for feature extraction.
    
        Parameters
        ----------
        X : ndarray
            Input image(s) as a NumPy array with shape (height, width, channels) 
            for a single image or (n_samples, height, width, channels) for 
            multiple images.
    
        Returns
        -------
        X : ndarray
            The preprocessed image(s) ready for feature extraction.
        """
        if self.target_size is not None:
            from tensorflow.image import resize
            X = resize(X, self.target_size)
        if self.preprocess_input is not None:
            X = self.preprocess_input(X)
        return X
    
    def transform(self, X):
        """
        Transforms the input images into a high-dimensional feature space using 
        the pre-trained model. This method applies any required preprocessing 
        steps to the input images, ensures they have the correct dimensions, 
        and then passes them through the model to extract features.
    
        Parameters
        ----------
        X : ndarray
            Input image(s) as a NumPy array. The array should have a shape of 
            (height, width, channels) for a single image or 
            (n_samples, height, width, channels) for multiple images.
    
        Returns
        -------
        features : ndarray
            The extracted features from the input image(s). If global pooling is 
            applied, the output will have shape (n_samples, n_features). Otherwise,
            the shape of the output depends on the model's architecture and the 
            presence of global pooling.
        """
        X = self._preprocess(X)
        if X.ndim == 3:
            X = np.expand_dims(X, axis=0)
        return self.model.predict(X)


class ImageEdgeDetector(BaseEstimator, TransformerMixin):
    """
    Detect edges in images using a specified method.

    Parameters
    ----------
    method : str, default='sobel'
        The method to use for edge detection. Options include 'sobel',
        'canny', and others.

    Examples
    --------
    >>> from skimage.filters import sobel
    >>> detector = ImageEdgeDetector(method='sobel')
    >>> image = np.random.rand(256, 256)
    >>> edges = detector.transform(image)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable 
        parameters and does nothing during fitting.

    transform(X)
        Detect edges in the input images using the specified edge detection 
        method.

    Notes
    -----
    Edge detection is a fundamental image processing technique used to identify 
    boundaries within images. It enhances the regions in an image where there 
    are significant changes in intensity or color, typically indicating
    object boundaries.

    This transformer allows you to perform edge detection using different
    methods such as Sobel and Canny.

    """
    
    def __init__(self, method='sobel'):
        """
        Initialize an ImageEdgeDetector.

        Parameters
        ----------
        method : str, default='sobel'
            The method to use for edge detection. Options include 'sobel',
            'canny', and others.

        Returns
        -------
        ImageEdgeDetector
            Returns an instance of the ImageEdgeDetector.

        """
        self.method = method
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        
        return self
    
    def transform(self, X):
        """
        Detect edges in the input images using the specified edge 
        detection method.

        Parameters
        ----------
        X : ndarray, shape (height, width)
            Input image(s) on which edge detection will be performed.

        Returns
        -------
        edges : ndarray, shape (height, width)
            Image(s) with edges detected using the specified method.

        Raises
        ------
        ValueError
            If an unsupported edge detection method is specified.

        Notes
        -----
        This method applies edge detection to the input image(s) using the 
        specified method, such as Sobel or Canny.

        If 'sobel' is selected as the method, the Sobel filter is applied 
        to detect edges.

        If 'canny' is selected as the method, the Canny edge detection 
        algorithm is applied.

        """
        import_optional_dependency ('skimage', extra = EMSG )
        from skimage.filters import sobel, canny
        
        if self.method == 'sobel':
            return sobel(X)
        elif self.method == 'canny':
            return canny(X)
        else:
            raise ValueError("Unsupported edge detection method.")



class ImageHistogramEqualizer(BaseEstimator, TransformerMixin):
    """
    Apply histogram equalization to images to improve contrast.

    Parameters
    ----------
    None

    Examples
    --------
    >>> from skimage.exposure import equalize_hist
    >>> equalizer = ImageHistogramEqualizer()
    >>> image = np.random.rand(256, 256)
    >>> equalized_image = equalizer.transform(image)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable 
        parameters and does nothing during fitting.

    transform(X)
        Apply histogram equalization to the input images to improve contrast.

    Notes
    -----
    Histogram equalization is a technique used to enhance the contrast of an 
    image by redistributing the intensity values of its pixels. It works by 
    transforming the intensity histogram of the image to achieve a more uniform
    distribution.

    This transformer applies histogram equalization to input images, making 
    dark areas darker and bright areas brighter, thus enhancing the 
    visibility of details.

    """
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X):
        """
        Apply histogram equalization to the input images to improve contrast.

        Parameters
        ----------
        X : ndarray, shape (height, width)
            Input image(s) to which histogram equalization will be applied.

        Returns
        -------
        X_equalized : ndarray, shape (height, width)
            Image(s) with histogram equalization applied for improved contrast.

        Notes
        -----
        This method applies histogram equalization to the input image(s). It
        enhances the contrast of the image by redistributing pixel intensity 
        values, making dark regions darker and bright regions brighter.

        """
        import_optional_dependency ('skimage', extra = EMSG )
        from skimage.exposure import equalize_hist
        return equalize_hist(X)


class ImagePCAColorAugmenter(BaseEstimator, TransformerMixin):
    """
    Apply PCA color augmentation as described in the AlexNet paper.

    Parameters
    ----------
    alpha_std : float
        Standard deviation of the normal distribution used for PCA noise.
        This parameter controls the strength of the color augmentation.
        Larger values result in more significant color changes.

    Examples
    --------
    >>> augmenter = ImagePCAColorAugmenter(alpha_std=0.1)
    >>> image = np.random.rand(256, 256, 3)
    >>> pca_augmented_image = augmenter.transform(image)

    Attributes
    ----------
    alpha_std : float
        The standard deviation used for PCA noise.

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable 
        parameters and does nothing during fitting.

    transform(X)
        Apply PCA color augmentation to the input images.

    Notes
    -----
    PCA color augmentation is a technique used to perform color variations on 
    images. It applies a PCA transformation to the color channels of the image 
    and adds random noise to create color diversity.

    The `alpha_std` parameter controls the strength of the color augmentation.
    Smaller values (e.g., 0.1) result in subtle color changes, while larger 
    values (e.g., 1.0) result in more dramatic color shifts.

    This transformer reshapes the input images to a flattened form for PCA 
    processing and then reshapes them back to their original shape after 
    augmentation.

    """
    def __init__(self, alpha_std):
        self.alpha_std = alpha_std
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X):
        """
        Apply PCA color augmentation to the input images.

        Parameters
        ----------
        X : ndarray, shape (height, width, channels)
            Input image(s) to which PCA color augmentation will be applied.

        Returns
        -------
        X_augmented : ndarray, shape (height, width, channels)
            Augmented image(s) with PCA color changes applied.

        Notes
        -----
        This method applies PCA color augmentation to the input image(s). 
        It reshapes the input image(s) to a flattened form, performs PCA 
        transformation on the color channels, adds random noise, and reshapes 
        the result back to the original shape.

        """
        orig_shape = X.shape
        X = X.reshape(-1, 3)
        X_centered = X - X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        X_pca = U.dot(np.diag(S)).dot(Vt)  # PCA transformation
        alpha = np.random.normal(0, self.alpha_std, size=3)
        X_augmented = X_pca + Vt.T.dot(alpha)
        return X_augmented.reshape(orig_shape)


class ImageBatchLoader(BaseEstimator, TransformerMixin):
    """
    A transformer that loads images in batches from a specified directory.
    This is particularly useful for processing large datasets that cannot
    fit entirely in memory. It supports optional preprocessing of images,
    resizing, and the use of a custom function for reading images.

    Parameters
    ----------
    batch_size : int
        The number of images to load in each batch. This parameter controls
        the size of the batches of images that are yielded each time the
        `transform` method is called.

    directory : str
        The path to the directory where the images are stored. The loader
        will attempt to load all images matching the specified `image_format`
        from this directory.

    preprocess_func : callable, optional
        A function to apply preprocessing to each image after loading but
        before yielding. This function should take an image array as input
        and return a processed image array. Typical preprocessing steps
        might include normalization, scaling, or color space conversion.

    target_size : tuple of int, optional
        A tuple specifying the target size of the images (height, width) to
        which the images will be resized. If `None`, images are not resized.
        This can be useful when a specific input size is required for model
        processing.

    image_format : str, default '.jpg'
        The file format of the images to load. Only files with this extension
        will be loaded from the directory. Default is '.jpg', but can be
        changed to accommodate different image formats (e.g., '.png').

    custom_reader : callable, optional
        A custom function for reading images from disk. It should take a file
        path as input and return an image array. If `None`, the default image
        reader `matplotlib.pyplot.imread` is used. This allows for the use of
        custom image reading logic, such as handling special image formats or
        performing initial preprocessing at the time of image loading.

    Notes
    -----
    This class is designed to be compatible with scikit-learn pipelines and
    transformers, following the fit/transform pattern. However, it does not
    learn from the data and thus `fit` method does not have any effect.

    Examples
    --------
    >>> from gofast.transformers import ImageBatchLoader
    >>> loader = ImageBatchLoader(batch_size=32, directory='/path/to/images',
                                  image_format='.png')
    >>> for batch in loader.transform(None):
    >>>     # Process the batch of images here
    >>>     print(batch.shape)

    In this example, `ImageBatchLoader` is used to load batches of 32 PNG
    images from '/path/to/images'. Each batch of images can then be processed
    as needed.
    """

    def __init__(self, batch_size, directory, preprocess_func=None, 
                 target_size=None, image_format='.jpg', custom_reader=None):
        self.batch_size = batch_size
        self.directory = directory
        self.preprocess_func = preprocess_func
        self.target_size = target_size
        self.image_format = image_format
        self.custom_reader = custom_reader if custom_reader else plt.imread
        
    def fit(self, X=None, y=None):
        """
        Fits the transformer to the data. This method does not perform any
        operation as the transformer does not learn from the data. It is
        implemented for compatibility with the scikit-learn transformer API.
    
        Parameters
        ----------
        X : None or Ignored
            Not used, present for API consistency by convention.
        y : None or Ignored, optional
            Not used, present for API consistency by convention.
    
        Returns
        -------
        self : object
            Returns the instance itself, unchanged.
    
        Notes
        -----
        Since this transformer is designed for loading and preprocessing images
        rather than learning from data, the `fit` method does not have any
        functionality. It exists to maintain compatibility with scikit-learn's
        transformer interface.
        """
        return self
    
    def transform(self, X=None):
        """
        Loads and yields batches of images from the specified directory. This
        method ignores its arguments, as the operation is determined by the
        directory, batch size, and other parameters specified at initialization.
    
        Parameters
        ----------
        X : None or Ignored
            Not used, present to maintain compatibility with the scikit-learn
            transformer interface.
    
        Yields
        ------
        batch_images : ndarray
            A batch of images loaded from the directory. The shape of the batch
            is determined by the `batch_size`, and the shape of each image is
            influenced by `target_size` (if specified) and the image's original
            dimensions. If `preprocess_func` is provided, it is applied to each
            image before yielding.
    
        Notes
        -----
        The method iterates over the specified directory, loading images in
        batches of the size determined by `batch_size`. Each image is optionally
        preprocessed and/or resized according to the `preprocess_func` and
        `target_size` parameters. This method is suitable for processing large
        directories of images in manageable batches, facilitating use cases like
        feeding batches of images into a machine learning model for training or
        inference.
    
        Examples
        --------
        >>> loader = ImageBatchLoader(batch_size=32, directory='/path/to/images')
        >>> for batch in loader.transform():
        >>>     process(batch)  # Example function to process each batch
    
        In this example, `transform` yields batches of 32 images from the directory
        '/path/to/images'. Each batch can then be processed as needed, for example,
        by feeding it into a neural network for image classification.
        """
        import_optional_dependency('skimage', extra=EMSG)
        from skimage.transform import resize
        image_files = [os.path.join(self.directory, fname) for fname in sorted(
            os.listdir(self.directory))
                       if fname.endswith(self.image_format)]
        for i in range(0, len(image_files), self.batch_size):
            batch_files = image_files[i:i + self.batch_size]
            batch_images = [self.custom_reader(file) for file in batch_files]
            if self.preprocess_func:
                batch_images = [self.preprocess_func(img) for img in batch_images]
            if self.target_size:
                batch_images = [resize(img, self.target_size, preserve_range=True, 
                                       anti_aliasing=True).astype(img.dtype) 
                                for img in batch_images]
            yield np.array(batch_images)


if __name__=='__main__': 
    import pytest
    from unittest.mock import patch
    
    # A fixture for creating a list of mock image file paths
    @pytest.fixture
    def mock_image_paths():
        return ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg']
    
    # A fixture for creating a mock preprocess function
    @pytest.fixture
    def mock_preprocess_func():
        def preprocess(image):
            # Pretend to process the image
            return image * 2  # Example operation
        return preprocess
    
    @patch('os.listdir', return_value=['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg'])
    @patch('matplotlib.pyplot.imread', return_value=np.zeros((100, 100, 3)))
    def test_image_batch_loader(mock_imread, mock_listdir, mock_image_paths, mock_preprocess_func):
        """
        Test to ensure the ImageBatchLoader yields batches of the specified size,
        correctly applies preprocessing, and works with mocked image paths.
        """
        # Initialize the ImageBatchLoader with a batch size of 2
        loader = ImageBatchLoader(batch_size=2, directory='mock/path/to/images', 
                                  preprocess_func=mock_preprocess_func, custom_reader=plt.imread)
    
        # Use the loader to transform (load) images and collect the batches
        batches = list(loader.transform())
    
        # There should be 2 batches given 4 mock images and a batch size of 2
        assert len(batches) == 2
    
        # Each batch should have 2 images
        for batch in batches:
            assert batch.shape == (2, 100, 100, 3)
    
        # The mock imread function should be called once for each image
        assert mock_imread.call_count == 4
    
        # The images should be processed by the preprocess_func (e.g., values doubled)
        # Checking this by confirming the array isn't all zeros (as returned by mock_imread)
        assert not np.all(batches[0] == 0)

    




















