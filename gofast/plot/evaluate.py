# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
:mod:`~gofast.plot.evaluate ` is a set of plot templates for visualising and 
inspecting the learning models.  It gives a quick depiction for users for 
models visualization and evaluation with : :class:`~gofast.plot.EvalPlotter`
"""
from __future__ import annotations 
import re
import warnings
import inspect 
import copy 
import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib as mpl 
import matplotlib.pyplot  as plt
import matplotlib.ticker as mticker
from abc import ABCMeta 
from scipy.cluster.hierarchy import dendrogram 
from matplotlib import cm 
from matplotlib.colors import BoundaryNorm

from sklearn.model_selection import learning_curve , train_test_split
from sklearn.metrics import mean_squared_error, silhouette_samples
from sklearn.metrics import  silhouette_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize
from sklearn.cluster import KMeans 
from sklearn.impute import   SimpleImputer

from .._gofastlog import gofastlog
from .._docstring import _core_docs, _baseplot_params, DocstringComponents
from .._typing import  _F, Optional, Tuple, List, ArrayLike, NDArray 
from .._typing import DataFrame,  Series 
from ..analysis.dimensionality import nPCA
from ..exceptions import NotFittedError, EstimatorError, PlotError
from ..metrics import precision_recall_tradeoff, roc_curve_, confusion_matrix_
from ..property import BasePlot 
from ..tools._dependency import import_optional_dependency 
from ..tools.funcutils import  to_numeric_dtypes, fancier_repr_formatter 
from ..tools.funcutils import  smart_strobj_recognition, reshape 
from ..tools.funcutils import  str2columns, make_ids, type_of_target, is_iterable 
from ..tools.mathex import linkage_matrix 
from ..tools.mlutils import categorize_target, projection_validator, export_target  
from ..tools.validator import  _check_consistency_size,  get_estimator_name  
from ..tools.validator import build_data_if,  check_array, check_X_y, check_y 
from ..tools.validator import _is_numeric_dtype , check_consistent_length
from ..tools.validator import assert_xy_in 

from .utils import _get_xticks_formatage,  make_mpl_properties

_logger=gofastlog.get_gofast_logger(__name__)

class MetricPlotter (BasePlot):
    def __init__(self, line_style='-',line_width=2, color_map='viridis',
                 **kws):
        """
        Initializes the PlotClass with custom plot styles.

        Parameters
        ----------
        line_style : str, optional
            The line style for the plots (default is '-').
        line_width : int, optional
            The line width for the plots (default is 2).
        color_map : str, optional
            The color map for the plots (default is 'viridis').
        """
        self.line_style = line_style
        self.line_width = line_width
        self.color_map = color_map
        super().__init__(**kws) 
     
    def fit(self, y_true=None, y_pred=None, *, data=None):
        """
        Fit the model with the true and predicted values for further 
        evaluation.
    
        This method is used to set the true and predicted values for a 
        model's output. It allows for further evaluation or comparison 
        between the predicted and actual results. The method accepts either 
        separate true and predicted values, or a combined data structure 
        from which these values will be extracted.
    
        Parameters
        ----------
        y_true : array-like, str, optional
            True values of the target variable. If 'data' is provided and 
            'y_true' is None, 'y_true' will be extracted from 'data'.
        y_pred : array-like, str, optional
            Predicted values from the model. If 'data' is provided and 
            'y_pred' is None, 'y_pred' will be extracted from 'data'.
        data : pandas.DataFrame, array-like, tuple, optional
            A combined data structure containing both true and predicted values. 
            If provided, 'y_true' and 'y_pred' are expected to be None or
            will be ignored.
    
        Raises
        ------
        ValueError
            If 'y_true' or 'y_pred' are not numeric arrays.
    
        Returns
        -------
        self : object
            Returns the instance itself.
    
        Examples
        --------
        >>> from gofast.plot import MetricPlotter
        >>> plotter = MetricPlotter()
        >>> y_true = [1, 2, 3, 4]
        >>> y_pred = [1.1, 2.1, 3.1, 4.1]
        >>> plotter.fit(y_true, y_pred)
        # Alternatively, using a combined data structure
        >>> data = ([1, 2, 3, 4], [1.1, 2.1, 3.1, 4.1])
        >>> plotter.fit(data=data)
        
        Notes
        -----
        This method is primarily used in the context of model evaluation, 
        where the true and predicted values are compared to assess the 
        performance of a regression model. Ensure that 'y_true' and 'y_pred' 
        are numeric and have compatible shapes.
        """
        # get y values if data is passed. 
        if data is not None: 
            if isinstance ( data, (tuple, list)): 
                y_true, y_pred = data 
            elif ( 
                    isinstance ( data, np.ndarray) 
                    and data.shape[1]==2
                    ): 
                y_true, y_pred = np.hsplit(data, 2 )
                y_true = reshape (y_true) ; y_pred = reshape ( y_pred ) 
       
        y_true, y_pred = assert_xy_in(y_true, y_pred, data= data )
        if ( 
                not _is_numeric_dtype(y_true , to_array=True) 
                or not _is_numeric_dtype(y_pred, to_array=True)
                ): 
            raise ValueError("y_true and y_pred must be numeric arrays. Got"
                             f" {type(y_true).__name__!r} and"
                             f" {type(y_pred).__name__!r} respectively." )
        self.y_true = y_true 
        self.y_pred =y_pred
        
        return self 
    
    def plotConfusionMatrix(self, class_names):
        """
        Plot a confusion matrix to visualize the performance of a 
        classification model.
    
        This method creates and displays a confusion matrix, which is a useful tool 
        for understanding how well a classification model is performing and where it 
        might be making errors. Each cell in the matrix represents the counts of 
        predictions in each predicted versus actual target class.
    
        Parameters
        ----------
        class_names : list of str
            A list of class names corresponding to each unique target label. 
            The order of class names should align with the encoded class 
            indices used by the model. These names are used to label the 
            matrix axes for better readability.
    
        Notes
        -----
        - The confusion matrix is calculated using the true and predicted 
          values provided in the model evaluator.
        - The matrix is colored for easier interpretation, with a color bar 
          included for reference.
        - Cell values represent the count of instances for each predicted 
          versus true class combination.
        - The diagonal cells correspond to correct predictions, while 
          off-diagonal cells indicate misclassifications.
    
        Examples
        --------
        >>> from gofast.plot import MetricPlotter
        >>> evaluator = MetricPlotter().fit(y_true, y_pred)
        >>> class_names = ['Class A', 'Class B', 'Class C']
        >>> evaluator.plotConfusionMatrix(class_names)
        # This will display a confusion matrix plot with the specified 
        class names.
    
        Raises
        ------
        ValueError
            If 'class_names' does not match the number of unique classes in 
            'y_true'.
    
        Returns
        -------
        self : object
            Returns the instance itself after rendering the plot.
        """
        self.inspect 
        # Validate class_names length
        if len(class_names) != len(np.unique(self.y_true)):
            raise ValueError("Length of class_names does not match the number"
                             " of classes in y_true.")
    
        # Compute confusion matrix and plot it
        cm = confusion_matrix(self.y_true, self.y_pred)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=self.color_map)
        ax.figure.colorbar(im, ax=ax)
        # Set axis labels and titles
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names, yticklabels=class_names,
               title='Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')
        # Rotate x labels for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", 
                 rotation_mode="anchor")
        # Annotate cells with counts
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
    
        return self

    def plotRocCurve(self, y_scores=None):
        """
        Plot a Receiver Operating Characteristic (ROC) curve for the model.
    
        This method visualizes the performance of a binary classifier by 
        plotting the ROC curve. The ROC curve illustrates the diagnostic 
        ability of the classifier by plotting the True Positive Rate (TPR) 
        against the False Positive Rate (FPR) at various threshold settings. 
        The area under the curve (AUC) provides a single metric to summarize 
        the curve's overall performance.
    
        The method requires the model to be fitted using the `fit` method 
        beforehand, where `y_true` (true binary class labels) and `y_pred` 
        (predicted probabilities or decision function scores for the positive class)
        are provided.
    
        Parameters
        ----------
        y_scores : array-like, optional
            Target scores or probability estimates of the positive class. 
            These can be either probability estimates of the positive class, 
            confidence values, or non-thresholded measure of decisions. If None,
            the scores provided during the fitting process (self.y_pred) are 
            used.
    
        Notes
        -----
        - The method assumes binary classification. Ensure that the target 
          values are appropriately encoded for binary classification (0 and 1).
        - The ROC curve and AUC are commonly used for evaluating classifiers in 
          binary classification tasks.
    
        Examples
        --------
        >>> from gofast.plot import MetricPlotter
        >>> from sklearn.metrics import roc_curve, auc
        >>> metric_plotter = MetricPlotter()
        >>> y_true = [0, 1, 0, 1]
        >>> y_pred = [0.1, 0.4, 0.35, 0.8]
        >>> metric_plotter.fit(y_true, y_pred)
        >>> metric_plotter.plotRocCurve()
    
        Returns
        -------
        self : object
            Returns the instance itself after rendering the plot.
        """
        self.inspect 
        # Ensure y_scores is set, defaulting to self.y_pred
        y_scores = y_scores if y_scores is not None else self.y_pred
    
        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(self.y_true, y_scores)
        roc_auc = auc(fpr, tpr)
    
        # Plotting ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=self.line_width,
                 linestyle=self.line_style,
                 label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=self.line_width,
                 linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
        
        return self

    def plotPrecisionRecallCurve(self, y_scores=None):
        """
        Plot a precision-recall curve to evaluate the model's performance.
    
        This method visualizes the trade-off between precision and recall for
        different thresholds, which is a key aspect in understanding the 
        performance of a binary classifier. A high area under the curve 
        represents both high recall and high precision.
    
        Parameters
        ----------
        y_scores : array-like, optional
            Target scores or probability estimates of the positive class.
            If None, uses the scores provided during the fitting process.
    
        Notes
        -----
        - Assumes the model has been fitted with true labels (y_true) and
          predicted probabilities or decision function scores (y_pred).
        - Precision-recall curves are most appropriate for imbalanced datasets.
    
        Examples
        --------
        >>> from gofast.plot import MetricPlotter
        >>> metric_plotter = MetricPlotter()
        >>> y_true = [0, 1, 0, 1]
        >>> y_pred = [0.1, 0.4, 0.35, 0.8]
        >>> metric_plotter.fit(y_true, y_pred)
        >>> metric_plotter.plotPrecisionRecallCurve()
    
        Returns
        -------
        self : object
            Returns the instance itself after rendering the plot.
            
        """
        self.inspect 
        # Ensure y_scores is set, defaulting to self.y_pred
        y_scores = y_scores if y_scores is not None else self.y_pred
    
        # Compute precision and recall
        precision, recall, _ = precision_recall_curve(self.y_true, y_scores)
    
        # Plotting Precision-Recall curve
        plt.figure()
        plt.plot(recall, precision, color='blue', lw=self.line_width,
                 linestyle=self.line_style)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.show()
        
        return self

    def plotActualVSPredicted(self, title='Actual vs Predicted'):
        """
        Plot a scatter plot to compare actual and predicted values.
    
        This method visualizes the relationship between actual and predicted 
        values for a regression model, using a scatter plot. This plot is 
        helpful for assessing the accuracy of predictions. Points along the 
        diagonal line indicate perfect predictions.
    
        Parameters
        ----------
        title : str, optional
            The title of the plot. Defaults to 'Actual vs Predicted'.
    
        Notes
        -----
        - Assumes that the model has been fitted with actual values (y_true) 
          and predicted values (y_pred).
        - A diagonal line (y=x) is plotted as a reference for perfect predictions.
    
        Examples
        --------
        >>> from gofast.plot import MetricPlotter
        >>> metric_plotter = MetricPlotter()
        >>> y_true = [1, 2, 3, 4]
        >>> y_pred = [1.1, 2.1, 3.1, 4.1]
        >>> metric_plotter.fit(y_true, y_pred)
        >>> metric_plotter.plotActualVSPredicted()
    
        Returns
        -------
        self : object
            Returns the instance itself after rendering the plot.
        
        """
        self.inspect 
        plt.figure()
        plt.scatter(self.y_true, self.y_pred, edgecolor='k', alpha=0.7)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        # Plotting diagonal line for reference
        plt.plot([self.y_true.min(), self.y_true.max()],
                 [self.y_true.min(), self.y_true.max()], 'k--', lw=2)
        plt.show()

        return self 

    def plotPrecisionRecallPerClass(
        self,  
        n_classes,
        y_scores=None,
        title='Precision-Recall per Class'
        ):
        """
        Plot precision-recall curves for each class in multi-class classification.
    
        This method generates precision-recall curves for each class, enabling 
        the evaluation of classifier performance on a class-by-class basis in 
        multi-class settings. It is particularly useful for understanding 
        class-specific behavior in imbalanced datasets.
    
        Parameters
        ----------
        n_classes : int
            The number of classes in the classification task.
        y_scores : array-like, optional
            Target scores (probabilities or decision function) for each class. 
            If None, uses the scores provided during the fitting process.
        title : str, optional
            The title of the plot. Defaults to 'Precision-Recall per Class'.
    
        Notes
        -----
        - Assumes the model has been fitted with true labels (y_true) in one-hot 
          encoded format and predicted probabilities or scores (y_pred).
        - Each curve in the plot corresponds to a different class, allowing for 
          comparison of precision-recall trade-offs across classes.
    
        Examples
        --------
        >>> from gofast.plot import MetricPlotter
        >>> from sklearn.preprocessing import label_binarize
        >>> metric_plotter = MetricPlotter()
        >>> y_true = [0, 1, 2, 0, 1]
        >>> y_pred = np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1], ...])
        >>> n_classes = 3
        >>> metric_plotter.fit(y_true, y_pred)
        >>> metric_plotter.plotPrecisionRecallPerClass(n_classes)
    
        Returns
        -------
        self : object
            Returns the instance itself after rendering the plot.
        """
        self.inspect 
        y_scores = y_scores if y_scores is not None else self.y_pred
        y_true_bin = label_binarize(self.y_true, classes=range(n_classes))
    
        plt.figure()
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], 
                                                          y_scores[:, i])
            plt.plot(recall, precision, lw=2, label=f'Class {i}')
    
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc='best')
        plt.show()
    
        return self

    def plotCumulativeGain(
        self, 
        y_probas=None, 
        title='Cumulative Gains Curve'
        ):
        """
        Plot a cumulative gain curve for a binary classification model.
    
        The cumulative gain curve illustrates the effectiveness of the binary 
        classifier by plotting the percentage of positive instances captured 
        against the proportion of the total number of cases. This curve is 
        helpful in assessing how well the classifier ranks instances.
    
        Parameters
        ----------
        y_probas : array-like, optional
            Probability estimates of the positive class, typically as output 
            by a classifier. If None, uses the probabilities provided during 
            the fitting process.
        title : str, optional
            The title of the plot. Defaults to 'Cumulative Gains Curve'.
    
        Notes
        -----
        - Assumes the model has been fitted with true labels (y_true) and 
          predicted probabilities (y_pred).
        - This plot is particularly useful for evaluating the performance of 
          models in terms of how effectively they identify positive instances.
    
        Examples
        --------
        >>> from gofast.plot import MetricPlotter
        >>> metric_plotter = MetricPlotter()
        >>> y_true = [0, 1, 0, 1]
        >>> y_probas = [0.1, 0.9, 0.2, 0.8]
        >>> metric_plotter.fit(y_true, y_probas)
        >>> metric_plotter.plotCumulativeGain()
    
        Returns
        -------
        self : object
            Returns the instance itself after rendering the plot.
        """
        self.inspect 
        y_probas = y_probas if y_probas is not None else self.y_pred 
        import_optional_dependency('scikitplot')
        from scikitplot.metrics import plot_cumulative_gain
        
        plt.figure()
        plot_cumulative_gain(self.y_true, y_probas)
        plt.title(title)
        plt.show()
    
        return self

    def plotLiftCurve(self, y_probas=None, title='Lift Curve'):
        """
        Plot a lift curve for a binary classification model.
    
        The lift curve is a visual tool for assessing the performance of a binary 
        classifier. It shows how much more likely we are to receive positive responses 
        than if we contacted a random sample of customers. It is used to evaluate the 
        effectiveness of a classification model by comparing the proportion of positive 
        instances targeted by a certain proportion of cases to the proportion if 
        targeting was random.
    
        Parameters
        ----------
        y_probas : array-like, optional
            Probability estimates of the positive class. These should be the predicted 
            probabilities of the positive class from the model. If None, the method uses 
            the probabilities provided during the fitting process.
        title : str, optional
            The title of the plot. Defaults to 'Lift Curve'.
    
        Notes
        -----
        - Assumes the model has been fitted with true labels (y_true) and predicted 
          probabilities (y_pred).
        - The lift curve is particularly useful in marketing and sales to evaluate 
          the efficiency of a classification model in identifying positive instances 
          over random selection.
    
        Examples
        --------
        >>> from gofast.plot import MetricPlotter
        >>> metric_plotter = MetricPlotter()
        >>> y_true = [0, 1, 0, 1]
        >>> y_probas = [0.1, 0.9, 0.2, 0.8]
        >>> metric_plotter.fit(y_true, y_probas)
        >>> metric_plotter.plotLiftCurve()
    
        Returns
        -------
        self : object
            Returns the instance itself after rendering the plot.
        """
        self.inspect 
        y_probas = y_probas if y_probas is not None else self.y_pred
        import_optional_dependency('scikitplot')
        from scikitplot.metrics import  plot_lift_curve
        plt.figure()
        plot_lift_curve(self.y_true, y_probas)
        plt.title(title)
        plt.show()
    
        return self


    def plotSilhouette(
        self, 
        X, 
        cluster_labels, 
        n_clusters, 
        title='Silhouette Plot'
        ):
        """
        Plot a silhouette plot for the cluster labels of a dataset.
    
        A silhouette plot visually represents how close each point in one cluster 
        is to points in the neighboring clusters, thus providing a way to assess 
        the separation distance between the resulting clusters. It's a useful 
        tool to evaluate the quality of cluster assignments.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The feature matrix of the dataset.
        cluster_labels : array-like of shape (n_samples,)
            Cluster labels for each data point.
        n_clusters : int
            The number of clusters in the dataset.
        title : str, optional
            The title of the plot. Defaults to 'Silhouette Plot'.
    
        Notes
        -----
        - The silhouette coefficient for a sample is a measure of how similar that 
          sample is to samples in its own cluster compared to samples in other clusters.
        - The silhouette values range from -1 to +1, where a high value indicates 
          that the object is well matched to its own cluster and poorly matched to 
          neighboring clusters.
    
        Examples
        --------
        >>> from gofast.plot import MetricPlotter
        >>> from sklearn.cluster import KMeans
        >>> from sklearn.datasets import make_blobs
        >>> X, _ = make_blobs(n_samples=150, n_features=2, centers=3, random_state=42)
        >>> kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
        >>> cluster_labels = kmeans.labels_
        >>> metric_plotter = MetricPlotter()
        >>> metric_plotter.plotSilhouette(X, cluster_labels, 3)
    
        Returns
        -------
        None
            The method renders the silhouette plot but does not return any value.
        """
        silhouette_avg = silhouette_score(X, cluster_labels)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
        plt.figure()
        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            color = cm.nipy_spectral(float(i) / n_clusters)
            plt.fill_betweenx(np.arange(y_lower, y_upper), 0,
                              ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
            y_lower = y_upper + 10
    
        plt.title(title)
        plt.xlabel("Silhouette coefficient values")
        plt.ylabel("Cluster label")
    
        plt.axvline(x=silhouette_avg, color="red", linestyle="--")
        plt.yticks([])
        plt.show()
    
    @property 
    def inspect(self): 
        """ Inspect data and trigger plot after checking the data entry. 
        Raises `NotFittedError` if `ExPlot` is not fitted yet."""
        
        msg = ( "{expobj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if self.y_true is None: 
            raise NotFittedError(msg.format(expobj=self))
        return 1
    
    def __repr__(self):
        """ Pretty format for programmer guidance following the API... """
        return fancier_repr_formatter(self ) 
       
    def __getattr__(self, name):
        """
        Custom attribute accessor to provide informative error messages.

        This method is called if the attribute accessed is not found in the
        usual places (`__dict__` and the class tree). It checks for common 
        attribute patterns and raises informative errors if the attribute is 
        missing or if the object is not fitted yet.

        Parameters
        ----------
        name : str
            The name of the attribute being accessed.

        Raises
        ------
        NotFittedError
            If the attribute indicates a requirement for a prior fit method call.

        AttributeError
            If the attribute is not found, potentially suggesting a similar attribute.

        Returns
        -------
        Any
            The value of the attribute, if found through smart recognition.
        """
        if name.endswith('_'):
            # Special handling for attributes that are typically set after fitting
            if name not in self.__dict__:
                if name in ('data_', 'X_'):
                    raise NotFittedError(
                        f"Attribute '{name}' not found.Please fit the"
                        f" {self.__class__.__name__} object first.")

        # Attempt to find a similar attribute name for a more informative error
        similar_attr = self._find_similar_attribute(name)
        suggestion = f". Did you mean '{similar_attr}'?" if similar_attr else ""

        raise AttributeError(f"'{self.__class__.__name__}' object has "
                             f"no attribute '{name}'{suggestion}")

    def _find_similar_attribute(self, name):
        """
        Attempts to find a similar attribute name in the object's dictionary.

        Parameters
        ----------
        name : str
            The name of the attribute to find a similar match for.

        Returns
        -------
        str or None
            A similar attribute name if found, otherwise None.
        """
        # Implement the logic for finding a similar attribute name
        # For example, using a string comparison or a fuzzy search
        rv = smart_strobj_recognition(name, self.__dict__, deep =True)
        return rv 
    
        
MetricPlotter.__doc__="""\
A class to visualize the results of machine learning models.

This class provides methods to plot confusion matrices, ROC curves,
precision-recall curves, and learning curves.

Methods
-------
plot_confusion_matrix(y_true, y_pred, class_names)
    Plots a confusion matrix.

plot_roc_curve(y_true, y_scores)
    Plots a Receiver Operating Characteristic (ROC) curve.

plot_precision_recall_curve(y_true, y_scores)
    Plots a precision-recall curve.

plot_learning_curve(estimator, X, y, cv)
    Plots a learning curve.
    
Attributes
----------
line_style : str
    The line style for the plots.
line_width : int
    The line width for the plots.
color_map : str
    The color map for the plots.
    
Examples
--------
>>> from gofast,plot import MetricPlotter 
>>> plotter = MetricPlotter()
>>> import seaborn as sns
>>> iris = sns.load_dataset('iris')
>>> # Plotting examples
>>> plotter.plot_histogram(iris, 'sepal_length', 
                           title='Sepal Length Distribution')
>>> plotter.plot_box(iris, 'sepal_width', by='species', 
                     title='Sepal Width by Species')

>>> plotter.plot_heatmap(iris.corr(), title='Iris Feature Correlations')
>>> feature_names = ['feature1', 'feature2', 'feature3']
>>> importances = np.random.rand(3)
>>> plotter.plot_feature_importance(feature_names, importances)
>>> y_actual = np.random.rand(100)
>>> y_predicted = np.random.rand(100)
>>> plotter.plot_actual_vs_predicted(y_actual, y_predicted)
>>> # Assuming multi-class classification with 3 classes
>>> y_true = np.random.randint(0, 3, 100)
>>> y_scores = np.random.rand(100, 3)
>>> plotter.plot_precision_recall_per_class(
    y_true, y_scores, n_classes=3)

>>> # Example data for cumulative gain and lift curve
>>> y_true = np.array([0, 1, 1, 0])
>>> y_probas = np.array([[0.7, 0.3], [0.4, 0.6], [0.6, 0.4], [0.8, 0.2]])

>>> plotter.plot_cumulative_gain(y_true, y_probas)
>>> plotter.plot_lift_curve(y_true, y_probas)

>>> # Example data for silhouette plot
>>> from sklearn.cluster import KMeans
>>> from sklearn.datasets import make_blobs

>>> X, y = make_blobs(n_samples=300, n_features=2, centers=4, 
                      cluster_std=1.0, random_state=10)
>>> kmeans = KMeans(n_clusters=4, random_state=10).fit(X)
>>> cluster_labels = kmeans.labels_
>>> plotter.plot_silhouette(X, cluster_labels, n_clusters=4)
"""

class EvalPlotter(BasePlot): 
    def __init__(self, 
        tname:str =None, 
        encode_labels: bool=False,
        scale: str = None, 
        cv: int =None, 
        objective:str=None, 
        prefix: str=None, 
        label_values:List[int]=None, 
        litteral_classes: List[str]=None, 
        **kws 
        ): 
        self.tname=tname
        self.objective=objective
        self.scale=scale
        self.cv=cv
        self.prefix=prefix 
        self.encode_labels=encode_labels 
        self.litteral_classes=litteral_classes 
        self.label_values=label_values 
        self.rs =kws.pop('rs', '--')
        self.ps =kws.pop('ps', '-')
        self.rc =kws.pop('rc', (.6, .6, .6))
        self.pc =kws.pop('pc', 'k')
        self.yp_lc =kws.pop('yp_lc', 'k') 
        self.yp_marker= kws.pop('yp_marker', 'o')
        self.yp_marker_edgecolor = kws.pop('yp_markeredgecolor', 'r')
        self.yp_lw = kws.pop('yp_lw', 3.)
        self.yp_ls=kws.pop('yp_ls', '-')
        self.yp_marker_facecolor =kws.pop('yp_markerfacecolor', 'k')
        self.yp_marker_edgewidth= kws.pop('yp_markeredgewidth', 2.)
        
        super().__init__(**kws) 

    @property 
    def inspect(self): 
        """ Inspect data and trigger plot after checking the data entry. 
        Raises `NotFittedError` if `ExPlot` is not fitted yet."""
        
        msg = ( "{expobj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if self.X is None: 
            raise NotFittedError(msg.format(expobj=self))
        return 1 
     
    def save (self, fig): 
        """ savefigure if figure properties are given. """
        if self.savefig is not None: 
            fig.savefig (self.savefig,dpi = self.fig_dpi , 
                         bbox_inches = 'tight', 
                         orientation=self.fig_orientation 
                         )
        plt.show() if self.savefig is None else plt.close () 
        
    def fit(self, X, y=None, **fit_params):
        """
        Fit data and prepare the EvalPlotter instance for plotting.
    
        This method prepares the EvalPlotter instance with data for subsequent 
        plotting operations. It ensures that the data is in the correct format 
        and that only numerical features are retained for plotting. The method 
        handles preprocessing steps like converting data to a DataFrame and 
        managing target variables.
    
        Parameters
        ----------
        X : ndarray or DataFrame, shape (M, N)
            The training set, where `M` is the number of samples and `N` is the 
            number of features. `X` can be a matrix of feature vectors or a 
            vector of precomputed (dis)similarities. Non-matrix data requires a 
            feature extractor or a pairwise metric for transformation.
        y : array-like, shape (M,), optional
            The training target values, where `M` is the number of samples. `y` 
            represents the dependent variable in learning, typically used for 
            supervised tasks.
        fit_params : dict, optional
            Additional keyword arguments for data preprocessing. Supported keys:
            - 'columns': List of column names if `X` is an ndarray.
            - 'prefix': Prefix for encoding categorical target variables.
    
        Raises
        ------
        NotFittedError
            If methods requiring a fitted model are called before fitting.
        TypeError
            If the processed `X` does not contain any numeric data.
    
        Returns
        -------
        self : EvalPlotter
            The fitted EvalPlotter instance for method chaining.
    
        Examples
        --------
        >>> from gofast.plot import EvalPlotter
        >>> X, y = some_data_loader()
        >>> plotter = EvalPlotter()
        >>> plotter.fit(X, y)
    
        Notes
        -----
        - EvalPlotter is designed for plotting metrics and evaluations, 
          hence it requires numeric data.
        - Categorical data in `X` is removed during fitting.
    
        """
        columns = fit_params.pop('columns', None)
        prefix = fit_params.pop('prefix', None)
        
        # Ensuring data is in DataFrame format and managing target variables
        X = build_data_if(X, to_frame=True, force=True, input_name='X', 
                          columns=columns, raise_warning='silence')
        X, y = self._target_manager(X, y, prefix)
        
        # Retaining only numeric data for plotting
        self.X = to_numeric_dtypes(X, pop_cat_features=True)
        if len(X.columns) == 0: 
            raise TypeError(f"{self.__class__.__name__!r} expects numeric data frame only.")
        
        return self

    def plot2d(self, x_feature, y_feature, groups=None, xlabel=None, 
               ylabel=None, title=None):
        """
        Plot a two-dimensional graph of two features from the dataset.
    
        This method visualizes the relationship between two selected features 
        from the fitted dataset. It provides options for grouping and labeling 
        the data points, making it useful for exploratory data analysis.
    
        Parameters
        ----------
        x_feature : str or int
            The name or index of the feature to be plotted on the x-axis. 
            Specifies the horizontal dimension of the plot.
        y_feature : str or int
            The name or index of the feature to be plotted on the y-axis. 
            Specifies the vertical dimension of the plot.
        groups : Series or array-like, optional
            Group labels for the data points, used for coloring. 
            If provided, the plot will display data points in different 
            colors based on their group labels. Default is None.
        xlabel : str, optional
            Label for the x-axis. If None, uses `x_feature` as the label.
        ylabel : str, optional
            Label for the y-axis. If None, uses `y_feature` as the label.
        title : str, optional
            Title of the plot. If None, defaults to 'x_feature vs y_feature'.
    
        Notes
        -----
        - Assumes the EvalPlotter instance is already fitted with the dataset (`X`).
        - The method checks if `X` is a DataFrame or ndarray and retrieves the 
          specified features accordingly.
        - This method is particularly useful for visualizing the correlation or 
          patterns between two variables.
    
        Returns
        -------
        self : EvalPlotter
            Returns the instance itself after rendering the plot.
    
        Examples
        --------
        >>> from gofast.plot import EvalPlotter
        >>> plotter = EvalPlotter()
        >>> X = ... # Load or create a DataFrame or ndarray
        >>> plotter.fit(X)
        >>> plotter.plot2d('feature1', 'feature2', groups='class_label',
                           xlabel='Feature 1', ylabel='Feature 2', 
                           title='Feature 1 vs Feature 2')
    
        """
        self.inspect  # Check if EvalPlotter is fitted

        # Extract feature values for plotting
        if isinstance(self.X, pd.DataFrame):
            x_values = self.X[x_feature]
            y_values = self.X[y_feature]
        else:  # ndarray
            x_values = self.X[:, x_feature]
            y_values = self.X[:, y_feature]
    
        plt.figure()
        scatter = plt.scatter(x_values, y_values, c=groups,
                              cmap=self.color_map, edgecolor='k')
        
        # Adding legend for groups if provided
        if groups is not None:
            plt.legend(*scatter.legend_elements(), title="Groups")
    
        # Setting labels and title
        plt.xlabel(xlabel if xlabel else x_feature)
        plt.ylabel(ylabel if ylabel else y_feature)
        plt.title(title if title else f'{x_feature} vs {y_feature}')
        plt.grid(True)
        plt.show()
    
        return self

    def plotHistogram(self, feature, data=None, bins=30, xlabel=None,
                      ylabel='Frequency', title=None):
        """
        Plot a histogram for a specified feature in the dataset.
    
        This method creates a histogram to visualize the distribution of a 
        selected feature. It allows for customization of the number of bins, 
        axis labels, and plot title, making it a flexible tool for 
        exploratory data analysis.
    
        Parameters
        ----------
        feature : str or int
            The feature for which to plot the histogram. If 'data' is a DataFrame, 
            'feature' should be a column name; if 'data' is an ndarray, 'feature' 
            should be an index.
        data : DataFrame or ndarray, optional
            The dataset containing the feature. If None, uses the dataset provided 
            during the fitting process.
        bins : int, optional
            The number of bins for the histogram. More bins result in a finer 
            resolution. Default is 30.
        xlabel : str, optional
            Label for the x-axis. If None, defaults to the name or index of the feature.
        ylabel : str, optional
            Label for the y-axis. Default is 'Frequency'.
        title : str, optional
            Title of the histogram. If None, defaults to a generic title based on 
            the feature.
    
        Notes
        -----
        - Histograms are useful for getting a sense of the data distribution 
          of a single variable.
        - Assumes the EvalPlotter instance is already fitted with the dataset (`X`).
    
        Returns
        -------
        self : EvalPlotter
            Returns the instance itself after rendering the plot.
    
        Examples
        --------
        >>> from gofast.plot import EvalPlotter
        >>> plotter = EvalPlotter()
        >>> X = ... # Load or create a DataFrame or ndarray
        >>> plotter.fit(X)
        >>> plotter.plotHistogram('feature1', bins=20, xlabel='Feature 1 Value',
                                  ylabel='Count', title='Distribution of Feature 1')
    
        """
        self.inspect  # Check if EvalPlotter is fitted
    
        data = data if data is not None else self.X
        plt.figure()
    
        # Plotting histogram based on data type
        if isinstance(data, pd.DataFrame):
            plt.hist(data[feature], bins=bins, color='skyblue', edgecolor='black')
            xlabel = xlabel if xlabel else feature
        else:  # ndarray
            plt.hist(data[:, feature], bins=bins, color='skyblue', edgecolor='black')
            xlabel = xlabel if xlabel else f'Feature {feature}'
    
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title if title else f'Histogram of {xlabel}')
        plt.show()
    
        return self

    
    def plotBox(self, feature, data=None, 
                by=None, xlabel=None, ylabel=None, title=None):
        """
        Plot a box plot for a specified feature, optionally grouped by 
        another feature.
    
        This method creates a box plot to visualize the distribution of a selected 
        feature. When used with the 'by' parameter, it allows for comparison 
        across different groups, making it a valuable tool for examining 
        statistical measures like median, quartiles, and outliers.
    
        Parameters
        ----------
        feature : str
            The feature for which to plot the box plot. Represents the data 
            to be summarized and displayed in the box plot.
        data : DataFrame, optional
            The dataset containing the features. If None, uses the dataset 
            provided during the fitting process.
        by : str, optional
            A feature by which to group the data. If provided, creates separate 
            box plots for each group.
        xlabel : str, optional
            Label for the x-axis. If None, defaults to the grouping feature name.
        ylabel : str, optional
            Label for the y-axis. If None, defaults to the plotted feature name.
        title : str, optional
            Title of the box plot. If None, defaults to 'Box Plot of [feature]'.
    
        Notes
        -----
        - Box plots are useful for visualizing the spread and skewness of data, 
          as well as identifying potential outliers.
        - Assumes the EvalPlotter instance is already fitted with the dataset (`X`).
    
        Returns
        -------
        self : EvalPlotter
            Returns the instance itself after rendering the plot.
    
        Examples
        --------
        >>> from gofast.plot import EvalPlotter
        >>> plotter = EvalPlotter()
        >>> X = ... # Load or create a DataFrame
        >>> plotter.fit(X)
        >>> plotter.plotBox('feature1', by='group_feature',
                            xlabel='Group', ylabel='Feature Value',
                            title='Feature 1 Distribution by Group')
        """
        self.inspect  # Check if EvalPlotter is fitted
    
        data = data if data is not None else self.X
        plt.figure()
        data.boxplot(column=feature, by=by)
    
        # Setting labels and title
        plt.xlabel(xlabel if xlabel else (by if by else ''))
        plt.ylabel(ylabel if ylabel else feature)
        plt.title(title if title else f'Box Plot of {feature}')
        if by:
            plt.suptitle('')  # Remove default suptitle if 'by' is used
    
        plt.show()
        
        return self 
 
    def plotHeatmap(
        self,
        xlabel=None, 
        ylabel=None, 
        data=None, 
        title='Heatmap'
        ):
        """
        Plot a heatmap for visualizing correlations or 2-dimensional data.
    
        Heatmaps are effective for displaying the magnitude of values in a 2D matrix 
        format, using colors to represent different ranges of values. They are 
        commonly used for exploring correlations between variables or for visualizing 
        data matrices.
    
        Parameters
        ----------
        data : DataFrame or ndarray, optional
            The 2D dataset to be visualized in the heatmap. If None, uses the dataset 
            provided during the fitting process.
        xlabel : str, optional
            Label for the x-axis. Provides context about the columns in the heatmap.
        ylabel : str, optional
            Label for the y-axis. Provides context about the rows in the heatmap.
        title : str, optional
            Title of the heatmap. Default is 'Heatmap'.
    
        Notes
        -----
        - If 'data' is not specified, the method uses the dataset attached to the 
          EvalPlotter instance.
        - The method uses seaborn's heatmap function for visualization, providing 
          annotations and a color map for better readability.
    
        Returns
        -------
        self : EvalPlotter
            Returns the instance itself after rendering the plot.
    
        Examples
        --------
        >>> from gofast.plot import EvalPlotter
        >>> plotter = EvalPlotter()
        >>> X = ... # Load or create a DataFrame or ndarray
        >>> plotter.fit(X)
        >>> plotter.plotHeatmap(xlabel='Features', ylabel='Samples', 
                                title='Data Heatmap')
    
        """
        data = data if data is not None else self.X
        plt.figure()
        sns.heatmap(data, annot=True, fmt=".2f", cmap=self.color_map)
        plt.xlabel(xlabel if xlabel else '')
        plt.ylabel(ylabel if ylabel else '')
        plt.title(title)
        plt.show()
        
        return self 

    def plotFeatureImportance(
        self, 
        feature_names, 
        importances, 
        title='Feature Importances'
        ):
        """
        Plot a bar chart to visualize the importance scores of features.
    
        This method provides a visual representation of the importance or contribution 
        of each feature in a model. It is particularly useful for understanding the 
        impact of different features on model predictions.
    
        Parameters
        ----------
        feature_names : list
            Names of the features in the dataset. This list should align with 
            the 'importances' array, where each name corresponds to the 
            respective feature's importance score.
        importances : list or array
            Importance scores of the features. These scores indicate the relative 
            importance or contribution of each feature to the model's predictions.
        title : str, optional
            Title of the bar chart. Default is 'Feature Importances'.
    
        Notes
        -----
        - The method sorts features based on their importance scores in descending 
          order for a clearer visual comparison.
        - It is commonly used with tree-based models like Random Forest and Gradient 
          Boosting, where feature importance is a built-in attribute.
    
        Returns
        -------
        None
            The method renders the bar chart but does not return any value.
    
        Examples
        --------
        >>> from gofast.plot import EvalPlotter
        >>> plotter = EvalPlotter()
        >>> feature_names = ['feature1', 'feature2', 'feature3']
        >>> importances = [0.2, 0.5, 0.3]  # Example importance scores
        >>> plotter.plotFeatureImportance(feature_names, importances,
                                          title='Model Feature Importances')
    
        """
        # Sorting the features based on importance
        indices = np.argsort(importances)[::-1]
        sorted_names = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]
    
        plt.figure()
        plt.bar(range(len(importances)), sorted_importances, align='center')
        plt.xticks(range(len(importances)), sorted_names, rotation=45, ha='right')
        plt.title(title)
        plt.show()

    def _target_manager ( self, X, y , prefix =None, ): 
        """ Manage the target  and return X and y 
        
        Parameters 
        -----------
        X : pd.DataFrame 
           DataFrame that probably contain the flow 
           
        y: pd.Series 
           target to categorize. 
           
        prefix: str, Optional 
           The label to prefix the label values. 
           
        Return 
        --------
        X, y: DataFrame X and target y 
        
        """
        if self.tname: 
            if y is not None: y = pd.Series ( y, name = self.tname )
            else: y , X = export_target(X, tname= self.tname, drop=False )
                
        if y is None : 
            return X, y 
        
        check_consistent_length ( X, y )
        if not _is_numeric_dtype(y): 
            if not self.encode_labels : 
                warnings.warn("Non-numeric target is detected while" 
                              " `encode_labels` is not set. This behaviour"
                              " will raise an error in future.", FutureWarning) 
                self.litteral_classes = np.unique ( y )
            
        if self.encode_labels: 
            self.encode_y( y,  prefix =prefix )

        return X, self.y  
  
    def transform (self, X, **t_params): 
        """ Transform the data and imputs the numerical features. 
        
        It is not convenient to use `transform` if user want to keep 
        categorical values in the array 
        
        Parameters
        ------------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.
            
        t_params: dict, 
            Keyword arguments passed to :class:`sklearn.impute.SimpleImputer` 
            for imputing the missing data; default strategy is 'most_frequent'
            or keywords arguments passed to
            :func:gofast.tools.funcutils.to_numeric_dtypes`
            
        Return
        -------
        X: NDArray |Dataframe , shape (M x N )
            The transformed array or dataframe with numerical features 
            
        """
        self.inspect 
        X 

        strategy = t_params.pop('strategy', 'most_frequent')
        columns = list(X.columns )

        imp = SimpleImputer(strategy = strategy,  **t_params ) 
        # create new dataframe 
        X= imp.fit_transform(X )
        if self.scale: 
            if str(self.scale).find ('minmax') >=0 : 
                sc = MinMaxScaler() 
                
            else:sc =StandardScaler()
            
            X = sc.fit_transform(X)
            
        self.X = pd.DataFrame( X , columns = columns ) 
        
        return self.X  
    
    def fit_transform (self, X, y= None , **fit_params ): 
        """ Fit and transform at once. 
        
        Parameters
        ------------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.
            
        Return
        -------
        X: NDArray |Dataframe , shape (M x N )
            The transformed array or dataframe with numerical features 
        
        """
        self.X = self.fit(X, y, **fit_params).transform(self.X )
        
        return self.X 
   
    def encode_y ( self, y=None,  prefix:str =None ,values:List[int]=None, 
                      classes: List[str]=None,  objective:str =None ): 
        """ Encode y to hold the categorical values. 
        
        Note that if objective is set to 'flow', the `values` need to  be 
        supplied, otherwise an error will raises. 
        
        :param values: list of values to encoding the numerical target `y`. 
            for instance ``values=[0, 1, 2]`` 
        :param objective: str, relate to the flow rate prediction. Set to 
            ``None`` for any other predictions. 
        :param prefix: the prefix to add to the class labels. For instance, if 
            the `prefix` equals to ``FR``, class labels will become:: 
                
                [0, 1, 2] => [FR0, FR1, FR2]
                
        :classes: list of classes names to replace the default `FR` that is 
            used to specify the flow rate. For instance, it can be:: 
                
                [0, 1, 2] => [sf0, sf1, sf2]
        Returns 
        --------
        (self.y, classes): Array-like, list[int|str]
            Array of encoded labels and list of unique class label identifiers 
            
        """
        if ( y is None 
            and self.y is not None
            ): 
            y = copy.deepcopy ( self.y ) 
            
        if y is None: 
            raise TypeError (" Missing target y")
        y = pd.Series (y, name=self.tname or 'none')
        values = values or self.label_values 
        if values is not None: 
            y =  categorize_target(y , labels = values, 
                           rename_labels= classes or self.litteral_classes
                           )
        else:y = y.astype('category').cat.codes
            
        # add prefix and  update litteral classes and y 
        y = y.map(lambda o: prefix + str(o) ) if prefix else y 
        self.litteral_classes = np.unique (y) 
        self.y = y 
            
        return y,  self.litteral_classes  

    def plotRobustPCA(self,  n_components=None, n_axes=2, biplot=False,
            pc1_label='Axis 1',pc2_label='Axis 2', plot_dict=None,
            **pca_kws):
        """
        Plots PCA component analysis using sklearn's PCA implementation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        n_components : int or float, optional
            Number of dimensions to preserve. If a float between 0 and 1,
            it represents the ratio of variance to preserve. If None (default),
            95% of variance is preserved.

        n_axes : int, default=2
            Number of principal components to plot. Defaults to 2.

        biplot : bool, default=False
            If True, plots PCA feature importance (pc1 and pc2) and visualizes
            the level of variance and direction of components for different variables.

        pc1_label : str, default='Axis 1'
            Label for the first principal component axis.

        pc2_label : str, default='Axis 2'
            Label for the second principal component axis.

        plot_dict : dict, optional
            Dictionary of plot properties like colors and marker sizes.

        pca_kws : dict
            Additional keyword arguments passed to sklearn's PCA.

        Returns
        -------
        EvalPlotter
            The instance itself for method chaining.
        Examples 
        ---------
        >>> from gofast.plot import EvalPlotter 
        >>> from gofast.datasets import load_bagoue
        >>> X , y = load_bagoue(as_frame =True, return_X_y=True )
        >>> p =EvalPlotter(tname ='flow', scale = True, encode_labels=True )
        >>> _=p.fit_transform (X)
        >>> p.plotRobustPCA (n_components=2 ) 
        
        >>> p.plotRobustPCA(n_components=4. pc1_label='Axis2', pc2_label='Axis4')
        
        >>> p.plotRobustPCA (n_components=2, biplot=True)
        
        """
        self.inspect 
        # Setup and perform PCA analysis and return pca object.
        pca = nPCA(self.X, n_components=n_components, return_X=False, **pca_kws)
        X_pca = pca.X # pca.fit_transform(X)
        # n_axes = min(n_axes, X_pca.shape[1])
        # get number of axes 
        n_axes= X_pca.shape[1]
        # Extract labels for PCA axes
        pca_axes_labels = self._extract_pca_labels([pc1_label, pc2_label], n_axes)
        # Plotting
        D_COLORS = make_mpl_properties(1e3)
        plot_dict = plot_dict or  {'y_colors': D_COLORS,'s':100.}
        self._setup_plot(plot_dict)
        if biplot:
            self._plot_biplot(X_pca, pca, pca_axes_labels)
        else:
            self._plot_pca_components(X_pca, pca, pca_axes_labels, plot_dict)

        return self

    def _extract_pca_labels(self, labels, n_axes):
        """
        Extract numeric labels from PCA axis labels.

        Parameters
        ----------
        labels : list
            List of PCA axis labels.

        n_axes : int
            Number of PCA axes to consider.

        Returns
        -------
        list
            Numeric labels for PCA axes.
        """
        numeric_labels = []
        for label in labels:
            try:
                num = int(re.findall(r'\d+', label)[0])
                numeric_labels.append ( num -1 )
                # numeric_labels.append(min(num - 1, n_axes - 1))
            except IndexError:
                numeric_labels.append(0)
        if max( numeric_labels)>= n_axes: 
            msg =(f"Wrong number of axes. Expect {n_axes} components."
                  f" Got {max( numeric_labels) +1 }")
            raise ValueError(msg)
        return numeric_labels

    def _setup_plot(self, plot_dict):
        """
        Sets up plot properties from the provided dictionary.

        Parameters
        ----------
        plot_dict : dict
            Dictionary of plot properties.

        Returns
        -------
        None
        """
        # Default plot configurations
        default_dict = {'y_colors': ['b', 'g', 'r', 'c', 'm', 'y', 'k'], 's': 100}
        self.plot_config = {**default_dict, **(plot_dict or {})}

    def _plot_biplot(self, X_pca, pca, pca_axes_labels):
        """
        Plots a biplot for PCA analysis.

        Parameters
        ----------
        X_pca : array-like
            PCA transformed data.

        pca : PCA
            PCA object after fitting.

        pca_axes_labels : list
            List of numeric labels for PCA axes.

        Returns
        -------
        None
        """
        if self.y is None: 
            raise TypeError("Biplot expects the target y")
        axis1, axis2 = pca_axes_labels 
        # Additional implementation for biplot
        mpl.rcParams.update(mpl.rcParamsDefault) 
        # reset ggplot style
        components = np.concatenate(
            (pca.components_[axis1, :], pca.components_[axis2, :]))
        try: 
            plot_unified_pca( np.transpose(components), X_pca, y=self.y,
                classes=self.litteral_classes, colors=self.plot_config ['y_colors'] )
        except : 
            # plot defaults configurations if 
            # something wrong 
            plot_unified_pca(np.transpose(pca.components_[0:2, :]),X_pca[:,:2],
                             y=self.y, classes=self.litteral_classes, 
                             colors=self.plot_config ['y_colors'] 
                         )
 
        plt.show()

    def _plot_pca_components(self, X_pca, pca, pca_axes_labels, plot_dict ):
        """
        Plots the specified PCA components.

        Parameters
        ----------
        X_pca : array-like
            PCA transformed data.
        pca : PCA
            PCA object after fitting.
        pca_axes_labels : list
            List of numeric labels for PCA axes.
        plot_dict : dict
            Dictionary of plot properties.
        y : array-like, optional
            Target labels for coloring the data points.
        """
        pca_data = self._prepare_pca_data(X_pca, pca_axes_labels, self.y)
        feature_names, ratios = self._extract_feature_components(pca, pca_axes_labels)
        fig, ax = self._create_figure()
        self._scatter_plot(ax, pca_data, feature_names, plot_dict)
        self._style_plot(ax, pca_data, feature_names, ratios)

        plt.show()

    def _extract_feature_components(self, pca, pca_axes_labels):
        """
        Extracts feature names and their explained variance ratios.

        Parameters
        ----------
        pca : PCA
            PCA object after fitting.
        pca_axes_labels : list
            List of numeric labels for PCA axes.

        Returns
        -------
        tuple
            A tuple containing feature names and their explained variance ratios.
        """
        importances = pca.feature_importances_
        names = [importances[i][1][0] for i in pca_axes_labels]
        ratios = [round(abs(importances[i][2][0]) * 100, 2) for i in pca_axes_labels]

        return names, ratios

    def _prepare_pca_data(self, X_pca, pca_axes_labels, y):
        """
        Prepares the PCA data for plotting.

        Parameters
        ----------
        X_pca : array-like
            PCA transformed data.
        pca_axes_labels : list
            List of numeric labels for PCA axes.
        y : array-like, optional
            Target labels for coloring the data points.

        Returns
        -------
        DataFrame
            A DataFrame containing the PCA data and target labels.
        """
        pca_cols = [f'Axis {i + 1}' for i in pca_axes_labels]
        pca_data = pd.DataFrame(X_pca[:, pca_axes_labels], columns=pca_cols)

        if y is not None:
            pca_data[self.tname] = y

        return pca_data

    def _create_figure(self):
        """
        Creates a figure for the plot.

        Returns
        -------
        tuple
            A tuple containing the figure and axes objects.
        """
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.add_subplot(1, 1, 1)

        return fig, ax

    def _scatter_plot(self, ax, pca_data, feature_names, plot_dict):
        """
        Plots scatter points on the axes.

        Parameters
        ----------
        ax : Axes
            The axes object to plot on.
        pca_data : DataFrame
            DataFrame containing PCA data.
        feature_names : list
            List of feature names.
        plot_dict : dict
            Dictionary of plot properties.
        """
        # now update the feature components 
        # and replace to the exis in pca_data 
        pca_data.columns = ( 
            ( feature_names +[self.tname])  if self.y is not None 
            else feature_names ) 
        for target, color in zip(self.litteral_classes, self.plot_config['y_colors']):
            ax.scatter(pca_data.loc[pca_data[self.tname] == target, feature_names[0]],
                       pca_data.loc[pca_data[self.tname] == target, feature_names[1]],
                       color=color, s=plot_dict['s'])

    def _style_plot(self, ax, pca_data, feature_names, ratios):
        """
        Styles the plot with labels, lines, and grids.

        Parameters
        ----------
        ax : Axes
            The axes object to style.
        feature_names : list
            List of feature names.
        ratios : list
            List of explained variance ratios.
        """
        ax.set_xlabel(f'{feature_names[0]} ({ratios[0]}%)', fontsize=self.font_size * self.fs)
        ax.set_ylabel(f'{feature_names[1]} ({ratios[1]}%)', fontsize=self.font_size * self.fs)
        ax.set_title('PCA', fontsize=(self.font_size + 1) * self.fs)
        ax.grid(color=self.lc, linestyle=self.ls, linewidth=self.lw/10)

        # Add circle and lines.If components are the same weights
        # convert error as a warnings. 
        try: 
            max_lim = np.ceil(max(abs(pca_data[feature_names[0]]).max(),
                                  abs(pca_data[feature_names[1]]).max()))
        except BaseException as e: 
            raise ValueError(str(e) + " Please your PCA axes labels.")
 
        circle = plt.Circle((0, 0), max_lim, color='blue', fill=False)
        ax.add_artist(circle)
        ax.add_line(plt.Line2D((0, 0), (-max_lim, max_lim), 
                               color=self.lc, linewidth=self.lw, linestyle=self.ls))
        ax.add_line(plt.Line2D((-max_lim, max_lim), (0, 0),
                               color=self.lc, linewidth=self.lw, linestyle=self.ls))
        
    def plotBasePCA(self, labels=None, title='PCA Plot', figsize=(8, 6)):
        """
        Plots a 2D PCA of the given dataset.

        Parameters:
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to perform PCA on.

        labels : array-like of shape (n_samples,), optional
            Labels for each data point, used for coloring.

        title : str, optional
            Title of the plot.

        figsize : tuple, optional
            Size of the figure.

        Returns:
        -------
        None
        """
        self.inspect 
        # Standardizing the features
        if self.scale: 
            self.X  = StandardScaler().fit_transform(self.X)
        # Perform PCA
        pca = nPCA(self.X, n_components=2, return_X =False )
        principal_components = pca.X

        # Create a DataFrame for the PCA results
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

        # Plot initialization
        plt.figure(figsize=figsize)
        ax = plt.subplot(1, 1, 1)

        # Scatter plot for each label
        if labels is not None:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                indices = labels == label
                ax.scatter(pca_df.loc[indices, 'PC1'],
                           pca_df.loc[indices, 'PC2'], 
                           label=label, marker = self.marker, 
                           markeredgecolor= self.marker_edgecolor, 
                           makeredgewidth=self.marker_edgewidth, 
                           markerfacecolor=self.maker_facecolor, 
                           )
        else:
            ax.scatter(pca_df['PC1'], pca_df['PC2'])

        # Add bold lines at 0 for x-axis and y-axis
        ax.axhline(y=0, color='black', linewidth=self.lw, linestyle=self.ls)
        ax.axvline(x=0, color='black', linewidth=self.lw , linestyle=self.ls, )
        
        # Plot styling
        ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2f} variance)')
        ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2f} variance)')
        ax.set_title(title)
        ax.grid(True)
        if labels is not None:
            ax.legend()
        plt.show()

    def plotPR(self, clf, label=None, kind='threshold', 
               method=None, cvp_kws=None, **prt_kws)-> 'EvalPlotter':
        """ 
        Plots Precision-Recall (PR) or Precision-Recall tradeoff.

        This method computes scores based on the decision function or probability
        predictions of a classifier and plots either the precision-recall curve
        or precision-recall tradeoff.

        Parameters
        ----------
        clf : callable
            A classifier estimator that supports binary targets and implements 
            either `decision_function` or `predict_proba` methods.

        label : int or str, default=1 
            Specific class label for evaluating precision and recall tradeoff.
            The default is the positive class label {1}

        kind : str, optional
            Type of plot. 'threshold' for precision-recall tradeoff plot (default)
            or 'recall' for precision vs recall plot.

        method : str, optional
            Method to retrieve scores for each instance. Options are 'decision_function'
            or 'predict_proba'. Defaults to 'decision_function'.

        cvp_kws : dict, optional
            Additional keyword arguments for sklearn's `cross_val_predict`.

        prt_kws : dict
            Additional keyword arguments for `precision_recall_tradeoff`.

        Returns
        -------
        EvalPlotter
            The instance itself for method chaining.

        Raises
        ------
        ValueError
            If `kind` is not 'threshold' or 'recall'.
        
        TypeError
            If target 'y' is missing or labels are not encoded.

        Examples
        --------
        >>> from sklearn.linear_model import SGDClassifier
        >>> from gofast.datasets import load_bagoue
        >>> from gofast.tools import categorize_target
        >>> from gofast.plot.evaluate import EvalPlotter

        >>> X, y = load_bagoue(as_frame=True, return_X_y=True)
        >>> sgd_clf = SGDClassifier(random_state=42)
        >>> plotter = EvalPlotter(scale=True, encode_labels=True)
        >>> plotter.fit_transform(X, y)
        >>> ybin = categorize_target(plotter.y, labels=2)
        >>> plotter.y = ybin
        >>> plotter.plotPR(sgd_clf, label=1)
        """
        self.inspect 
        label= label or 1 # positive 
        if self.y is None or type_of_target(self.y )!='binary':  
            raise TypeError("Precision-recall requires encoded binary labels.")

        kind = kind.lower().strip()
        if kind not in ('threshold', 'recall'):
            raise ValueError(f"Invalid kind '{kind}'. Expected"
                             " 'threshold' or 'recall'.")
        # Retrieve precision-recall tradeoff data
        prtObj = precision_recall_tradeoff(clf, self.X, self.y, cv=self.cv, 
                       label=label, method=method, cvp_kws=cvp_kws, **prt_kws)

        # Plotting setup
        fig, ax = plt.subplots(figsize=self.fig_size)
        xlabel, ylabel = self._set_axis_labels(kind)

        # Plot based on the selected kind
        self._plot_curve(ax, prtObj, kind)

        # Styling and saving the plot
        self._style_pr_plot(ax, xlabel, ylabel, kind)
        self.save(fig)
        return self

    def _set_axis_labels(self, kind):
        """ Set the x-axis and y-axis labels based on the plot kind. """
        if kind == 'threshold':
            return 'Threshold', 'Score'
        elif kind == 'recall':
            return 'Recall', 'Precision'

    def _plot_curve(self, ax, prtObj, kind):
        """ Plot the curve based on the kind (threshold/recall). """
        if kind == 'threshold':
            ax.plot(prtObj.thresholds, prtObj.precisions[:-1], 'b-',
                    label='Precision', **self.plt_kws)
            ax.plot(prtObj.thresholds, prtObj.recalls[:-1], 'g-',
                    label='Recall', **self.plt_kws)
        elif kind == 'recall':
            ax.plot(prtObj.recalls[:-1], prtObj.precisions[:-1], 'r-',
                    label='Precision vs Recall', **self.plt_kws)

    def _style_pr_plot(self, ax, xlabel, ylabel, kind):
        """ Apply styles to the plot. """
        ax.set_xlabel(xlabel, fontsize=0.5 * self.font_size * self.fs)
        ax.set_ylabel(ylabel, fontsize=0.5 * self.font_size * self.fs)
        ax.tick_params(axis='both', labelsize=0.5 * self.font_size * self.fs)
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 1] if kind == 'recall' else None)
        ax.grid(self.show_grid, axis=self.gaxis, which=self.gwhich, 
                color=self.gc, linestyle=self.gls, linewidth=self.glw, 
                alpha=self.galpha)
        ax.legend(**self.leg_kws)


    def plotROC(self, clfs, label, method=None, cvp_kws=None, **roc_kws):
        """
        Plots Receiver Operating Characteristic (ROC) curves for classifiers.

        This method supports plotting ROC curves for multiple classifiers. If 
        multiple classifiers are provided, each should be a tuple of the form 
        (name, classifier, method).

        Parameters
        ----------
        clfs : callable or list of tuples
            Classifier or a list of tuples where each tuple contains:
            - name: Name of the classifier.
            - classifier: Classifier instance.
            - method: 'decision_function' or 'predict_proba'.
            
        label : int or str
            The class label to evaluate.

        method : str, optional
            Default method to get scores if not specified in classifier tuples.

        cvp_kws : dict, optional
            Additional arguments for cross-validation prediction.

        roc_kws : dict
            Additional arguments for ROC curve computation.

        Returns
        -------
        EvalPlotter
            The instance itself for method chaining.

        Examples
        --------
        # Single classifier
        >>> from sklearn.linear_model import SGDClassifier
        >>> plotter = EvalPlotter()
        >>> plotter.plotROC(SGDClassifier(random_state=42), label=1)

        # Multiple classifiers
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> classifiers = [
                ('SGD', SGDClassifier(), 'decision_function'),
                ('Forest', RandomForestClassifier(), 'predict_proba')
            ]
        >>> plotter.plotROC(classifiers, label=1)
        """
        self.inspect 
        # Prepare classifier tuples
        if not isinstance(clfs, list):
            clfs = [(clfs.__class__.__name__, clfs, method)]

        # Generate ROC curves for each classifier
        roc_curves = [self._generate_roc_curve(
            clf, label, meth, cvp_kws, roc_kws) for name, clf, meth in clfs]

        # Plotting
        fig, ax = self._setup_roc_plot()
        self._draw_roc_curves(ax, roc_curves, clfs)
        self._finalize_plot(ax, 'False Positive Rate', 'True Positive Rate')
        
        return self

    def _generate_roc_curve(self, clf, label, method, cvp_kws, roc_kws):
        """ Generates ROC curve data for a given classifier. """
        return roc_curve_(clf, self.X, self.y, cv=self.cv, label=label,
                          method=method, cvp_kws=cvp_kws, **roc_kws)

    def _setup_roc_plot(self):
        """ Sets up the ROC plot. """
        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.plot([0, 1], [0, 1], ls='--', color='k')
        return fig, ax

    def _draw_roc_curves(self, ax, roc_curves, clfs):
        """ Draws ROC curves on the axes. """
        for (name, _clf, _), roc_data in zip(clfs, roc_curves):
            ax.plot(roc_data.fpr, roc_data.tpr, label=f'{name} (AUC={roc_data.roc_auc_score:.4f})')

    def _finalize_plot(self, ax, xlabel, ylabel):
        """ Finalizes styling and layout of the plot. """
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel(xlabel, fontsize=0.5 * self.font_size * self.fs)
        ax.set_ylabel(ylabel, fontsize=0.5 * self.font_size * self.fs)
        ax.tick_params(axis='both', labelsize=0.5 * self.font_size * self.fs)
        ax.legend(loc='lower right')
        plt.show()

    def plotROC2(self, clfs, label, method=None, cvp_kws=None,
                 **roc_kws) -> 'EvalPlotter':
        """
        Plot Receiver Operating Characteristic (ROC) curves for multiple classifiers.
    
        This method allows the visualization of the ROC curves for one or more 
        classifiers. Each classifier's ROC curve is plotted to assess and compare 
        their performance in terms of the trade-off between the true positive rate 
        (sensitivity) and the false positive rate (1 - specificity).
    
        Parameters
        ----------
        clfs : list of tuples or a single classifier
            Classifiers to be evaluated. Each classifier is provided as a tuple: 
            (classifier name, classifier instance, scoring method). 
            For a single classifier, it can be directly passed without a tuple.
        label : int or str
            The class label to focus on for generating the ROC curve.
        method : str, optional
            The scoring method to obtain scores from classifiers. Typical options 
            include 'decision_function' or 'predict_proba'.
        cvp_kws : dict, optional
            Additional keyword arguments to be passed to cross_val_predict function.
        roc_kws : dict
            Additional arguments for the roc_curve function.
    
        Returns
        -------
        EvalPlotter
            The instance itself for method chaining.
    
        Examples
        --------
        >>> from gofast.plot.evaluate import EvalPlotter
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.linear_model import SGDClassifier
        >>> from sklearn.datasets import load_bagoue
    
        >>> X, y = load_bagoue(as_frame=True)
        >>> sgd_clf = SGDClassifier(random_state=42)
        >>> forest_clf = RandomForestClassifier(random_state=42)
        >>> plotter = EvalPlotter(scale=True, encode_labels=True)
        >>> plotter.fit_transform(X, y)
    
        # Plot ROC for multiple classifiers
        >>> classifiers = [
                ('SGD', sgd_clf, 'decision_function'),
                ('Random Forest', forest_clf, 'predict_proba')
            ]
        >>> plotter.plotROC2(clfs=classifiers, label=1)
    
        """
        self.inspect 
        # Prepare classifiers
        clfs = self._prepare_classifiers(clfs, method)
    
        # Plotting
        fig, ax = self._initiate_plot()
        for clf_name, clf, clf_method in clfs:
            roc_obj = self._compute_roc(clf, clf_method, label, cvp_kws, **roc_kws)
            self._draw_roc_curve(ax, roc_obj, clf_name)
    
        self._finalize_plot2(ax, 'False Positive Rate', 'True Positive Rate', 
                            grid=True, legend_loc='lower right')
        return self
    
    def _prepare_classifiers(self, clfs, default_method):
        # Converts single classifier to a list of tuples format
        # and ensures all classifiers are in the correct format
        if not isinstance(clfs, list):
            clfs = [(getattr(clfs, '__name__', clfs.__class__.__name__),
                     clfs, default_method)]
        return [(name or clf.__class__.__name__, clf, method or default_method)
                for name, clf, method in clfs]
    
    def _initiate_plot(self):
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.add_subplot(1, 1, 1)
        return fig, ax
    
    def _compute_roc(self, clf, method, label, cvp_kws, **roc_kws):
        # Computes ROC curve for a given classifier
        roc_obj = roc_curve_(clf, self.X, self.y, cv=self.cv, label=label,
                             method=method, cvp_kws=cvp_kws, **roc_kws)
        return roc_obj
    
    def _draw_roc_curve(self, ax, roc_obj, clf_name):
        # Draws the ROC curve on the provided axes
        ax.plot(roc_obj.fpr, roc_obj.tpr,
                label=f'{clf_name} (AUC={roc_obj.roc_auc_score:.4f})', 
                linewidth=self.lw, linestyle=self.ls)
    
    def _finalize_plot2(self, ax, xlabel, ylabel, grid=False, legend_loc=None):
        # Finalize the plot by setting labels, grid, and legend
        ax.set_xlabel(xlabel, fontsize=self.font_size)
        ax.set_ylabel(ylabel, fontsize=self.font_size)
        ax.legend(loc=legend_loc)
        if grid:
            ax.grid(True)
        plt.show()

    
    def plotLearningCurve(self, model, *,  cv=4):
        """
        Plots a learning curve.

        Parameters
        ----------
        estimator : object
            An estimator instance implementing 'fit' and 'predict'.
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target relative to X for classification or regression.
        cv : int, cross-validation generator or an iterable
            Determines the cross-validation splitting strategy.
        """
        self.inspect 
        
        train_sizes, train_scores, test_scores = learning_curve(
            model, self.X, self.y, cv=cv)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", lw=self.line_width,
                 linestyle=self.line_style, label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", lw=self.line_width,
                 linestyle=self.line_style, label="Cross-validation score")

        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.title("Learning Curve")
        plt.legend(loc="best")
        plt.show()
        
    def plotConfusionMatrix(
        self, clf, *, 
        kind=None, labels=None,
        matshow_kws=None, 
        **conf_mx_kws):
        """
        Plots a confusion matrix for a classifier.

        This method visualizes the confusion matrix, either showing the count of 
        instances per class (map) or the error rates (error).

        Parameters
        ----------
        clf : callable
            Classifier estimator used to compute the confusion matrix. Must support
            binary or multiclass targets and have a 'fit' method.

        kind : str, optional
            Type of plot. 'map' to show the count of instances (default), or 'error'
            to show the error rates.

        labels : list of int, optional
            List of class labels to include in the confusion matrix. If None, all 
            classes in `y` are used.

        matshow_kws : dict, optional
            Additional keyword arguments for `matplotlib.pyplot.matshow`.

        conf_mx_kws : dict
            Additional keyword arguments for the confusion matrix computation.

        Returns
        -------
        EvalPlotter
            The instance itself for method chaining.

        Examples
        --------
        >>> from sklearn.svm import SVC
        >>> from gofast.plot.evaluate import EvalPlotter
        >>> X, y = fetch_data ('bagoue', return_X_y=True, as_frame =True)
        >>> # partition the target into 4 clusters-> just for demo 
        >>> plotter= EvalPlotter(scale =True, label_values = 4 ) 
        >>> plotter.fit_transform (X, y) 
        >>> # prepare our estimator 
        >>> svc_clf = SVC(C=100, gamma=1e-2, kernel='rbf', random_state =42)
        >>> matshow_kwargs ={
        ...        'aspect': 'auto', # 'auto'equal
        ...        'interpolation': None, 
        ...       'cmap':'jet }                   
        >>> plot_kws ={'lw':3, 
        ...       'lc':(.9, 0, .8), 
        ...       'font_size':15., 
        ...        'cb_format':None,
        ...        'xlabel': 'Predicted classes',
        ...        'ylabel': 'Actual classes',
        ...        'font_weight':None,
        ...        'tp_labelbottom':False,
        ...        'tp_labeltop':True,
        ...        'tp_bottom': False
        ...        }
        >>> plotter.plotConfusionMatrix(clf=svc_clf, matshow_kws = matshow_kwargs, 
        ...                          **plot_kws)
        >>> svc_clf = SVC(C=100, gamma=1e-2, kernel='rbf', random_state =42) 
        >>> # replace the integer identifier with litteral string 
        >>> plotter.litteral_classes = ['FR0', 'FR1', 'FR2', 'FR3']
        >>> plotter.plotConfusionMatrix(svc_clf, matshow_kws=matshow_kwargs, 
        ...                          kind='error', **plot_kws) 
        """
        self.inspect

        kind = kind.lower().strip() if kind else 'map'
        matshow_kws = matshow_kws or {'cmap': plt.cm.gray}

        labels = labels or self.label_values 
        y = self.y if labels is None else self.encode_y(values=labels)[0]
        labels = self.litteral_classes or labels

        # Compute confusion matrix
        confObj = confusion_matrix_(clf, self.X, y, cv=self.cv, **conf_mx_kws)

        # Plotting
        fig, ax = plt.subplots(figsize=self.fig_size)
        if kind == 'map':
            cax = ax.matshow(confObj.conf_mx, **matshow_kws)
            cb_label = 'Items Confused'
        elif kind == 'error':
            cax = ax.matshow(confObj.norm_conf_mx, **matshow_kws)
            cb_label = 'Error Rate'

        self._style_matshow_plot(ax, cax, labels, cb_label, fig)
        
        return self

    def _style_matshow_plot(self, ax, cax, labels, cb_label, fig):
        """ Styles the matshow plot with labels, colorbar, and ticks. """
        cbax = fig.colorbar(cax)
        ax.set_xlabel(self.xlabel or 'Predicted Classes', fontsize=self.font_size)
        ax.set_ylabel(self.ylabel or 'Actual Classes', fontsize=self.font_size)

        if labels:
            ax.set_xticklabels([''] + list(labels))
            ax.set_yticklabels([''] + list(labels))
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)

        cbax.ax.tick_params(labelsize=self.font_size)
        cbax.set_label(cb_label, size=self.font_size)

        self.save(fig)

    def __repr__(self):
        """ Pretty format for programmer guidance following the API... """
        return fancier_repr_formatter(self ) 
       
    def __getattr__(self, name):
        """
        Custom attribute accessor to provide informative error messages.

        This method is called if the attribute accessed is not found in the
        usual places (`__dict__` and the class tree). It checks for common 
        attribute patterns and raises informative errors if the attribute is 
        missing or if the object is not fitted yet.

        Parameters
        ----------
        name : str
            The name of the attribute being accessed.

        Raises
        ------
        NotFittedError
            If the attribute indicates a requirement for a prior fit method call.

        AttributeError
            If the attribute is not found, potentially suggesting a similar attribute.

        Returns
        -------
        Any
            The value of the attribute, if found through smart recognition.
        """
        if name.endswith('_'):
            # Special handling for attributes that are typically set after fitting
            if name not in self.__dict__:
                if name in ('data_', 'X_'):
                    raise NotFittedError(
                        f"Attribute '{name}' not found.Please fit the"
                        f" {self.__class__.__name__} object first.")

        # Attempt to find a similar attribute name for a more informative error
        similar_attr = self._find_similar_attribute(name)
        suggestion = f". Did you mean '{similar_attr}'?" if similar_attr else ""

        raise AttributeError(f"'{self.__class__.__name__}' object has "
                             f"no attribute '{name}'{suggestion}")

    def _find_similar_attribute(self, name):
        """
        Attempts to find a similar attribute name in the object's dictionary.

        Parameters
        ----------
        name : str
            The name of the attribute to find a similar match for.

        Returns
        -------
        str or None
            A similar attribute name if found, otherwise None.
        """
        # Implement the logic for finding a similar attribute name
        # For example, using a string comparison or a fuzzy search
        rv = smart_strobj_recognition(name, self.__dict__, deep =True)
        return rv 

#------------------------------------------------------------------------------
# Add specific params to Evaldocs 
_eval_params = dict( 
    objective="""
objective: str, default=None, 
    The purpose of dataset; what probem do we intend to solve ? 
    This parameter is mostly useful for geoscientists to handle their datasets.
    For instance, if the `objective` is set to ``flow``, `EvalPlotter` expects 
    the flow rate prediction purpose. In that case, some condition 
    of target values need to be fullfilled.  Moreover, if the objective 
    is set to ``flow``, `label_values`` as well as the `litteral_classes`
    parameters need to be supplied to right encode the target according 
    to the hydraulic system requirement during the campaign for drinking 
    water supply. For any other purpose for the dataset, keep the objective  
    to ``None``. Default is ``None``.    
    """, 
    yp_ls="""
yp_ls: str, default='-', 
    Line style of `Predicted` label. Can be [ '-' | '.' | ':' ] 
    """, 
    yp_lw="""
yp_lw: str, default= 3
    Line weight of the `Predicted` plot
    """,
    yp_lc ="""
yp_lc: str or :func:`matplotlib.cm`, default= 'k'
    Line color of the `Prediction` plot. *default* is ``k``
    """, 
    yp_marker="""
yp_marker: str or :func:`matplotlib.markers`, default ='o'
    Style of marker in  of `Prediction` points. 
    """, 
    yp_markerfacecolor="""
yp_markerfacecolor: str or :func:`matplotlib.cm`, default='k'
    Facecolor of the `Predicted` label marker.
    """, 
    yp_markeredgecolor="""
yp_markeredgecolor: stror :func:`matplotlib.cm`,  default= 'r' 
    Edgecolor of the `Predicted` label marker.
    """, 
    yp_markeredgewidth="""
yp_markeredgewidth: int, default=2
    Width of the `Predicted`label marker.
    """, 
    rs="""
rs: str, default='--'
    Line style of `Recall` metric 
    """, 
    ps="""
ps: str, default='-'
    Line style of `Precision `metric
    """, 
    rc="""
rc: str, default=(.6,.6,.6)
    Recall metric colors 
    """, 
    pc="""
pc: str or :func:`matplotlib.cm`, default='k'
    Precision colors from Matplotlib colormaps. 
    """
    )
_param_docs = DocstringComponents.from_nested_components(
    core=_core_docs["params"], 
    base=DocstringComponents(_baseplot_params), 
    evdoc=DocstringComponents(_eval_params), 
    )
#------------------------------------------------------------------------------

EvalPlotter.__doc__ ="""\
A Tool for Visualization of Metrics and Dimensionality Reduction 
in Model Evaluation. 

This class, inheriting from `BasePlot`, focuses on plotting for dimensional 
reduction and metrics analysis. It is specifically designed to work with 
numerical features.

Important Notes:
----------------
- The `EvalPlotter` class is optimized for use in classification problems and 
thus works best with supervised learning methods that involve discrete class 
labels. 

- Continuous target values for classification metrics are not recommended. 
  Users are strongly advised to preprocess their datasets to ensure optimal 
  compatibility and effectiveness of the `EvalPlotter` methods.

- For classification metrics, the target values should be discretized or 
  categorized into class labels before using the `fit` method. Users can 
  do this by either:
    1. Providing individual class labels as a list of integers through the 
       `EvalPlotter.encode_y` method.
    2. Specifying the number of desired clusters for the target labels. 

- While the latter option might be suitable for testing or academic purposes, 
  it is generally discouraged in practical applications. Automatically 
  partitioning targets into clusters may not reflect real-world data 
  distributions and can lead to misleading interpretations.

Example Usage:
--------------
Each method in `EvalPlotter` is accompanied by demonstrative examples, 
highlighting how to preprocess continuous labels for classification metrics.

Remember:
---------
In practical scenarios, especially with real datasets, it is crucial to 
use realistically categorized targets to avoid misinterpretations and ensure 
that the evaluation metrics align closely with the true nature of the problem 
at hand.    

Parameters 
-----------
{params.core.X}
{params.core.y}
{params.core.tname}
{params.evdoc.objective}
    
encode_labels: bool, default=False,  
    label encoding works with `label_values` parameter. 
    If the `y` is a continous numerical values, we could turn the 
    regression to classification by setting `encode_labels` to ``True``.
    if value is set to ``True`` and values of labels is not given, an 
    unique identifier is created which can not fit the exact needs of the 
    users. So it is recommended to set this parameters in combinaison with 
    the`label_values`.  For instance:: 
        
        encode_labels=True ; label_values =3 
        
    indicates that the target `y` values should be categorized to hold 
    the integer identifier equals to ``[0 , 1, 2]``. `y` are splitted into 
    three subsets where::
        
        classes (c) = [ c{{0}} <= y. min(), y.min() < c {{1}}< y.max(),
                         >=y.max {{2}}]
        
    This auto-splitting could not fit the exact classification of the 
    target so it is recommended to set the `label_values` as a list of 
    class labels. For instance `label_values=[0 , 1, 2]` and else. 
   
scale: str, ['StandardScaler'|'MinMaxScaler'], default ='StandardScaler'
   kind of feature scaling to apply on numerical features. Note that when 
   using PCA, it is recommended to turn `scale` to ``True`` and `fit_transform`
   rather than only fit the method. Note that `transform` method also handle 
   the missing nan value in the data where the default strategy for filling 
   is ``most_frequent``.
   
{params.core.cv}
    
prefix: str, optional 
    litteral string to prefix the integer identical labels. 
    
label_values: list of int, optional 
    works with `encode_labels` parameters. It indicates the different 
    class labels. Refer to explanation of `encode_labels`. 
    
Litteral_classes: list or str, optional 
    Works when objective is ``flow``. Replace class integer names by its 
    litteral strings. For instance:: 
        
            label_values =[0, 1, 3, 6]
            Litteral_classes = ['rate0', 'rate1', 'rate2', 'rate3']

{params.evdoc.yp_ls}
{params.evdoc.yp_lw}
{params.evdoc.yp_lc}
{params.evdoc.rs}
{params.evdoc.ps}
{params.evdoc.rc}
{params.evdoc.pc}
{params.evdoc.yp_marker}
{params.evdoc.yp_markerfacecolor}
{params.evdoc.yp_markeredgecolor}
{params.evdoc.yp_markeredgewidth}
{params.base.savefig}
{params.base.fig_dpi}
{params.base.fig_num}
{params.base.fig_size}
{params.base.fig_orientation}
{params.base.fig_title}
{params.base.fs}
{params.base.ls}
{params.base.lc}
{params.base.lw}
{params.base.alpha}
{params.base.font_weight}
{params.base.font_style}
{params.base.font_size}
{params.base.ms}
{params.base.marker}
{params.base.marker_facecolor}
{params.base.marker_edgecolor}
{params.base.marker_edgewidth}
{params.base.xminorticks}
{params.base.yminorticks}
{params.base.bins}
{params.base.xlim}
{params.base.ylim}
{params.base.xlabel}
{params.base.ylabel}
{params.base.rotate_xlabel}
{params.base.rotate_ylabel}
{params.base.leg_kws}
{params.base.plt_kws}
{params.base.glc}
{params.base.glw}
{params.base.galpha}
{params.base.gaxis}
{params.base.gwhich}
{params.base.tp_axis}
{params.base.tp_labelsize}
{params.base.tp_bottom}
{params.base.tp_labelbottom}
{params.base.tp_labeltop}
{params.base.cb_orientation}
{params.base.cb_aspect}
{params.base.cb_shrink}
{params.base.cb_pad}
{params.base.cb_anchor}
{params.base.cb_panchor}
{params.base.cb_label}
{params.base.cb_spacing}
{params.base.cb_drawedges} 

Notes 
--------
This module works with numerical data  i.e if the data must contains the 
numerical features only. If categorical values are included in the 
dataset, they should be  removed and the size of the data should be 
chunked during the fit methods. 

""".format(
    params=_param_docs,
)

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# create a shadow class to hold the font and matplotlib properties
# from 'EvalPlotter` and giving an option for saving figure
_b= EvalPlotter () 
pobj = type ('Plot', (BasePlot, ), {**_b.__dict__} ) 
setattr(pobj, 'save', _b.save )
# redefine the pobj doc 
pobj.__doc__="""\
Shadow plotting class that holds the :class:`~gofast.property.BasePlot`
parameters. 

Each matplotlib properties can be modified as  :class:`~gofast.view.pobj`
attributes object. For instance:: 
    
    >>> pobj.ls ='-.' # change the line style 
    >>> pobj.fig_Size = (7, 5) # change the figure size 
    >>> pobj.lw=7. # change the linewidth 
    
.. seealso:: 
    
    Refer to :class:`~gofast.property.BasePlot` for parameter details. 
    
"""
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def plot_loc_projection(
    X: DataFrame | NDArray, 
    Xt: DataFrame | NDArray =None, *, 
    columns: List[str] =None, 
    test_kws: dict =None,  
    **baseplot_kws 
    ): 
    """ Visualize train and test dataset based on 
    the geographical coordinates.
    
    Since there is geographical information(latitude/longitude or
    easting/northing), it is a good idea to create a scatterplot of 
    all instances to visualize data.
    
    Parameters 
    ---------
    X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
        training set; Denotes data that is observed at training and prediction 
        time, used as independent variables in learning. The notation 
        is uppercase to denote that it is ordinarily a matrix. When a matrix, 
        each sample may be represented by a feature vector, or a vector of 
        precomputed (dis)similarity with each training sample. 

    Xt: Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
        Shorthand for "test set"; data that is observed at testing and 
        prediction time, used as independent variables in learning. The 
        notation is uppercase to denote that it is ordinarily a matrix.
    columns: list of str or index, optional 
        columns is usefull when a dataframe is given  with a dimension size 
        greater than 2. If such data is passed to `X` or `Xt`, columns must
        hold the name to considered as 'easting', 'northing' when UTM 
        coordinates are given or 'latitude' , 'longitude' when latlon are 
        given. 
        If dimension size is greater than 2 and columns is None , an error 
        will raises to prevent the user to provide the index for 'y' and 'x' 
        coordinated retrieval. 
        
    test_kws: dict, 
        keywords arguments passed to :func:`matplotlib.plot.scatter` as test
        location font and colors properties. 
        
    baseplot_kws: dict, 
        All all  the keywords arguments passed to the peroperty  
        :class:`gofast.property.BasePlot` class. 
        
    Examples
    --------
    >>> from gofast.datasets import fetch_data 
    >>> from gofast.plot.evaluate  import plot_loc_projection 
    >>> # Discard all the non-numeric data 
    >>> # then inut numerical data 
    >>> from gofast.utils import to_numeric_dtypes, naive_imputer
    >>> X, Xt, *_ = fetch_data ('bagoue', split_X_y =True, as_frame =True) 
    >>> X =to_numeric_dtypes(X, pop_cat_features=True )
    >>> X= naive_imputer(X)
    >>> Xt = to_numeric_dtypes(Xt, pop_cat_features=True )
    >>> Xt= naive_imputer(Xt)
    >>> plot_kws = dict (fig_size=(8, 12),
                     lc='k',
                     marker='o',
                     lw =3.,
                     font_size=15.,
                     xlabel= 'easting (m) ',
                     ylabel='northing (m)' , 
                     markerfacecolor ='k', 
                     markeredgecolor='r',
                     alpha =1., 
                     markeredgewidth=2., 
                     show_grid =True,
                     galpha =0.2, 
                     glw=.5, 
                     rotate_xlabel =90.,
                     fs =3.,
                     s =None )
    >>> plot_loc_projection( X, Xt , columns= ['east', 'north'], 
                        trainlabel='train location', 
                        testlabel='test location', **plot_kws
                       )
    """
    
    trainlabel =baseplot_kws.pop ('trainlabel', None )
    testlabel =baseplot_kws.pop ('testlabel', None  )
    
    for k  in list(baseplot_kws.keys()): 
        setattr (pobj , k, baseplot_kws[k])
        
    #check array
    X=check_array (
        X, 
        input_name="X", 
        to_frame =True, 
        )
    Xt =check_array (
        Xt, 
        input_name="Xt", 
        to_frame =True, 
        )
    # validate the projections.
    xy , xynames = projection_validator(X, Xt, columns )
    x, y , xt, yt =xy 
    xname, yname, xtname, yname=xynames 

    pobj.xlim =[np.ceil(min(x)), np.floor(max(x))]
    pobj.ylim =[np.ceil(min(y)), np.floor(max(y))]   
    
    xpad = abs((x -x.mean()).min())/5.
    ypad = abs((y -y.mean()).min())/5.
 
    if  Xt is not None: 

        min_x, max_x = xt.min(), xt.max()
        min_y, max_y = yt.min(), yt.max()
        
        
        pobj.xlim = [min([pobj.xlim[0], np.floor(min_x)]),
                     max([pobj.xlim[1], np.ceil(max_x)])]
        pobj.ylim = [min([pobj.ylim[0], np.floor(min_y)]),
                     max([pobj.ylim[1], np.ceil(max_y)])]
      
    pobj.xlim =[pobj.xlim[0] - xpad, pobj.xlim[1] +xpad]
    pobj.ylim =[pobj.ylim[0] - ypad, pobj.ylim[1] +ypad]
    
     # create figure obj 
    fig = plt.figure(figsize = pobj.fig_size)
    ax = fig.add_subplot(1,1,1)
    
    xname = pobj.xlabel or xname 
    yname = pobj.ylabel or yname 
    
    if pobj.s is None: 
        pobj.s = pobj.fs *40 
    ax.scatter(x, y, 
               color = pobj.lc,
                s = pobj.s if not pobj.s else pobj.fs * pobj.s, 
                alpha = pobj.alpha , 
                marker = pobj.marker,
                edgecolors = pobj.marker_edgecolor,
                linewidths = pobj.lw,
                linestyles = pobj.ls,
                facecolors = pobj.marker_facecolor,
                label = trainlabel 
            )
    
    if  Xt is not None:
        if pobj.s is not None: 
            pobj.s /=2 
        test_kws = test_kws or dict (
            color = 'r',s = pobj.s, alpha = pobj.alpha , 
            marker = pobj.marker, edgecolors = 'r',
            linewidths = pobj.lw, linestyles = pobj.ls,
            facecolors = 'k'
            )
        ax.scatter(xt, yt, 
                    label = testlabel, 
                    **test_kws
                    )

    ax.set_xlim (pobj.xlim)
    ax.set_ylim (pobj.ylim)
    ax.set_xlabel( xname,
                  fontsize= pobj.font_size )
    ax.set_ylabel (yname,
                   fontsize= pobj.font_size )
    ax.tick_params(axis='both', 
                   labelsize= pobj.font_size )
    plt.xticks(rotation = pobj.rotate_xlabel)
    plt.yticks(rotation = pobj.rotate_ylabel)
    
    if pobj.show_grid is True : 
        ax.grid(pobj.show_grid,
                axis=pobj.gaxis,
                which = pobj.gwhich, 
                color = pobj.gc,
                linestyle=pobj.gls,
                linewidth=pobj.glw, 
                alpha = pobj.galpha
                )
        if pobj.gwhich =='minor': 
            ax.minorticks_on()
            
    if len(pobj.leg_kws) ==0 or 'loc' not in pobj.leg_kws.keys():
         pobj.leg_kws['loc']='upper left'
    ax.legend(**pobj.leg_kws)
    pobj.save(fig)

              
def plot_model(
    yt: ArrayLike |Series, 
    ypred:ArrayLike |Series=None,
    *, 
    clf:_F=None, 
    Xt:DataFrame|NDArray=None, 
    predict:bool =False, 
    prefix:Optional[bool]=None, 
    index:List[int|str] =None, 
    fill_between:bool=False, 
    labels:List[str]=None, 
    return_ypred:bool=False, 
    **baseplot_kws 
    ): 
    """ Plot model 'y' (true labels) versus 'ypred' (predicted) from test 
    data.
    
    Plot will allow to know where estimator/classifier fails to predict 
    correctly the target 
    
    Parameters
    ----------
    yt:array-like, shape (M, ) ``M=m-samples``,
        test target; Denotes data that may be observed at training time 
        as the dependent variable in learning, but which is unavailable 
        at prediction time, and is usually the target of prediction. 
        
    ypred:array-like, shape (M, ) ``M=m-samples``
        Array of the predicted labels. It has the same number of samples as 
        the test data 'Xt' 
        
    clf :callable, always as a function, classifier estimator
        A supervised predictor with a finite set of discrete possible 
        output values. A classifier must supports modeling some of binary, 
        targets. It must store a classes attribute after fitting.
        
    Xt: Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
        Shorthand for "test set"; data that is observed at testing and 
        prediction time, used as independent variables in learning. The 
        notation is uppercase to denote that it is ordinarily a matrix.
        
    prefix: str, optional 
        litteral string to prefix the samples/examples considered as 
        tick labels in the abscissa. For instance:: 
            
            index =[0, 2, 4, 7]
            prefix ='b' --> index =['b0', 'b2', 'b4', 'b7']

    predict: bool, default=False, 
        Expected to be 'True' when user want to predict the array 'ypred'
        and plot at the same time. Otherwise, can be set to 'False' and use 
        the'ypred' data already predicted. Note that, if 'True', an  
        estimator/classifier must be provided as well as the test data 'Xt', 
        otherwise an error will occur. 
        
    index: array_like, optional
        list integer values or string expected to be the index of 'Xt' 
        and 'yt' turned into pandas dataframe and series respectively. Note 
        that one of them has already and index and new index is given, the 
        latter must be consistent. This is usefull when data are provided as
        ndarray rathern than a dataframe. 
        
    fill_between: bool 
        Fill a line between the actual classes i.e the true labels. 
        
    labels: list of str or int, Optional
       list of labels names  to hold the name of each category.
       
    return_pred: bool, 
        return predicted 'ypred' if 'True' else nothing. 
    
    baseplot_kws: dict, 
        All all  the keywords arguments passed to the peroperty  
        :class:`gofast.property.BasePlot` class. 
 
    Examples
    --------
    (1)-> Prepare our data - Use analysis data of Bagoue dataset 
            since data is alread scaled and imputed
            
    >>> from gofast.exlib.sklearn  import SVC 
    >>> from gofast.datasets import fetch_data 
    >>> from gofast.plot import  plot_model 
    >>> from gofast.tools.mlutils import split_train_test_by_id
    >>> X, y = fetch_data('bagoue analysis' ) 
    >>> _, Xtest = split_train_test_by_id(X, 
                                          test_ratio=.3 ,  # 30% in test set 
                                          keep_colindex= False
                                        )
    >>> _, ytest = split_train_test_by_id(y, .3 , keep_colindex =False) 
    
   (2)-> prepared our demo estimator and plot model predicted 
   
    >>> svc_clf = SVC(C=100, gamma=1e-2, kernel='rbf', random_state =42) 
    >>> base_plot_params ={
                        'lw' :3.,                  # line width 
                        'lc':(.9, 0, .8), 
                        'ms':7.,                
                        'yp_marker' :'o', 
                        'fig_size':(12, 8),
                        'font_size':15.,
                        'xlabel': 'Test examples',
                        'ylabel':'Flow categories' ,
                        'marker':'o', 
                        'markeredgecolor':'k', 
                        'markerfacecolor':'b', 
                        'markeredgewidth':3, 
                        'yp_markerfacecolor' :'k', 
                        'yp_markeredgecolor':'r', 
                        'alpha' :1., 
                        'yp_markeredgewidth':2.,
                        'show_grid' :True,          
                        'galpha' :0.2,              
                        'glw':.5,                   
                        'rotate_xlabel' :90.,
                        'fs' :3.,                   
                        's' :20 ,                  
                        'rotate_xlabel':90
                   }
    >>> plot_model(yt= ytest ,
                   Xt=Xtest , 
                   predict =True , # predict the result (estimator fit)
                   clf=svc_clf ,  
                   fill_between= False, 
                   prefix ='b', 
                   labels=['FR0', 'FR1', 'FR2', 'FR3'], # replace 'y' labels. 
                   **base_plot_params 
                   )
    >>> # plot show where the model failed to predict the target 'yt'
    
    """
    def format_ticks (ind, tick_number):
        """ Format thick parameter with 'FuncFormatter(func)'
        rather than using:: 
            
        axi.xaxis.set_major_locator (plt.MaxNLocator(3))
        
        ax.xaxis.set_major_formatter (plt.FuncFormatter(format_thicks))
        """
        if ind % 7 ==0: 
            return '{}'.format (index[ind])
        else: None 
        
    #xxxxxxxxxxxxxxxx update base plot keyword arguments xxxxxxxxxxxxxx
    for k  in list(baseplot_kws.keys()): 
        setattr (pobj , k, baseplot_kws[k])

    # index is used for displaying the examples label in x-abscissa  
    # for instance index = ['b4, 'b5', 'b11',  ... ,'b425', 'b427', 'b430'] 
    
    Xt, yt,index, clf, ypred= _chk_predict_args (
        Xt, yt,index, clf, ypred , predict= predict 
        )
    if prefix is not None: 
        index =np.array([f'{prefix}' +str(item) for item in index ])        
        
    # create figure obj 
    fig = plt.figure(figsize = pobj.fig_size)
    ax = fig.add_subplot(1,1,1) # create figure obj 
    # control the size of predicted items 
    pobj.s = pobj.s or pobj.fs *30 
    # plot obverved data (test label =actual)
    ax.scatter(x= index,
               y =yt ,
                color = pobj.lc,
                s = pobj.s*10,
                alpha = pobj.alpha, 
                marker = pobj.marker,
                edgecolors = pobj.marker_edgecolor,
                linewidths = pobj.lw,
                linestyles = pobj.ls,
                facecolors = pobj.marker_facecolor,
                label = 'Observed'
                   )   
    # plot the predicted target
    ax.scatter(x= index, y =ypred ,
              color = pobj.yp_lc,
               s = pobj.s/2,
               alpha = pobj.alpha, 
               marker = pobj.yp_marker,
               edgecolors = pobj.yp_marker_edgecolor,
               linewidths = pobj.yp_lw,
               linestyles = pobj.yp_ls,
               facecolors = pobj.yp_marker_facecolor,
               label = 'Predicted'
               )
  
    if fill_between: 
        ax.plot(yt, 
                c=pobj.lc,
                ls=pobj.ls, 
                lw=pobj.lw, 
                alpha=pobj.alpha
                )
    if pobj.ylabel is None:
        pobj.ylabel ='Categories '
    if pobj.xlabel is None:
        pobj.xlabel = 'Test data'
        
    if labels is not None: 
        if not  is_iterable(labels): 
            labels =[labels]

        if len(labels) != len(np.unique(yt)): 
            warnings.warn(
                "Number of categories in 'yt' and labels must be consistent."
                f" Expected {len(np.unique(yt))}, got {len(labels)}")
        else:
            ax.set_yticks(np.unique(yt))
            ax.set_yticklabels(labels)
            
    ax.set_ylabel (pobj.ylabel,
                   fontsize= pobj.font_size  )
    ax.set_xlabel (pobj.xlabel,
           fontsize= pobj.font_size  )
   
    if pobj.tp_axis is None or pobj.tp_axis =='both': 
        ax.tick_params(axis=pobj.tp_axis, 
            labelsize= pobj.tp_labelsize *5 , 
            )
        
    elif pobj.tp_axis =='x':
        param_='y'
    elif pobj.tp_axis =='y': 
        param_='x'
        
    if pobj.tp_axis in ('x', 'y'):
        ax.tick_params(axis=pobj.tp_axis, 
                        labelsize= pobj.tp_labelsize *5 , 
                        )
        
        ax.tick_params(axis=param_, 
                labelsize= pobj.font_size, 
                )
    # show label every 14 samples 
    if len(yt ) >= 14 : 
        ax.xaxis.set_major_formatter (plt.FuncFormatter(format_ticks))

    plt.xticks(rotation = pobj.rotate_xlabel)
    plt.yticks(rotation = pobj.rotate_ylabel)
    
    if pobj.show_grid: 
        ax.grid(pobj.show_grid,
                axis=pobj.gaxis,
                which = pobj.gwhich, 
                color = pobj.gc,
                linestyle=pobj.gls,
                linewidth=pobj.glw, 
                alpha = pobj.galpha
                )
        if pobj.gwhich =='minor': 
            ax.minorticks_on()
            
    if len(pobj.leg_kws) ==0 or 'loc' not in pobj.leg_kws.keys():
         pobj.leg_kws['loc']='upper left'
    ax.legend(**pobj.leg_kws)
    
    pobj.save(fig)
    
    return ypred if return_ypred else None   

def plot_reg_scoring(
    reg, X, y, test_size=None, random_state =42, scoring ='mse',
    return_errors: bool=False, **baseplot_kws
    ): 
    #xxxxxxxxxxxxxxxx update base plot keyword arguments
    for k  in list(baseplot_kws.keys()): 
        setattr (pobj , k, baseplot_kws[k])
        
    scoring = scoring or 'mse'
    scoring = str(scoring).lower().strip() 
    if scoring not in ('mse', 'rme'): 
        raise ValueError ("Acceptable scorings are'mse' are 'rmse'"
                          f" got {scoring!r}")
    if not hasattr(reg, '__class__') and not inspect.isclass(reg.__class__): 
        raise TypeError(f"{reg!r} isn't a model estimator.")
         
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    
    train_errors, val_errors = [], []
    for m in range(1, len(y_train)): 
        try:
            reg.fit(X_train[:m], y_train[:m])
        except ValueError: # value_error 
            # raise ValueError (msg) from value_error 
            # skip the valueError
            # <The number of classes has to be greater 
            # than one; got 1 class>
            continue
        
        y_train_pred = reg.predict(X_train[:m])
        y_val_pred = reg.predict(X_val)
        if scoring in ('mse','rmse') :
            train_errors.append(mean_squared_error(
                y_train_pred, y_train[:m]))
            val_errors.append(
                mean_squared_error(y_val_pred, y_val))
        else:
            train_errors.append(sum(
                y_train_pred==y_train[:m])/len(y_train_pred))
            val_errors.append(
                sum(y_val_pred==y_val)/len(y_val_pred))
            
    # create figure obj 
     
    if scoring =='rmse': 
        train_errors= np.sqrt(train_errors)
        val_errors = np.sqrt(val_errors)
        
    if pobj.ylabel is None:
            pobj.ylabel ='Score'
            
    if pobj.xlabel is None: 
        pobj.xlabel = 'Training set size'
        
    fig = plt.figure(figsize = pobj.fig_size)
    ax = fig.add_subplot(1,1,1) # create figure obj 
    
    # set new attributes 
    for nv, vv in zip(('vlc', 'vls'), ('b', ':')): 
        if not hasattr(pobj, nv): 
            setattr(pobj, nv, vv)
        
    ax.plot(train_errors,
            color = pobj.lc, 
            linewidth = pobj.lw,
            linestyle = pobj.ls , 
            label = 'training set',
            **pobj.plt_kws )
    ax.plot(val_errors,
            color = pobj.vlc, 
            linewidth = pobj.lw,
            linestyle = pobj.vls , 
            label = 'validation set',
            **pobj.plt_kws )
    
    _remaining_plot_roperties(pobj, ax,  fig=fig )
    
    return (train_errors, val_errors) if return_errors else None 

plot_reg_scoring.__doc__ ="""\
Plot regressor learning curves using root-mean squared error scorings. 

Use the hold-out cross-validation technique for score evaluation [1]_. 

Parameters 
-----------
reg: callable, always as a function
    A regression estimator; Estimators must provide a fit method, and 
    should provide `set_params` and `get_params`, although these are usually 
    provided by inheritance from `base.BaseEstimator`. The estimated model 
    is stored in public and private attributes on the estimator instance, 
    facilitating decoding through prediction and transformation methods. 
    The core functionality of some estimators may also be available as 
    a ``function``.
     
{params.core.X}
{params.core.y}
scoring: str, ['mse'|'rmse'], default ='mse'
    kind of error to visualize on the regression learning curve. 
{params.core.test_size}
{params.core.random_state}

return_errors: bool, default='False'
    returns training eror and validation errors. 
    
baseplot_kws: dict, 
    All all  the keywords arguments passed to the peroperty  
    :class:`gofast.property.BasePlot` class. 
    
Returns 
--------
(train_errors, val_errors): Tuple, 
    training score and validation scores if `return_errors` is set to 
    ``True``, otherwise returns nothing   
    
Examples 
--------- 
>>> from gofast.datasets import fetch_data 
>>> from gofast.plot.evaluate  import plot_reg_scoring 
>>> # Note that for the demo, we import SVC rather than LinearSVR since the 
>>> # problem of Bagoue dataset is a classification rather than regression.
>>> # if use regression instead, a convergence problem will occurs. 
>>> from gofast.exlib.sklearn import SVC 
>>> X, y = fetch_data('bagoue analysed')# got the preprocessed and imputed data
>>> svm =SVC() 
>>> t_errors, v_errors =plot_reg_scoring(svm, X, y, return_errors=True)


Notes  
------
The hold-out technique is the classic and most popular approach for 
estimating the generalization performance of the machine learning. The 
dataset is splitted into training and test sets. The former is used for the 
model training whereas the latter is used for model performance evaluation. 
However in typical machine learning we are also interessed in tuning and 
comparing different parameter setting for futher improve the performance 
for the name refering to the given classification or regression problem for 
which we want the optimal values of tuning the hyperparameters. Thus, reusing 
the same datset over and over again during the model selection is not 
recommended since it will become a part of the training data and then the 
model will be more likely to overfit. From this issue, the hold-out cross 
validation is not a good learning practice. A better way to use the hold-out 
method is to separate the data into three parts such as the traing set, the 
the validation set and the test dataset. See more in [2]_. 

References 
------------
.. [1] Pedregosa, _F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
    Grisel, O., Blondel, M., et al. (2011) Scikit-learn: Machine learning in 
    Python. J. Mach. Learn. Res., 12, 2825–2830.
.. [2] Raschka, S. & Mirjalili, V. (2019) Python Machine Learning. 
    (J. Malysiak, S. Jain, J. Lovell, C. Nelson, S. D’silva & R. Atitkar, Eds.), 
    3rd ed., Packt.
""".format(params = _param_docs)    


def plot_model_scores(models, scores=None, cv_size=None, **baseplot_kws): 
    #xxxxxxxxxxxxxxxx set base plot keywords arguments
    for k  in list(baseplot_kws.keys()): 
        setattr (pobj , k, baseplot_kws[k])
        
    # if scores is None: 
    #     raise ValueError('NoneType can not be plot.')
    if isinstance(models, str): 
        models = str2columns (models)

    if not is_iterable(models): 
        models =[models]
        
    _ckeck_score = scores is not None 
    if _ckeck_score :
        scores = is_iterable(scores, exclude_string=True, transform= True )
        # if is_iterable(models) and is_iterable(scores): 
        if len(models) != len(scores): 
            raise TypeError(
                "Fined-tuned model and scores sizes must be consistent;"
                f" got {len(models)!r} and {len(scores)} respectively.")
            
    elif scores is None: 
        # check wether scores are appended to model
        try : 
            scores = [score for _, score in models]
        except: 
            raise TypeError (
                "Missing score(s). Scores are needed for each model.")
        models= [model for model, _  in models ]
    # for item assigments, use list instead. 
    models=[[bn, bscore] for bn, bscore in zip(models, scores)]

    for ii, (model, _) in enumerate(models) : 
        model = model or 'None'
        if not isinstance (model, str): 
            if inspect.isclass(model.__class__): 
                models[ii][0] = model.__class__.__name__
            else: 
                models[ii][0] = type(model).__name__
                
    # get_the minimal size from cv if not isinstance(cv, (int, float) ):
    cv_size_min = min (
        [ len(models[i][1]) for i in range (len(models))])

    if cv_size is None: 
        cv_size = cv_size_min

    if cv_size is not None: 
        try : 
            cv_size = int(cv_size)
        except: 
            raise ValueError(
                f"Expect a number for 'cv', got {type(cv_size).__name__!r}.")
            
        if cv_size < 1 : 
            raise ValueError (
                f"cv must contain at least one positivevalue, got {cv_size}")
        elif cv_size > cv_size_min : 
            raise ValueError(f"Size for cv is too large; expect {cv_size_min}"
                             f" as a maximum size, got {cv_size}")
        # shrink to the number of validation to keep the same size for all 
        # give model 
        models = [(modelname, modelval[:cv_size] ) 
                  for modelname, modelval in models]
    # customize plots with colors lines and styles 
    # and create figure obj 
    lcs_kws = {'lc': make_mpl_properties(cv_size), 
             'ls':make_mpl_properties(cv_size, 'line')
             }
    lcs_kws ['ls']= [pobj.ls] + lcs_kws['ls']
    lcs_kws ['lc']= [pobj.lc] + lcs_kws['lc']
    # create figure obj and change style
    # if sns_style is passed as base_plot_params 
    fig = plt.figure(figsize = pobj.fig_size)
    ax = fig.add_subplot(1,1,1) 
    if pobj.sns_style is not None: 
       sns.set_style(pobj.sns_style)
       
    for k in range(len(models)): 
        ax.plot(
            # np.array([i for i in range(cv_size)]) +1,
                np.arange (cv_size) +1, 
                models[k][1],
                color = lcs_kws['lc'][k], 
                linewidth = pobj.lw,
                linestyle = lcs_kws['ls'][k], 
                label = models[k][0],
                )
    # appendLineParams(pobj, ax, xlim=pobj.xlim, ylim=pobj.ylim)
    _remaining_plot_roperties(pobj, ax, xlim=pobj.xlim, 
                              ylim=pobj.ylim, fig=fig 
                       )
    pobj.save(fig)
    
plot_model_scores.__doc__="""\
uses the cross validation to get an estimation of model performance 
generalization.

It Visualizes model fined tuned scores vs the cross validation

Parameters 
----------
models: list of callables, always as a functions,   
    list of estimator names can also be  a pair estimators and validations 
    scores.For instance estimators and scores can be arranged as:: 
        
        models =[('SVM', scores_svm), ('LogRegress', scores_logregress), ...]
        
    If that arrangement is passed to `models` parameter then no need to pass 
    the score values of each estimators in `scores`. 
    Note that a model is an object which manages the estimation and 
    decoding. The model is estimated as a deterministic function of:

        * parameters provided in object construction or with set_params;
        * the global numpy.random random state if the estimator’s random_state 
            parameter is set to None; and
        * any data or sample properties passed to the most recent call to fit, 
            fit_transform or fit_predict, or data similarly passed in a sequence 
            of calls to partial_fit.
            
    list of estimators names or a pairs estimators and validations scores.
    For instance:: 
        
        clfs =[('SVM', scores_svm), ('LogRegress', scores_logregress), ...]
        
scores: array like 
    list of scores on different validation sets. If scores are given, 
    set only the name of the estimators passed to `models` like:: 
        
        models =['SVM', 'LogRegress', ...]
        scores=[scores_svm, scores_logregress, ...]

cv_size: float or int,
    The number of fold used for validation. If different models have different 
    cross validation values, the minimum size of cross validation is used and the 
    scored of each model is resized to match the minimum size number. 
    
baseplot_kws: dict, 
    All all  the keywords arguments passed to the peroperty  
    :class:`gofast.property.BasePlot` class.  
    
Examples 
---------
(1) -> Score is appended to the model 
>>> from gofast.exlib.sklearn import SVC 
>>> from gofast.plot.evaluate  import plot_model_scores
>>> import numpy as np 
>>> svc_model = SVC() 
>>> fake_scores = np.random.permutation (np.arange (0, 1,  .05))
>>> plot_model_scores([(svc_model, fake_scores )])
... 
(2) -> Use model and score separately 

>>> plot_model_scores([svc_model],scores =[fake_scores] )# 
>>> # customize plot by passing keywords properties 
>>> base_plot_params ={
                    'lw' :3.,                  
                    'lc':(.9, 0, .8), 
                    'ms':7.,                
                    'fig_size':(12, 8),
                    'font_size':15.,
                    'xlabel': 'samples',
                    'ylabel':'scores' ,
                    'marker':'o', 
                    'alpha' :1., 
                    'yp_markeredgewidth':2.,
                    'show_grid' :True,          
                    'galpha' :0.2,              
                    'glw':.5,                   
                    'rotate_xlabel' :90.,
                    'fs' :3.,                   
                    's' :20 ,
                    'sns_style': 'darkgrid', 
               }
>>> plot_model_scores([svc_model],scores =[fake_scores] , **base_plot_params ) 
"""
def plot_dendroheat(
    df: DataFrame |NDArray, 
    columns: List[str] =None, 
    labels:Optional[List[str]] =None,
    metric:str ='euclidean',  
    method:str ='complete', 
    kind:str = 'design', 
    cmap:str ='hot_r', 
    fig_size:Tuple[int] =(8, 8), 
    facecolor:str ='white', 
    **kwd
):
    """
    Attaches dendrogram to a heat map. 
    
    Hierachical dendrogram are often used in combination with a heat map which
    allows us to represent the individual value in data array or matrix 
    containing our training examples with a color code. 
    
    Parameters 
    ------------
    df: dataframe or NDArray of (n_samples, n_features) 
        dataframe of Ndarray. If array is given , must specify the column names
        to much the array shape 1 
    columns: list 
        list of labels to name each columns of arrays of (n_samples, n_features) 
        If dataframe is given, don't need to specify the columns. 
        
    kind: str, ['squareform'|'condense'|'design'], default is {'design'}
        kind of approach to summing up the linkage matrix. 
        Indeed, a condensed distance matrix is a flat array containing the 
        upper triangular of the distance matrix. This is the form that ``pdist`` 
        returns. Alternatively, a collection of :math:`m` observation vectors 
        in :math:`n` dimensions may be passed as  an :math:`m` by :math:`n` 
        array. All elements of the condensed distance matrix must be finite, 
        i.e., no NaNs or infs.
        Alternatively, we could used the ``squareform`` distance matrix to yield
        different distance values than expected. 
        the ``design`` approach uses the complete inpout example matrix  also 
        called 'design matrix' to lead correct linkage matrix similar to 
        `squareform` and `condense``. 
        
    metric : str or callable, default is {'euclidean'}
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`sklearn.metrics.pairwise.pairwise_distances`.
        If ``X`` is the distance array itself, use "precomputed" as the metric.
        Precomputed distance matrices must have 0 along the diagonal.
        
    method : str, optional, default is {'complete'}
        The linkage algorithm to use. See the ``Linkage Methods`` section below
        for full descriptions in :func:`gofast.tools.exmath.linkage_matrix`
        
    labels : ndarray, optional
        By default, ``labels`` is None so the index of the original observation
        is used to label the leaf nodes.  Otherwise, this is an :math:`n`-sized
        sequence, with ``n == Z.shape[0] + 1``. The ``labels[i]`` value is the
        text to put under the :math:`i` th leaf node only if it corresponds to
        an original observation and not a non-singleton cluster.
        
    cmap: str , default is {'hot_r'}
        matplotlib color map 
     
    fig_size: str , Tuple , default is {(8, 8)}
        the size of the figure 
    facecolor: str , default is {"white"}
        Matplotlib facecolor 
  
    kwd: dict 
        additional keywords arguments passes to 
        :func:`scipy.cluster.hierarchy.dendrogram` 
        
    Examples
    ---------
    >>> # (1) -> Use random data
    >>> import numpy as np 
    >>> from gofast.plot.evaluate  import plot_dendroheat
    >>> np.random.seed(123) 
    >>> variables =['X', 'Y', 'Z'] ; labels =['ID_0', 'ID_1', 'ID_2',
                                             'ID_3', 'ID_4']
    >>> X= np.random.random_sample ([5,3]) *10 
    >>> df =pd.DataFrame (X, columns =variables, index =labels)
    >>> plot_dendroheat (df)
    >>> # (2) -> Use Bagoue data 
    >>> from gofast.datasets import load_bagoue  
    >>> X, y = load_bagoue (as_frame=True )
    >>> X =X[['magnitude', 'power', 'sfi']].astype(float) # convert to float
    >>> plot_dendroheat (X )
    
    
    """
 
    df=check_array (
        df, 
        input_name="Data 'df' ", 
        to_frame =True, 
        )
    if columns is not None: 
        if isinstance (columns , str):
            columns = [columns]
        if len(columns)!= df.shape [1]: 
            raise TypeError("X and columns must be consistent,"
                            f" got {len(columns)} instead of {df.shape [1]}"
                            )
        df = pd.DataFrame(data = df, columns = columns )
        
    # create a new figure object  and define x axis position 
    # and y poaition , width, heigh of the dendrogram via the  
    # add_axes attributes. Furthermore, we rotate the dengrogram
    # to 90 degree counter-clockwise. 
    fig = plt.figure (figsize = fig_size , facecolor = facecolor )
    axd = fig.add_axes ([.09, .1, .2, .6 ])
    
    row_cluster = linkage_matrix(df = df, metric= metric, 
                                 method =method , kind = kind ,  
                                 )
    orient ='left' # use orientation 'right for matplotlib version < v1.5.1
    mpl_version = mpl.__version__.split('.')
    if mpl_version [0] =='1' : 
        if mpl_version [1] =='5' : 
            if float(mpl_version[2]) < 1. :
                orient = 'right'
                
    r = dendrogram(row_cluster , orientation= orient,  
                   **kwd )
    # 2. reorder the data in our initial dataframe according 
    # to the clustering label that can be accessed by a dendrogram 
    # which is essentially a Python dictionnary via a key leaves 
    df_rowclust = df.iloc [r['leaves'][::-1]] if hasattr(
        df, 'columns') else df  [r['leaves'][::-1]]
    
    # 3. construct the heatmap from the reordered dataframe and position 
    # in the next ro the dendrogram 
    axm = fig.add_axes ([.23, .1, .63, .6]) #.6 # [.23, .1, .2, .6]
    cax = axm.matshow (df_rowclust , 
                       interpolation = 'nearest' , 
                       cmap=cmap, 
                       )
    #4.  modify the asteric  of the dendogram  by removing the axis 
    # ticks and hiding the axis spines. Also we add a color bar and 
    # assign the feature and data record names to names x and y axis  
    # tick lables, respectively 
    axd.set_xticks ([]) # set ticks invisible 
    axd.set_yticks ([])
    for i in axd.spines.values () : 
        i.set_visible (False) 
        
    fig.colorbar(cax )
    xticks_loc = list(axm.get_xticks())
    yticks_loc = list(axm.get_yticks())

    df_rowclust_cols = df_rowclust.columns if hasattr (
        df_rowclust , 'columns') else [f"{i+1}" for i in range (df.shape[1])]
    axm.xaxis.set_major_locator(mticker.FixedLocator(xticks_loc))
    axm.xaxis.set_major_formatter(mticker.FixedFormatter(
        [''] + list (df_rowclust_cols)))
    
    df_rowclust_index = df_rowclust.index if hasattr(
        df_rowclust , 'columns') else [f"{i}" for i in range (df.shape[0])]
    axm.yaxis.set_major_locator(mticker.FixedLocator(yticks_loc))
    axm.yaxis.set_major_formatter(mticker.FixedFormatter(
        [''] + list (df_rowclust_index)))
    
    plt.show () 
    
    
def plot_dendrogram (
    df:DataFrame, 
    columns:List[str] =None, 
    labels: ArrayLike =None,
    metric:str ='euclidean',  
    method:str ='complete', 
    kind:str = None,
    return_r:bool =False, 
    verbose:bool=False, 
    **kwd ): 
    r""" Visualizes the linkage matrix in the results of dendrogram. 
    
    Note that the categorical features if exist in the dataframe should 
    automatically be discarded. 
    
    Parameters 
    -----------
    df: dataframe or NDArray of (n_samples, n_features) 
        dataframe of Ndarray. If array is given , must specify the column names
        to much the array shape 1 
        
    columns: list 
        list of labels to name each columns of arrays of (n_samples, n_features) 
        If dataframe is given, don't need to specify the columns. 
        
    kind: str, ['squareform'|'condense'|'design'], default is {'design'}
        kind of approach to summing up the linkage matrix. 
        Indeed, a condensed distance matrix is a flat array containing the 
        upper triangular of the distance matrix. This is the form that ``pdist`` 
        returns. Alternatively, a collection of :math:`m` observation vectors 
        in :math:`n` dimensions may be passed as  an :math:`m` by :math:`n` 
        array. All elements of the condensed distance matrix must be finite, 
        i.e., no NaNs or infs.
        Alternatively, we could used the ``squareform`` distance matrix to yield
        different distance values than expected. 
        the ``design`` approach uses the complete inpout example matrix  also 
        called 'design matrix' to lead correct linkage matrix similar to 
        `squareform` and `condense``. 
        
    metric : str or callable, default is {'euclidean'}
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`sklearn.metrics.pairwise.pairwise_distances`.
        If ``X`` is the distance array itself, use "precomputed" as the metric.
        Precomputed distance matrices must have 0 along the diagonal.
        
    method : str, optional, default is {'complete'}
        The linkage algorithm to use. See the ``Linkage Methods`` section below
        for full descriptions in :func:`gofast.tools.exmath.linkage_matrix`
        
    labels : ndarray, optional
        By default, ``labels`` is None so the index of the original observation
        is used to label the leaf nodes.  Otherwise, this is an :math:`n`-sized
        sequence, with ``n == Z.shape[0] + 1``. The ``labels[i]`` value is the
        text to put under the :math:`i` th leaf node only if it corresponds to
        an original observation and not a non-singleton cluster.
        
    return_r: bool, default='False', 
        return r-dictionnary if set to 'True' otherwise returns nothing 
    
    verbose: int, bool, default='False' 
        If ``True``, output message of the name of categorical features 
        dropped.
    
    kwd: dict 
        additional keywords arguments passes to 
        :func:`scipy.cluster.hierarchy.dendrogram` 
        
    Returns
    -------
    r : dict
        A dictionary of data structures computed to render the
        dendrogram. Its has the following keys:

        ``'color_list'``
          A list of color names. The k'th element represents the color of the
          k'th link.

        ``'icoord'`` and ``'dcoord'``
          Each of them is a list of lists. Let ``icoord = [I1, I2, ..., Ip]``
          where ``Ik = [xk1, xk2, xk3, xk4]`` and ``dcoord = [D1, D2, ..., Dp]``
          where ``Dk = [yk1, yk2, yk3, yk4]``, then the k'th link painted is
          ``(xk1, yk1)`` - ``(xk2, yk2)`` - ``(xk3, yk3)`` - ``(xk4, yk4)``.

        ``'ivl'``
          A list of labels corresponding to the leaf nodes.

        ``'leaves'``
          For each i, ``H[i] == j``, cluster node ``j`` appears in position
          ``i`` in the left-to-right traversal of the leaves, where
          :math:`j < 2n-1` and :math:`i < n`. If ``j`` is less than ``n``, the
          ``i``-th leaf node corresponds to an original observation.
          Otherwise, it corresponds to a non-singleton cluster.

        ``'leaves_color_list'``
          A list of color names. The k'th element represents the color of the
          k'th leaf.
          
    Examples 
    ----------
    >>> from gofast.datasets import load_iris 
    >>> from gofast.plot import  plot_dendrogram
    >>> data = load_iris () 
    >>> X =data.data[:, :2] 
    >>> plot_dendrogram (X, columns =['X1', 'X2' ] ) 

    """
    if hasattr (df, 'columns') and columns is not None: 
        df = df [columns ]
        
    df = to_numeric_dtypes(df, pop_cat_features= True, verbose =verbose )
    
    df=check_array (
        df, 
        input_name="Data 'df' ", 
        to_frame =True, 
        )

    kind:str = kind or 'design'
    row_cluster = linkage_matrix(df = df, columns = columns, metric= metric, 
                                 method =method , kind = kind ,
                                 )
    #make dendogram black (1/2)
    # set_link_color_palette(['black']) 
    r= dendrogram(row_cluster, labels= labels  , 
                           # make dendogram colors (2/2)
                           # color_threshold= np.inf,  
                           **kwd)
    plt.tight_layout()
    plt.ylabel ('Euclidian distance')
    plt.show ()
    
    return r if return_r else None 

def plot_silhouette (
    X:NDArray |DataFrame, 
    labels:ArrayLike=None, 
    prefit:bool=True, 
    n_clusters:int =3,  
    n_init: int=10 , 
    max_iter:int=300 , 
    random_state:int=None , 
    tol:float=1e4 , 
    metric:str='euclidean', 
    **kwd 
 ): 
    r"""
    quantifies the quality  of clustering samples. 
    
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training instances to cluster. It must be noted that the data
        will be converted to C ordering, which will cause a memory
        copy if the given data is not C-contiguous.
        If a sparse matrix is passed, a copy will be made if it's not in
        CSR format.
        
    labels : array-like 1d of shape (n_samples,)
        Label values for each sample.
         
    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.
        
    prefit : bool, default=False
        Whether a prefit `labels` is expected to be passed into the function
        directly or not.
        If `True`, `labels` must be a fit predicted values target.
        If `False`, `labels` is fitted and updated from `X` by calling
        `fit_predict` methods. Any other values passed to `labels` is 
        discarded.
         
    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance or None, default=42
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
    
    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        
    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`sklearn.metrics.pairwise.pairwise_distances`.
        If ``X`` is the distance array itself, use "precomputed" as the metric.
        Precomputed distance matrices must have 0 along the diagonal.

    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a ``scipy.spatial.distance`` metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.
        
    Note
    -------
    The sihouette coefficient is bound between -1 and 1 
    
    See More
    ---------
    Silhouette is used as graphical tools,  to plot a measure how tighly is  
    grouped the examples of the clusters are.  To calculate the silhouette 
    coefficient, three steps is allows: 
        
    * calculate the **cluster cohesion**, :math:`a(i)`, as the average 
        distance between examples, :math:`x^{(i)}`, and all the others 
        points
    * calculate the **cluster separation**, :math:`b^{(i)}` from the next 
        average distance between the example , :math:`x^{(i)}` amd all 
        the example of nearest cluster 
    * calculate the silhouette, :math:`s^{(i)}`, as the difference between 
        the cluster cohesion and separation divided by the greater of the 
        two, as shown here: 
    
        .. math:: 
            
            s^{(i)}=\frac{b^{(i)} - a^{(i)}}{max {{b^{(i)},a^{(i)} }}}
                
    Examples 
    --------
    >>> from gofast.datasets import load_hlogs 
    >>> from gofast.plot.evaluate  import plot_silhouette
    >>> # use resistivity and gamma for this demo
    >>> X_res_gamma = load_hlogs().frame[['resistivity', 'gamma_gamma']]  
    
    (1) Plot silhouette with 'prefit' set to 'False' 
    >>> plot_silhouette (X_res_gamma, prefit =False)
    
    """
    if  ( 
        not prefit 
        and labels is not None
        ): 
        warnings.warn("'labels' is given while 'prefix' is 'False'"
                      "'prefit' will set to 'True'")
        prefit=True 
        
    if labels is not None: 
        if not hasattr (labels, '__array__'): 
            raise TypeError( "Labels (target 'y') expects an array-like: "
                            f"{type(labels).__name__!r}")
        labels=check_y (
            labels, 
            to_frame =True, 
            )
        if len(labels)!=len(X): 
            raise TypeError("X and labels must have a consistency size."
                            f"{len(X)} and {len(labels)} respectively.")
            
    if prefit and labels is None: 
        raise TypeError ("Labels can not be None, while 'prefit' is 'True'"
                         " Turn 'prefit' to 'False' or provide the labels "
                         "instead.")
    if not prefit : 
        km= KMeans (n_clusters =n_clusters , 
                    init='k-means++', 
                    n_init =n_init , 
                    max_iter = max_iter , 
                    tol=tol, 
                    random_state =random_state
                        ) 
        labels = km.fit_predict(X ) 
        
    return _plot_silhouette(X, labels, metric = metric , **kwd)
    
    
def _plot_silhouette (X, labels, metric ='euclidean', **kwds ):
    r"""Plot quantifying the quality  of clustering silhouette 
    
    Parameters 
    ---------
    X : array-like of shape (n_samples_a, n_samples_a) if metric == \
            "precomputed" or (n_samples_a, n_features) otherwise
        An array of pairwise distances between samples, or a feature array.

    labels : array-like of shape (n_samples,)
        Label values for each sample.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`sklearn.metrics.pairwise.pairwise_distances`.
        If ``X`` is the distance array itself, use "precomputed" as the metric.
        Precomputed distance matrices must have 0 along the diagonal.

    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a ``scipy.spatial.distance`` metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.
        
    Examples
    ---------
    >>> import numpy as np 
    >>> from gofast.exlib.sklearn import KMeans 
    >>> from gofast.datasets import load_iris 
    >>> from gofast.plot.evaluate  import plot_silhouette
    >>> d= load_iris ()
    >>> X= d.data [:, 0][:, np.newaxis] # take the first axis 
    >>> km= KMeans (n_clusters =3 , init='k-means++', n_init =10 , 
                    max_iter = 300 , 
                    tol=1e-4, 
                    random_state =0 
                    )
    >>> y_km = km.fit_predict(X) 
    >>> plot_silhouette (X, y_km)
  
    See also 
    ---------
    gofast.tools.plotutils.plot_silhouette: Plot naive silhouette 
    
    Notes
    ------ 
    
    Silhouette is used as graphical tools,  to plot a measure how tighly is  
    grouped the examples of the clusters are.  To calculate the silhouette 
    coefficient, three steps is allows: 
        
    * calculate the **cluster cohesion**, :math:`a(i)`, as the average 
        distance between examples, :math:`x^{(i)}`, and all the others 
        points
    * calculate the **cluster separation**, :math:`b^{(i)}` from the next 
        average distance between the example , :math:`x^{(i)}` amd all 
        the example of nearest cluster 
    * calculate the silhouette, :math:`s^{(i)}`, as the difference between 
        the cluster cohesion and separation divided by the greater of the 
        two, as shown here: 
            
        .. math:: 
            
            s^{(i)}=\frac{b^{(i)} - a^{(i)}}{max {{b^{(i)},a^{(i)} }}}
    
    Note that the sihouette coefficient is bound between -1 and 1 
    
    """
    cluster_labels = np.unique (labels) 
    n_clusters = cluster_labels.shape [0] 
    silhouette_vals = silhouette_samples(X, labels= labels, metric = metric ,
                                         **kwds)
    y_ax_lower , y_ax_upper = 0, 0 
    yticks =[]
    
    for i, c  in enumerate (cluster_labels ) : 
        c_silhouette_vals = silhouette_vals[labels ==c ] 
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color =cm.jet (float(i)/n_clusters )
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, 
                 height =1.0 , 
                 edgecolor ='none', 
                 color =color, 
                 )
        yticks.append((y_ax_lower + y_ax_upper)/2.)
        y_ax_lower += len(c_silhouette_vals)
    silhouette_avg = np.mean(silhouette_vals) 
    plt.axvline (silhouette_avg, 
                 color='red', 
                 linestyle ='--'
                 )
    plt.yticks(yticks, cluster_labels +1 ) 
    plt.ylabel ("Cluster") 
    plt.xlabel ("Silhouette coefficient")
    plt.tight_layout()

    plt.show() 
    
def plot_learning_inspections (
    models:List[object] , 
    X:NDArray, y:ArrayLike,  
    fig_size:Tuple[int] = ( 22, 18 ) , 
    cv: int = None, 
    savefig:Optional[str] = None, 
    titles = None, 
    subplot_kws =None, 
    **kws 
  ): 
    """ Inspect multiple models from their learning curves. 
    
    Mutiples Inspection plots that generate the test and training learning 
    curve, the training  samples vs fit times curve, the fit times 
    vs score curve for each model.  
    
    Parameters
    ----------
    models : list of estimator instances
        Each estimator instance implements `fit` and `predict` methods which
        will be cloned for each validation.
    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer Sckikit-learn :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
        
    savefig: str, default =None , 
        the path to save the figures. Argument is passed to matplotlib.Figure 
        class. 
    titles: str, list 
       List of model names if changes are needed. If ``None``, model names 
       are used by default. 
    kws: dict, 
        Additional keywords argument passed to :func:`plot_learning_inspection`. 
        
    Returns
    ----------
    axes: Matplotlib axes
    
    See also 
    ---------
    plot_learning_inspection:  Inspect single model 
    
    Examples 
    ---------
    >>> from gofast.datasets import fetch_data
    >>> from gofast.models.premodels import p 
    >>> from gofast.plot.evaluate  import plot_learning_inspections 
    >>> # import sparse  matrix from Bagoue dataset 
    >>> X, y = fetch_data ('bagoue prepared') 
    >>> # import the two pretrained models from SVM 
    >>> models = [p.SVM.rbf.best_estimator_ , p.SVM.poly.best_estimator_]
    >>> plot_learning_inspections (models , X, y, ylim=(0.7, 1.01) )
    
    """
    models = is_iterable(models, exclude_string= True, transform =True )
    titles = list(is_iterable( 
        titles , exclude_string= True, transform =True )) 

    if len(titles ) != len(models): 
        titles = titles + [None for i in range (len(models)- len(titles))]
    # set the cross-validation to 4 
    cv = cv or 4  
    #set figure and subplots 
    if len(models)==1:
        msg = ( f"{plot_learning_inspection.__module__}."
               f"{plot_learning_inspection.__qualname__}"
               ) 
        raise PlotError ("For a single model inspection, use the"
                         f" function {msg!r} instead."
                         )
        
    fig , axes = plt.subplots (3 , len(models), figsize = fig_size )
    subplot_kws = subplot_kws or  dict(
        left=0.0625, right = 0.95, wspace = 0.1, hspace = .5 )
    
    fig.subplots_adjust(**subplot_kws)
    
    if not is_iterable( axes) :
        axes =[axes ] 
    for kk, model in enumerate ( models ) : 
        title = titles[kk] or  get_estimator_name (model )
        plot_learning_inspection(model, X=X , y=y, axes = axes [:, kk], 
                               title =title, 
                               **kws)
        
    if savefig : 
        fig.savefig (savefig , dpi = 300 )
    plt.show () if savefig is None else plt.close () 
    
def plot_learning_inspection(
    model,  
    X,  
    y, 
    axes=None, 
    ylim=None, 
    cv=5, 
    n_jobs=None,
    train_sizes=None, 
    display_legend = True, 
    title=None,
):
    """Inspect model from its learning curve. 
    
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    model : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
        
    display_legend: bool, default ='True' 
        display the legend
        
    Returns
    ----------
    axes: Matplotlib axes 
    
    Examples 
    ----------
    >>> from gofast.datasets import fetch_data
    >>> from gofast.models import p 
    >>> from gofast.plot.evaluate  import plot_learning_inspection 
    >>> # import sparse  matrix from Bagoue datasets 
    >>> X, y = fetch_data ('bagoue prepared') 
    >>> # import the  pretrained Radial Basis Function (RBF) from SVM 
    >>> plot_learning_inspection (p.SVM.rbf.best_estimator_  , X, y )
    
    """ 
    train_sizes = train_sizes or np.linspace(0.1, 1.0, 5)
    
    X, y = check_X_y(
        X, 
        y, 
        accept_sparse= True,
        to_frame =True 
        )
    
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    axes[0].set_title(title or get_estimator_name(model))
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        model,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].hlines(
        np.mean([train_scores[-1], test_scores[-1]]), 
        train_sizes[0],
        train_sizes[-1], 
        color="gray", 
        linestyle ="--", 
        label="Convergence score"
        )
    axes[0].plot(
        train_sizes, 
        train_scores_mean, 
        "o-", 
        color="r", 
        label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, 
        "o-", 
        color="g", 
        label="Cross-validation score"
    )

    if display_legend:
        axes[0].legend(loc="best")

    # set title name
    title_name = ( 
        f"{'the model'if title else get_estimator_name(model)}"
        )
    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title(f"Scalability of {title_name}")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, 
                 test_scores_mean_sorted, "o-"
                 )
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title(f"Performance of {title_name}")

    return axes

def plot_matshow(
    arr, / , labelx:List[str] =None, labely:List[str]=None, 
    matshow_kws=None, **baseplot_kws
    ): 
    
    #xxxxxxxxx update base plot keyword arguments
    for k  in list(baseplot_kws.keys()): 
        setattr (pobj , k, baseplot_kws[k])
        
    arr= check_array(
        arr, 
        to_frame =True, 
        input_name="Array 'arr'"
        )
    matshow_kws= matshow_kws or dict()
    fig = plt.figure(figsize = pobj.fig_size)

    ax = fig.add_subplot(1,1,1)
    
    cax = ax.matshow(arr, **matshow_kws) 
    cbax= fig.colorbar(cax, **pobj.cb_props)
    
    if pobj.cb_label is None: 
        pobj.cb_label=''
    ax.set_xlabel( pobj.xlabel,
          fontsize= pobj.font_size )
    
    
    # for label in zip ([labelx, labely]): 
    #     if label is not None:
    #         if not is_iterable(label):
    #             label = [label]
    #         if len(label) !=arr.shape[1]: 
    #             warnings.warn(
    #                 "labels and arr dimensions must be consistent"
    #                 f" Expect {arr.shape[1]}, got {len(label)}. "
    #                 )
                #continue
    if labelx is not None: 
        ax = _check_labelxy (labelx , arr, ax )
    if labely is not None: 
        ax = _check_labelxy (labely, arr, ax , axis ='y')
    
    if pobj.ylabel is None:
        pobj.ylabel =''
    if pobj.xlabel is None:
        pobj.xlabel = ''
    
    ax.set_ylabel (pobj.ylabel,
                   fontsize= pobj.font_size )
    ax.tick_params(axis=pobj.tp_axis, 
                    labelsize= pobj.font_size, 
                    bottom=pobj.tp_bottom, 
                    top=pobj.tp_top, 
                    labelbottom=pobj.tp_labelbottom, 
                    labeltop=pobj.tp_labeltop
                    )
    if pobj.tp_labeltop: 
        ax.xaxis.set_label_position('top')
    
    cbax.ax.tick_params(labelsize=pobj.font_size ) 
    cbax.set_label(label=pobj.cb_label,
                   size=pobj.font_size,
                   weight=pobj.font_weight)
    
    plt.xticks(rotation = pobj.rotate_xlabel)
    plt.yticks(rotation = pobj.rotate_ylabel)

    pobj.save(fig)

plot_matshow.__doc__ ="""\
Quick matrix visualization using matplotlib.pyplot.matshow.

Parameters
----------
arr: 2D ndarray, 
    matrix of n rowns and m-columns items 
matshow_kws: dict
    Additional keywords arguments for :func:`matplotlib.axes.matshow`
    
labelx: list of str, optional 
        list of labels names that express the name of each category on 
        x-axis. It might be consistent with the matrix number of 
        columns of `arr`. 
        
label: list of str, optional 
        list of labels names that express the name of each category on 
        y-axis. It might be consistent with the matrix number of 
        row of `arr`.
    
Examples
---------
>>> import numpy as np
>>> from gofast.plot.evaluate  import plot_matshow 
>>> matshow_kwargs ={
    'aspect': 'auto',
    'interpolation': None,
   'cmap':'copper_r', 
        }
>>> baseplot_kws ={'lw':3, 
           'lc':(.9, 0, .8), 
           'font_size':15., 
            'cb_format':None,
            #'cb_label':'Rate of prediction',
            'xlabel': 'Predicted flow classes',
            'ylabel': 'Geological rocks',
            'font_weight':None,
            'tp_labelbottom':False,
            'tp_labeltop':True,
            'tp_bottom': False
            }
>>> labelx =['FR0', 'FR1', 'FR2', 'FR3', 'Rates'] 
>>> labely =['VOLCANO-SEDIM. SCHISTS', 'GEOSYN. GRANITES', 
             'GRANITES', '1.0', 'Rates']
>>> array2d = np.array([(1. , .5, 1. ,1., .9286), 
                    (.5,  .8, 1., .667, .7692),
                    (.7, .81, .7, .5, .7442),
                    (.667, .75, 1., .75, .82),
                    (.9091, 0.8064, .7, .8667, .7931)])
>>> plot_matshow(array2d, labelx, labely, matshow_kwargs,**baseplot_kws )  

"""
  
def plot_unified_pca(
    components:NDArray,
    Xr: NDArray,
    y: ArrayLike,
    classes: ArrayLike=None,
    markers:List [str]=None, 
    colors: List [str ]=None, 
    **baseplot_kws, 
 ):
    """
    The biplot is the best way to visualize all-in-one following a PCA analysis.
    
    There is an implementation in R but there is no standard implementation
    in Python. 

    Parameters  
    -----------
    components: NDArray, shape (n_components, n_eigenvectors ), 
        the eigenvectors of the PCA. The shape in axis must much the number 
        of component computed using PCA. If the `Xr` shape 1 equals to the 
        shape 0 of the component matrix `components`, it will be transposed 
        to fit `Xr` shape 1.
        
    Xr: NDArray of transformed X. 
        the PCA projected data scores on n-given components.The reduced  
        dimension of train set 'X' with maximum ratio as sorted eigenvectors 
        from first to the last component. 
 
    y: Array-like, 
        the target composing the class labels.
    classes: list or int, 
        class categories or class labels 
    markers: str, 
        Matplotlib list of markers for plotting  classes.
    colors: str, 
        Matplotlib list of colors to customize plots 
        
    baseplot: dict, :class:`gofast.property.BasePlot`. 
        Matplotlib property from `BasePlot` instances. 

    Examples 
    ---------
    >>> from gofast.analysis import nPCA
    >>> from gofast.datasets import fetch_data
    >>> from gofast.plot import  plot_unified_pca, pobj_obj  #  is Baseplot instance 
    >>> X, y = fetch_data ('bagoue pca' )  # fetch pca data 
    >>> pca= nPCA (X, n_components= 2 , return_X= False ) # return PCA object 
    >>> components = pca.components_ [:2, :] # for two components 
    >>> plot_unified_pca ( components , pca.X, y ) # pca.X is the reduced dim X 
    >>> # to change for instance line width (lw) or style (ls) 
    >>> # just use the baseplotobject (pobj_obj)
    
    References 
    -----------
    Originally written by `Serafeim Loukas`_, serafeim.loukas@epfl.ch 
    and was edited to fit the :term:`gofast` package API. 
    
    .. _Serafeim Loukas: https://towardsdatascience.com/...-python-7c274582c37e
    
    """
    #xxxxxxxxx update base plot keyword arguments
    for k  in list(baseplot_kws.keys()): 
        setattr (pobj , k, baseplot_kws[k])
        
    Xr = check_array(
        Xr, 
        to_frame= False, 
        input_name="X reduced 'Xr'"
        )
    components = check_array(
        components, 
        to_frame =False ,
        input_name="PCA components"
        )
    Xr = np.array (Xr); components = np.array (components )
    xs = Xr[:,0] # projection on PC1
    ys = Xr[:,1] # projection on PC2
    
    if Xr.shape[1]==components.shape [0] :
        # i.e components is not transposed 
        # transposed then 
        components = components.T 
    n = components.shape[0] # number of variables
    
    fig = plt.figure(figsize=pobj.fig_size, #(10,8),
               dpi=pobj.fig_dpi #100
               )
    if classes is None: 
        classes = np.unique(y)
    if colors is None:
        # make color based on group
        # to fit length of classes
        colors = make_mpl_properties(
            len(classes))
        
    colors = [colors[c] for c in range(len(classes))]
    if markers is None:
        markers= make_mpl_properties(len(classes), prop='marker')
        
    markers = [markers[m] for m in range(len(classes))]
    
    for s,l in enumerate(classes):
        plt.scatter(xs[y==l],ys[y==l], 
                    color = colors[s], 
                    marker=markers[s]
                    ) 
    for i in range(n):
        # plot as arrows the variable scores 
        # (each variable has a score for PC1 and one for PC2)
        plt.arrow(0, 0, components[i,0], components[i,1], 
                  color = pobj.lc, #'k', 
                  alpha = pobj.alpha, #0.9,
                  linestyle = pobj.ls, # '-',
                  linewidth = pobj.lw, #1.5,
                  overhang=0.2)
        plt.text(components[i,0]* 1.15, components[i,1] * 1.15, 
                 "Var"+str(i+1),
                 color = 'k', 
                 ha = 'center',
                 va = 'center',
                 fontsize= pobj.font_size
                 )
    plt.tick_params(axis ='both', labelsize = pobj.font_size)
    
    plt.xlabel(pobj.xlabel or "PC1",size=pobj.font_size)
    plt.ylabel(pobj.ylabel or "PC2",size=pobj.font_size)
    limx= int(xs.max()) + 1
    limy= int(ys.max()) + 1
    plt.xlim([-limx,limx])
    plt.ylim([-limy,limy])
    plt.grid()
    plt.tick_params(axis='both',
                    which='both', 
                    labelsize=pobj.font_size
                    )
    
    pobj.save(fig)
    # if self.savefig is not None: 
    #     savefigure (plt, self.savefig, dpi = self.fig_dpi )
    
def _remaining_plot_roperties (self, ax, xlim=None, ylim=None, fig=None ): 
    """Append the remaining lines properties such as xlabel, grid , 
    legend and ticks parameters. Relevant idea to not 
    DRY(Don't Repeat Yourself). 
    :param ax: matplotlib.pyplot.axis 
    :param (xlim, ylim): Limit of x-axis and y-axis 
    :param fig: Matplotlib.figure name. 
    
    :return: self- Plot object. 
    """
    
    if self.xlabel is None: 
        self.xlabel =''
    if self.ylabel is None: 
        self.ylabel =''
        
    if xlim is not None: 
        ax.set_xlim(xlim)

    if ylim is not None: 
        ax.set_ylim(ylim)
        
    ax.set_xlabel( self.xlabel,
                  fontsize= .5 * self.font_size * self.fs )
    ax.set_ylabel (self.ylabel,
                   fontsize= .5 * self.font_size * self.fs)
    ax.tick_params(axis='both', 
                   labelsize=.5 * self.font_size * self.fs)
    
    if self.show_grid is True : 
       if self.gwhich =='minor': 
             ax.minorticks_on() 
       ax.grid(self.show_grid,
               axis=self.gaxis,
               which = self.gwhich, 
               color = self.gc,
               linestyle=self.gls,
               linewidth=self.glw, 
               alpha = self.galpha
               )
       
    if len(self.leg_kws) ==0 or 'loc' not in self.leg_kws.keys():
         self.leg_kws['loc']='best'
    
    ax.legend(**self.leg_kws)

    self.save(fig)
        
    return self 


def _chk_predict_args (Xt, yt, *args,  predict =False ): 
    """ Validate arguments passed  for model prediction 
    
    :param Xt: ndarray|DataFrame, test data 
    :param yt: array-like, pandas serie for test label 
    :param args: list of other keyword arguments which seems to be usefull. 
    :param predict: bool, expect a prediction or not. 
    :returns: Tuple (Xt, yt, index , clf ,  ypred )- tuple of : 
        * Xt : test data 
        * yt : test label data 
        * index :index to fit the samples in the dataframe or the 
            shape [0] of ndarray 
        * clf: the predictor or estimator 
        * ypred: the estimator predicted values 
        
    """
    # index is used for displayed the examples label in x-abscissa  
    # for instance index = ['b4, 'b5', 'b11',  ... ,'b425', 'b427', 'b430']
    
    index , clf ,  ypred = args 
    if index is not None:
        #control len of index and len of y
        if not is_iterable (index): 
            raise TypeError("Index is an iterable object with the same length"
                            "as 'y', got '{type (index).__name__!r}'") 
        len_index= len(yt)==len(index)
        
        if not len_index:
            warnings.warn(
                "Expect an index size be consistent with 'y' size={len(yt)},"
                  " got'{len(index)}'. Given index can not be used."
                  )
            index =None
            
        if len_index : 
            if isinstance(yt, (pd.Series, pd.DataFrame)):
                if not np.all(yt.index.isin(index)):
                    warnings.warn(
                        "Given index values are mismatched. Note that for "
                        "overlaying the model plot, 'Xt' indexes must be "
                        "identical to the one in target 'yt'. The indexes"
                        " provided are wrong and should be resetted."
                        )
                    index =yt.index 
                    yt=yt.values()
            yt= pd.Series(yt, index = index )
            
    if predict: 
        if clf is None: 
            warnings.warn("An estimator/classifier is needed for prediction."
                          " Got Nonetype.")
            raise EstimatorError("No estimator detected. Could not predict 'y'") 
        if Xt is None: 
            raise TypeError(
                "Test data 'Xt' is needed for prediction. Got nothing")
  
        # check estimator as callable object or ABCMeta classes
        if not hasattr(clf, '__call__') and  not inspect.isclass(clf)\
            and  type(clf.__class__)!=ABCMeta: 
            raise EstimatorError(
                f"{clf.__class__.__name__!r} is not an estimator/classifier."
                " 'y' prediction is aborted!")
            
        clf.fit(Xt, yt)
        ypred = clf.predict(Xt)
        
        if isinstance(Xt, (pd.DataFrame, pd.Series)):
            if index is None:
                index = Xt.index
                
    if isinstance(yt, pd.Series): 
        index = yt.index.astype('>U12')
    
    if index is None: 
        # take default values if  indexes are not given 
        index =np.array([i for i in range(len(yt))])

    if len(yt)!=len(ypred): 
        raise TypeError("'ypred'(predicted) and 'yt'(true target) sizes must"
                        f" be consistent. Expected {len(yt)}, got {len(ypred)}")
        
    return Xt, yt, index , clf ,  ypred 

def _check_labelxy (lablist, ar, ax, axis = 'x' ): 
    """ Assert whether the x and y labels given for setting the ticklabels 
    are consistent. 
    
    If consistent, function set x or y labels along the x or y axis 
    of the given array. 
    
    :param lablist: list, list of the label to set along x/y axis 
    :param ar: arraylike 2d, array to set x/y axis labels 
    :param ax: matplotlib.pyplot.Axes, 
    :param axis: str, default="x", kind of axis to set the label. 
    
    """
    warn_msg = ("labels along axis {axis} and arr dimensions must be"
                " consistent. Expects {shape}, got {len_label}")
    ax_ticks, ax_labels  = (ax.set_xticks, ax.set_xticklabels 
                         ) if axis =='x' else (
                             ax.set_yticks, ax.set_yticklabels )
    if lablist is not None: 
        lablist = is_iterable(lablist, exclude_string=True, 
                              transform =True )
        if not _check_consistency_size (
                lablist , ar[0 if axis =='x' else 1], error ='ignore'): 
            warnings.warn(warn_msg.format(
                axis = axis , shape=ar.shape[0 if axis =='x' else 1], 
                len_label=len(lablist))
                )
        else:
            ax_ticks(np.arange(0, ar.shape[0 if axis =='x' else 1]))
            ax_labels(lablist)
        
    return ax         
        
def plot2d(
    ar, 
    y=None,  
    x =None,  
    distance=50., 
    stnlist =None, 
    prefix ='S', 
    how= 'py',
    to_log10=False, 
    plot_contours=False,
    top_label='', 
    **baseplot_kws
    ): 
    """Two dimensional template for visualization matrices.
    
    It is a wrappers that can plot any matrice by customizing the position 
    X and y. By default X is considering as stations  and y the resistivity 
    log data. 
    
    Parameters 
    -----------
    ar: Array-like 2D, shape (M, N) 
        2D array for plotting. For instance, it can be a 2D resistivity 
        collected at all stations (N) and all frequency (M) 
    y: array-like, default=None
        Y-coordinates. It should have the length N, the same of the ``arr2d``.
        the rows of the ``arr2d``.
    x: array-like, default=None,  
        X-coordinates. It should have the length M, the same of the ``arr2d``; 
        the columns of the 2D dimensional array.  Note that if `x` is 
        given, the `distance is not needed. 

    distance: float 
        The step between two stations. If given, it creates an array of  
        position for plotting purpose. Default value is ``50`` meters. 
        
    stnlist: list of str 
        List of stations names. If given,  it should have the same length of 
        the columns M, of `arr2d`` 
       
    prefix: str 
        string value to add as prefix of given id. Prefix can be the site 
        name. Default is ``S``. 
        
    how: str 
        Mode to index the station. Default is 'Python indexing' i.e. 
        the counting of stations would starts by 0. Any other mode will 
        start the counting by 1.
     
    to_log10: bool, default=False 
       Recompute the `ar`  in logarithm  base 10 values. Note when ``True``, 
       the ``y`` should be also in log10. 
    plot_contours: bool, default=True 
       Plot the contours map. Is available only if the plot_style is set to 
       ``pcolormesh``. 
       
    top_label: str, 
       Name of the top label. 
       
    baseplot_kws: dict, 
       All all  the keywords arguments passed to the property  
       :class:`gofast.property.BasePlot` class. 
       
    Returns 
    -------
    axe: <AxesSubplot> object 
    
    Examples 
    -------- 
    >>> import numpy as np
    >>> import gofast 
    >>> np.random.seed (42) 
    >>> data = np.random.randn ( 15, 20 )
    >>> data_nan = data.copy() 
    >>> data_nan [2, 1] = np.nan; data_nan[4, 2]= np.nan;  data_nan[6, 3]=np.nan
    >>> gofast.plot.evaluate .plot2d (data )
    <AxesSubplot:xlabel='Distance(m)', ylabel='log10(Frequency)[Hz]'>
    >>> gofast.plot.evaluate .plot2d (data_nan ,  plt_style = 'imshow', 
                                  fig_size = (10, 4))
    """
    #xxxxxxxxx update base plot keyword arguments
    for k  in list(baseplot_kws.keys()): 
        setattr (pobj , k, baseplot_kws[k])
        
    if y is not None: 
        if len(y) != ar.shape [0]: 
            raise ValueError ("'y' array must have an identical number " 
                              f" of row of 2D array: {ar.shape[0]}")
            
    if x is not None: 
        if len(x) != ar.shape[1]: 
            raise ValueError (" 'x' array must have the same number " 
                              f" of columns of 2D array: {ar.shape[1]}")

    d= distance or 1.
    try : 
         distance = float(distance) 
    except : 
        raise TypeError (
             f'Expect a float value not {type(distance).__name__!r}')
        
    # put value to log10 if True 
    if to_log10: 
        ar = np.log10 (ar ) # assume the resistivity data 
        y = np.log10(y) if y is not None else y # assume the frequency data 

    y = np.arange(ar.shape [0]) if y is None else y 
    x=  x  or np.arange(ar.shape[1]) * d
         
    stn = stnlist or make_ids ( x , prefix , how = how) 
    #print(stnlis)
    if stn is not None: 
        stn = np.array(stn)
        
    if not _check_consistency_size(stn, x, error ="ignore"): 
        raise ValueError("The list of stations and positions must be"
                         f" consistent. {len(stnlist)} and {len(x)}"
                         " were given respectively")
            
    # make figure 
    fig, axe = plt.subplots(1,figsize = pobj.fig_size, 
                            num = pobj.fig_num,
                            dpi = pobj.fig_dpi
                            )
    
    cmap = plt.get_cmap( pobj.cmap)
    
    if pobj.plt_style not in ('pcolormesh','imshow' ): 
        warnings.warn(f"Unrecognized plot style {pobj.plt_style!r}."
                      " Expect ['pcolormesh'|'imshow']."
                      " 'pcolormesh' ( default) is used instead.")
        pobj.plt_style= 'pcolormesh'
        
    if pobj.plt_style =='pcolormesh': 
        X, Y = np.meshgrid (x, y)
        # ar = np.ma.masked_where(np.isnan(ar), ar)
        #Zm = ma.array(Z,mask=np.isnan(Z))
        pkws = dict (vmax = np.nanmax (ar),
                     vmin = np.nanmin (ar), 
                     ) 
        
        if plot_contours: 
            levels = mticker.MaxNLocator(nbins=15).tick_values(
                    np.nanmin (ar), np.nanmax(ar) )
            # delete vmin and Vmax : not supported 
            # when norm is passed 
            del pkws ['vmin'] ; del pkws ['vmax']
            pkws ['norm'] = BoundaryNorm(
                levels, ncolors=plt.colormaps[pobj.cmap].N, clip=True)
            
        
        ax = axe.pcolormesh ( X, Y, np.flipud (ar),
                    shading= pobj.plt_shading, 
                    cmap =cmap, 
                    **pkws 
            )
        if plot_contours: 
             # contours are *point* based plots, so convert 
             # our bound into point centers
            dx, dy = 0.05, 0.05
            axe.contourf(X+ dx/2.,
                         Y + dy/2., np.flipud (ar) , levels=levels,
                         cmap=plt.colormaps[pobj.cmap]
                         )
    if pobj.plt_style =='imshow': 
        ax = axe.imshow (ar,
                    interpolation = pobj.imshow_interp, 
                    cmap =cmap,
                    aspect = pobj.fig_aspect ,
                    origin= 'lower', 
                    extent=(  np.nanmin(x),
                              np.nanmax (x), 
                              np.nanmin(y), 
                              np.nanmax(y)
                              )
            )
    # set axis limit 
    axe.set_ylim(np.nanmin(y), 
                 np.nanmax(y))
    axe.set_xlim(np.nanmin(x), 
                 np.nanmax (x))

    cbl = 'log_{10}' if to_log10 else ''
    axe.set_xlabel(pobj.xlabel or 'Distance(m)', 
                 fontdict ={
                  'size': 1.5 * pobj.font_size ,
                  'weight': pobj.font_weight}
                 )
      
    axe.set_ylabel(pobj.ylabel or  f"{cbl}Frequency$[Hz]$",
             fontdict ={
                     #'style': pobj.font_style, 
                    'size':  1.5 * pobj.font_size ,
                    'weight': pobj.font_weight})
    if pobj.show_grid is True : 
        axe.minorticks_on()
        axe.grid(color='k', ls=':', lw =0.25, alpha=0.7, 
                     which ='major')
    
   
    labex = pobj.cb_label or f"{cbl}App.Res$[Ω.m]$" 
    
    cb = fig.colorbar(ax , ax= axe)
    cb.ax.yaxis.tick_left()
    cb.ax.tick_params(axis='y', direction='in', pad=2., 
                      labelsize = pobj.font_size )
    
    cb.set_label(labex,fontdict={'size': 1.2 * pobj.font_size ,
                              'style':pobj.font_style})
    #--> set second axis 
    axe2 = axe.twiny() 
    axe2.set_xticks(range(len(x)),minor=False )
    
    # set ticks params to reformat the size 
    axe.tick_params (  labelsize = pobj.font_size )
    axe2.tick_params (  labelsize = pobj.font_size )
    # get xticks and format labels using the auto detection 
    _get_xticks_formatage(axe2, stn, fmt = 'S{:02}',  auto=True, 
                          rotation=pobj.rotate_xlabel )
    
    axe2.set_xlabel(top_label, fontdict ={
        'style': pobj.font_style,
        'size': 1.5 * pobj.font_size ,
        'weight': pobj.font_weight}, )
      
    fig.suptitle(pobj.fig_title,ha='left',
                 fontsize= 15* pobj.fs, 
                 verticalalignment='center', 
                 style =pobj.font_style,
                 bbox =dict(boxstyle='round',
                            facecolor ='moccasin')
                 )
   
    #plt.tight_layout(h_pad =1.8, w_pad =2*1.08)
    plt.tight_layout()  
    if pobj.savefig is not None :
        fig.savefig(pobj.savefig, dpi = pobj.fig_dpi,
                    orientation =pobj.orient)
 
    plt.show() if pobj.savefig is None else plt.close(fig=fig) 
    
    
    return axe        

      
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        