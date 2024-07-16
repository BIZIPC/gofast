# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

r"""
Provides classes to create, save, and load configuration files for
hyperparameter tuning. It includes advanced optimization techniques such as 
genetic algorithms and Bayesian optimization.

Classes:
    - TuningManager
    - AdvancedTuningManager

Example
-------
First, define the content for the configuration file and other necessary data.

# Content of the configuration file (example.yaml)
config_content = '''
data:
  instruct_data: '/path/to/train_data.csv'
  eval_instruct_data: '/path/to/eval_data.csv'
  data: '/path/to/pretrain_data.csv'
model:
  model_id_or_path: '/path/to/model_id'
run_dir: '/path/to/run_dir'
'''

# Write the configuration content to a file
with open('example.yaml', 'w') as file:
    file.write(config_content)

# Define paths and variables
config_path = 'example.yaml'
model_id_path = '/path/to/model_id'
run_dir = '/path/to/run_dir'
train_data = '/path/to/train_data.csv'
eval_data = '/path/to/eval_data.csv'
pretrain_data = '/path/to/pretrain_data.csv'

# Create a TuningManager instance
from gofast.models.tune import TuningManager, AdvancedTuningManager

tuning_manager = TuningManager(
    config_path=config_path,
    model_id_path=model_id_path,
    run_dir=run_dir,
    train_data=train_data,
    eval_data=eval_data,
    pretrain_data=pretrain_data
)

# Save the configuration to a file
tuning_manager.save_config(format='yaml')

# Load the configuration from a file
config = tuning_manager.load_config()
print(config)

# Define an estimator and parameter grid
from sklearn.svm import SVC
estimator = SVC()
param_grid = {'C': [1, 10], 'kernel': ['linear', 'rbf']}

# Run the tuning process using the AdvancedTuningManager
advanced_tuning_manager = AdvancedTuningManager(
    config_path=config_path,
    model_id_path=model_id_path,
    run_dir=run_dir,
    train_data=train_data,
    eval_data=eval_data,
    pretrain_data=pretrain_data
)

advanced_tuning_manager.run(
    estimator, 
    param_grid=param_grid, 
    strategy='bayesian', 
    cv=5, 
    scoring='accuracy', 
    n_jobs=1,
    test_size=0.25,
    random_state=24,
    max_iter=30
)

See Also
--------
gofast.models.optimize.Optimizer : Class for hyperparameter optimization.

References
----------
.. [1] Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter 
       Optimization. Journal of Machine Learning Research, 13, 281-305.
.. [2] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
       Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine 
       Learning in Python. Journal of Machine Learning Research, 12, 
       2825-2830.
"""

import os
import yaml
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from ..api.summary import ModelSummary
from ..tools.funcutils import ensure_pkg
from .optimize import Optimizer
# Try importing TPOTClassifier, handle the exception if not installed
try: 
    from tpot import TPOTClassifier
except ImportError: 
    pass 

__all__= ["TuningManager", "AdvancedTuningManager"]

class TuningManager:
    """
    A class to manage hyperparameter tuning configurations and execution.

    This class provides methods to create, save, and load configuration 
    files for hyperparameter tuning. It uses the Optimizer class from 
    `gofast.models.optimize` to perform the tuning process based on the 
    configuration.

    Parameters
    ----------
    config_path : str
        Path to the configuration file. This can be either a YAML or JSON 
        file that contains the hyperparameter tuning setup, including 
        paths to data files, model parameters, and other necessary 
        configuration settings.
    model_id_path : str, optional
        Path to the model ID or directory containing the model. This 
        directory should contain the pretrained models or model 
        checkpoints that will be used or fine-tuned during the hyperparameter 
        optimization process.
    run_dir : str, optional
        Directory where the results of the training runs will be saved. 
        This directory will store outputs such as logs, model checkpoints, 
        and any other files generated during the optimization process.
    train_data : str, optional
        Path to the training data file. This file should contain the 
        dataset used to train the model. Supported formats include CSV, 
        TSV, JSON, Excel, and NumPy binary files.
    eval_data : str, optional
        Path to the evaluation data file. This file should contain the 
        dataset used to evaluate the model during training. It is optional 
        but recommended to monitor the performance of the model on a 
        separate validation set.
    pretrain_data : str, optional
        Path to the pretraining data file. This file can be used to specify 
        an additional dataset for pretraining the model before fine-tuning 
        on the main training dataset.

    Methods
    -------
    create_config():
        Create a configuration dictionary for hyperparameter tuning.
        This dictionary includes all the specified paths and settings 
        required for the tuning process.

    save_config(format='yaml'):
        Save the configuration dictionary to a file in the specified format 
        (YAML or JSON). The format is determined by the file extension 
        provided.

    load_config():
        Load the configuration from a file (YAML or JSON). The method 
        automatically detects the file format based on the file extension 
        and parses it accordingly.

    run(estimator, param_grid, strategy='GSCV', cv=5, scoring=None, n_jobs=-1, 
        test_size=0.2, random_state=42):
        Run the optimizer using the specified parameters. This method 
        initializes the optimizer with the provided estimator and parameter 
        grid, splits the data into training and testing sets, and executes 
        the hyperparameter optimization process.

    load_data(data_path, file_format='csv', delimiter=',', header='infer', 
              target_column='target', feature_columns=None, na_values=None, 
              dropna=True, dtype=None):
        Load data from a given path. This method supports various file 
        formats such as CSV, TSV, JSON, Excel, and NumPy binary files. 
        It allows for flexible handling of data, including specifying 
        delimiters, handling missing values, and selecting specific columns.

    Notes
    -----
    The TuningManager class is designed to simplify the management of 
    hyperparameter tuning tasks. It supports configuration files in both 
    YAML and JSON formats, making it flexible and easy to integrate into 
    different workflows.

    The goal of hyperparameter tuning is to find the best set of 
    hyperparameters :math:`\theta` that minimizes or maximizes a given 
    scoring function. Formally, given a set of hyperparameters :math:`\theta`, 
    the objective is to find:

    .. math::
        \theta^* = \arg\min_{\theta \in \Theta} \mathbb{E}_{(X, y) \sim D} 
        [L(f(X; \theta), y)]

    where :math:`\Theta` represents the hyperparameter space, :math:`D` 
    denotes the data distribution, :math:`L` is the loss function, and 
    :math:`f(X; \theta)` is the model's prediction function.

    Examples
    --------
    >>> from gofast.models.optimize import TuningManager
    >>> tuning_manager = TuningManager(
            config_path='example.yaml',
            model_id_path='/content/mistral_models',
            run_dir='/content/test_ultra',
            train_data='/content/data/ultrachat_chunk_train.csv',
            eval_data='/content/data/ultrachat_chunk_eval.csv'
        )

    Save configuration as YAML
    >>> tuning_manager.save_config(format='yaml')

    Save configuration as JSON
    >>> tuning_manager.save_config(format='json')

    Load configuration
    >>> config = tuning_manager.load_config()
    >>> print(config)

    Run optimizer
    >>> from sklearn.svm import SVC
    >>> estimator = SVC()
    >>> param_grid = {'C': [1, 10], 'kernel': ['linear', 'rbf']}
    >>> tuning_manager.run(
            estimator, 
            param_grid, 
            strategy='SWCV', 
            cv=5, 
            scoring='accuracy', 
            n_jobs=1,
            test_size=0.25,
            random_state=24
        )

    See Also
    --------
    gofast.models.optimize.Optimizer : Class for hyperparameter optimization.

    References
    ----------
    .. [1] Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter 
           Optimization. Journal of Machine Learning Research, 13, 281-305.
    .. [2] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
           Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine 
           Learning in Python. Journal of Machine Learning Research, 12, 
           2825-2830.
    """

    def __init__(
        self, 
        config_path, 
        model_id_path=None, 
        run_dir=None, 
        train_data=None, 
        eval_data=None, 
        pretrain_data=None
    ):
        self.config_path = config_path
        self.model_id_path = model_id_path
        self.run_dir = run_dir
        self.train_data = train_data
        self.eval_data = eval_data
        self.pretrain_data = pretrain_data


    def run(
        self, 
        estimator, 
        param_grid, 
        strategy='GSCV', 
        cv=5, 
        scoring=None, 
        n_jobs=-1,
        test_size=0.2, 
        random_state=42
    ):
        """
        Run the optimizer using the specified parameters.

        This method initializes the optimizer with the provided estimator 
        and parameter grid, splits the data into training and testing sets, 
        and executes the hyperparameter optimization process. It combines 
        training and evaluation data if both are provided.

        Parameters
        ----------
        estimator : estimator object
            The machine learning estimator to optimize. It should implement 
            the scikit-learn estimator interface (`fit`, `predict`, etc.).
        
        param_grid : dict
            Dictionary with parameters names (`str`) as keys and lists of 
            parameter settings to try as values. This can also be a list of 
            such dictionaries, in which case the grids spanned by each 
            dictionary in the list are explored. This enables searching over 
            any sequence of parameter settings.
        
        strategy : str, optional, default='GSCV'
            The search strategy to apply for hyperparameter optimization. 
            Supported strategies include:
            - 'GSCV' for Grid Search Cross Validation,
            - 'RSCV' for Randomized Search Cross Validation,
            - 'BSCV' for Bayesian Optimization,
            - 'ASCV' for Simulated Annealing-based Search,
            - 'SWSCV' for Particle Swarm Optimization,
            - 'SQSCV' for Sequential Model-Based Optimization,
            - 'EVSCV' for Evolutionary Algorithms-based Search,
            - 'GBSCV' for Gradient-Based Optimization,
            - 'GENSCV' for Genetic Algorithms-based Search.
        
        cv : int, default=5
            Determines the cross-validation splitting strategy. Possible 
            inputs for `cv` are:
            - None, to use the default 5-fold cross-validation,
            - int, to specify the number of folds in a (Stratified)KFold,
            - CV splitter,
            - An iterable yielding (train, test) splits as arrays of indices.
        
        scoring : str or callable, optional
            A string (see model evaluation documentation) or a scorer 
            callable object / function with signature 
            `scorer(estimator, X, y)`.
        
        n_jobs : int, default=-1
            The number of jobs to run in parallel. `-1` means using all 
            processors.
        
        test_size : float or int, default=0.2
            If float, should be between 0.0 and 1.0 and represent the 
            proportion of the dataset to include in the test split. If int, 
            represents the absolute number of test samples.
        
        random_state : int, RandomState instance or None, default=42
            Controls the shuffling applied to the data before applying the 
            split. Pass an int for reproducible output across multiple 
            function calls.

        Returns
        -------
        None
        
        Notes
        -----
        This method leverages the Optimizer class from the `gofast.models.optimize` 
        module to perform hyperparameter tuning. It manages the data loading 
        and preparation process, ensuring that the data is properly split into 
        training and testing sets before optimization.

        Examples
        --------
        >>> from gofast.models.tune import TuningManager
        >>> from sklearn.svm import SVC
        >>> tuning_manager = TuningManager(
                config_path='example.yaml',
                model_id_path='/content/mistral_models',
                run_dir='/content/test_ultra',
                train_data='/content/data/ultrachat_chunk_train.csv',
                eval_data='/content/data/ultrachat_chunk_eval.csv'
            )
        >>> estimator = SVC()
        >>> param_grid = {'C': [1, 10], 'kernel': ['linear', 'rbf']}
        >>> tuning_manager.run(
                estimator, 
                param_grid, 
                strategy='SWCV', 
                cv=5, 
                scoring='accuracy', 
                n_jobs=1,
                test_size=0.25,
                random_state=24
            )

        See Also
        --------
        gofast.models.optimize.Optimizer : Class for hyperparameter optimization.

        References
        ----------
        .. [1] Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter 
               Optimization. Journal of Machine Learning Research, 13, 281-305.
        .. [2] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
               Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine 
               Learning in Python. Journal of Machine Learning Research, 12, 
               2825-2830.
        """
        
        config = self.load_config()

        # Check and load training data
        if 'instruct_data' not in config['data'] or not config['data']['instruct_data']:
            raise ValueError(
                "Training data path must be provided in the configuration.")
        X, y = self.load_data(config['data']['instruct_data'])
    
        # Check and load evaluation data if provided
        if 'eval_instruct_data' in config['data'] and config['data']['eval_instruct_data']:
            X_eval, y_eval = self.load_data(config['data']['eval_instruct_data'])
            X = np.concatenate((X, X_eval), axis=0)
            y = np.concatenate((y, y_eval), axis=0)
    
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
    
        optimizer = Optimizer(
            estimators={type(estimator).__name__: estimator},
            param_grids={type(estimator).__name__: param_grid},
            strategy=strategy,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs
        )
    
        results = optimizer.fit(X_train, y_train)
        print(results)

    def load_data(
        self, 
        data_path, 
        file_format='csv', 
        delimiter=',', 
        header='infer', 
        target_column='target', 
        feature_columns=None,
        na_values=None, 
        dropna=True, 
        dtype=None
    ):
        """
        Load data from a given path.
    
        This method supports various file formats such as CSV, TSV, JSON, 
        Excel, and NumPy binary files. It allows for flexible handling of 
        data, including specifying delimiters, handling missing values, and 
        selecting specific columns.
    
        Parameters
        ----------
        data_path : str
            Path to the data file. The file should contain the dataset used 
            for training or evaluation.
        
        file_format : str, optional, default='csv'
            The format of the data file. Supported formats are:
            - 'csv' : Comma-separated values file.
            - 'tsv' : Tab-separated values file.
            - 'txt' : Text file with a custom delimiter.
            - 'json' : JSON formatted file.
            - 'excel' : Excel file (xlsx or xls).
            - 'npy' : NumPy binary file containing a dictionary with 'features' 
              and 'target' keys.
        
        delimiter : str, optional, default=','
            The delimiter to use if the file is a delimited text file (csv, tsv, txt).
        
        header : int, list of int, str, or None, optional, default='infer'
            Row number(s) to use as the column names. Defaults to 'infer', 
            which means the first line of the file is used as the column names.
        
        target_column : str, optional, default='target'
            The name of the target column. This column contains the labels or 
            target values for the dataset.
        
        feature_columns : list of str, optional
            List of feature columns to use. If None, all columns except the 
            target column are used as features.
        
        na_values : scalar, str, list-like, or dict, optional
            Additional strings to recognize as NA/NaN. If dict passed, specific 
            per-column NA values.
        
        dropna : bool, optional, default=True
            If True, drop rows with missing values.
        
        dtype : Type name or dict of column -> type, optional
            Data type for data or columns. E.g. `{'a': np.float64, 'b': np.int32}`
    
        Returns
        -------
        X : array-like
            Feature data. This is the input data for training or evaluation.
        
        y : array-like
            Target data. These are the labels or target values corresponding to 
            the feature data.
    
        Raises
        ------
        ValueError
            If the specified file format is not supported, or if the .npy file 
            format is invalid.
    
        Notes
        -----
        This method provides a flexible and robust way to load datasets from 
        various file formats commonly used in machine learning. It ensures that 
        the data is properly formatted and ready for use in training or 
        evaluation.
    
        Examples
        --------
        >>> from gofast.models.tune import TuningManager
        >>> tuning_manager = TuningManager(
                config_path='example.yaml',
                model_id_path='/content/mistral_models',
                run_dir='/content/test_ultra',
                train_data='/content/data/ultrachat_chunk_train.csv',
                eval_data='/content/data/ultrachat_chunk_eval.csv'
            )
    
        Load CSV data
        >>> X, y = tuning_manager.load_data(
                data_path=tuning_manager.train_data, 
                file_format='csv', 
                target_column='label'
            )
    
        Load JSON data
        >>> X, y = tuning_manager.load_data(
                data_path='data.json', 
                file_format='json', 
                target_column='label'
            )
    
        Load Excel data
        >>> X, y = tuning_manager.load_data(
                data_path='data.xlsx', 
                file_format='excel', 
                target_column='label'
            )
    
        Load TSV data
        >>> X, y = tuning_manager.load_data(
                data_path='data.tsv', 
                file_format='tsv', 
                target_column='label'
            )
    
        Load NumPy data
        >>> X, y = tuning_manager.load_data(
                data_path='data.npy', 
                file_format='npy'
            )
    
        See Also
        --------
        pandas.read_csv : Read a comma-separated values (csv) file into DataFrame.
        pandas.read_json : Convert a JSON string to pandas object.
        pandas.read_excel : Read an Excel file into a pandas DataFrame.
        numpy.load : Load arrays or pickled objects from .npy, .npz, or pickled files.
    
        References
        ----------
        .. [1] McKinney, W. (2010). Data Structures for Statistical Computing in 
               Python. In Proceedings of the 9th Python in Science Conference, 
               51-56.
        .. [2] Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., 
               Virtanen, P., Cournapeau, D., ... & Oliphant, T. E. (2020). 
               Array programming with NumPy. Nature, 585(7825), 357-362.
        """
        if file_format == 'csv':
            data = pd.read_csv(data_path, delimiter=delimiter, header=header, 
                               na_values=na_values, dtype=dtype)
        elif file_format == 'tsv':
            data = pd.read_csv(data_path, delimiter='\t', header=header, 
                               na_values=na_values, dtype=dtype)
        elif file_format == 'txt':
            data = pd.read_csv(data_path, delimiter=delimiter, header=header,
                               na_values=na_values, dtype=dtype)
        elif file_format == 'json':
            data = pd.read_json(data_path)
        elif file_format == 'excel':
            data = pd.read_excel(data_path, header=header, na_values=na_values,
                                 dtype=dtype)
        elif file_format == 'npy':
            data = np.load(data_path, allow_pickle=True).item()
            if isinstance(data, dict):
                X = np.array(data['features'])
                y = np.array(data['target'])
                return X, y
            else:
                raise ValueError(
                    "Invalid .npy file format. Expected a dictionary with"
                    " 'features' and 'target' keys.")
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
        if dropna:
            data.dropna(inplace=True)
    
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
    
        X = data[feature_columns].values
        y = data[target_column].values
    
        return X, y

    def create_config(self):
        """
        Create a configuration dictionary for hyperparameter tuning.

        This method compiles all the specified paths and settings into a 
        dictionary format that is used for the hyperparameter tuning process. 
        The resulting configuration includes data paths, model paths, and 
        directory settings required for running the optimization.

        Returns
        -------
        config : dict
            A dictionary containing the configuration settings including 
            paths for training, evaluation, and pretraining data, model ID 
            or path, and the run directory.
        """
        config = {
            'data': {
                'instruct_data': self.train_data,
                'eval_instruct_data': self.eval_data if self.eval_data else '',
                'data': self.pretrain_data if self.pretrain_data else ''
            },
            'model': {
                'model_id_or_path': self.model_id_path
            },
            'run_dir': self.run_dir
        }
        return config

    def save_config(self, format='yaml'):
        """
        Save the configuration dictionary to a file in the specified format 
        (YAML or JSON).

        Parameters
        ----------
        format : str, optional
            The format in which to save the configuration file. Supported 
            formats are 'yaml' and 'json'. The default is 'yaml'.

        Raises
        ------
        ValueError
            If the specified format is not supported (i.e., not 'yaml' or 'json').
        """
        config = self.create_config()
        with open(self.config_path, 'w') as file:
            if format == 'yaml':
                yaml.dump(config, file)
            elif format == 'json':
                json.dump(config, file, indent=4)
            else:
                raise ValueError("Unsupported file format. Use 'yaml' or 'json'.")

    def load_config(self):
        """
        Load the configuration from a file (YAML or JSON).

        This method reads the configuration file specified in `config_path` 
        and loads the contents into a dictionary. It automatically detects 
        the file format based on the file extension (either `.yaml`, `.yml`, 
        or `.json`).

        Returns
        -------
        config : dict
            A dictionary containing the configuration settings loaded from 
            the file.

        Raises
        ------
        ValueError
            If the file format is not supported (i.e., not 'yaml', 'yml', or 'json').
        """
        _, file_extension = os.path.splitext(self.config_path)
        with open(self.config_path, 'r') as file:
            if file_extension == '.yaml' or file_extension == '.yml':
                config = yaml.safe_load(file)
            elif file_extension == '.json':
                config = json.load(file)
            else:
                raise ValueError("Unsupported file format. Use 'yaml' or 'json'.")
        return config


class AdvancedTuningManager(TuningManager):
    """
    AdvancedTuningManager extends TuningManager to provide more sophisticated
    hyperparameter tuning strategies, including genetic algorithms and Bayesian
    optimization.

    Parameters
    ----------
    config_path : str
        Path to the configuration file. This can be either a YAML or JSON 
        file that contains the hyperparameter tuning setup, including 
        paths to data files, model parameters, and other necessary 
        configuration settings.
    model_id_path : str, optional
        Path to the model ID or directory containing the model. This 
        directory should contain the pretrained models or model 
        checkpoints that will be used or fine-tuned during the hyperparameter 
        optimization process.
    run_dir : str, optional
        Directory where the results of the training runs will be saved. 
        This directory will store outputs such as logs, model checkpoints, 
        and any other files generated during the optimization process.
    train_data : str, optional
        Path to the training data file. This file should contain the 
        dataset used to train the model. Supported formats include CSV, 
        TSV, JSON, Excel, and NumPy binary files.
    eval_data : str, optional
        Path to the evaluation data file. This file should contain the 
        dataset used to evaluate the model during training. It is optional 
        but recommended to monitor the performance of the model on a 
        separate validation set.
    pretrain_data : str, optional
        Path to the pretraining data file. This file can be used to specify 
        an additional dataset for pretraining the model before fine-tuning 
        on the main training dataset.

    Methods
    -------
    run(estimator, param_grid=None, strategy='RSCV', cv=5, scoring=None, 
        n_jobs=-1, test_size=0.2, random_state=42, max_iter=50, 
        population_size=50, generations=50, verbose=2):
        Run the optimizer using the specified parameters. This method 
        initializes the optimizer with the provided estimator and parameter 
        grid, splits the data into training and testing sets, and executes 
        the hyperparameter optimization process.

    Notes
    -----
    The AdvancedTuningManager class builds on the TuningManager class by 
    adding support for advanced hyperparameter tuning techniques such as 
    genetic algorithms (via TPOT) and Bayesian optimization (via skopt). 
    It ensures the necessary packages are installed and provides a flexible 
    framework for hyperparameter tuning.

    The goal of hyperparameter tuning is to find the best set of 
    hyperparameters :math:`\theta` that minimizes or maximizes a given 
    scoring function. Formally, given a set of hyperparameters :math:`\theta`, 
    the objective is to find:

    .. math::
        \theta^* = \arg\min_{\theta \in \Theta} \mathbb{E}_{(X, y) \sim D} 
        [L(f(X; \theta), y)]

    where :math:`\Theta` represents the hyperparameter space, :math:`D` 
    denotes the data distribution, :math:`L` is the loss function, and 
    :math:`f(X; \theta)` is the model's prediction function.

    Examples
    --------
    >>> from gofast.models.tune import AdvancedTuningManager
    >>> tuning_manager = AdvancedTuningManager(
            config_path='example.yaml',
            model_id_path='/content/mistral_models',
            run_dir='/content/test_ultra',
            train_data='/content/data/ultrachat_chunk_train.csv',
            eval_data='/content/data/ultrachat_chunk_eval.csv'
        )

    >>> from sklearn.svm import SVC
    >>> estimator = SVC()
    >>> param_grid = {'C': [1, 10], 'kernel': ['linear', 'rbf']}

    >>> tuning_manager.run(
            estimator, 
            param_grid=param_grid, 
            strategy='bayesian', 
            cv=5, 
            scoring='accuracy', 
            n_jobs=1,
            test_size=0.25,
            random_state=24,
            max_iter=30
        )

    See Also
    --------
    gofast.models.optimize.Optimizer : Class for hyperparameter optimization.

    References
    ----------
    .. [1] Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter 
           Optimization. Journal of Machine Learning Research, 13, 281-305.
    .. [2] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
           Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine 
           Learning in Python. Journal of Machine Learning Research, 12, 
           2825-2830.
    """
    def __init__(
        self, 
        config_path, 
        model_id_path=None, 
        run_dir=None, 
        train_data=None, 
        eval_data=None, 
        pretrain_data=None
    ):
        # Initialize the superclass TuningManager
        super().__init__(
            config_path=config_path, 
            model_id_path=model_id_path, 
            run_dir=run_dir, 
            train_data=train_data, 
            eval_data=eval_data, 
            pretrain_data=pretrain_data
        )
    
    # Decorator to ensure TPOT is installed if the strategy requires it
    @ensure_pkg(
        "tpot", 
        extra=(
            "`tpot` should be installed when setting strategy to"
            " `tpogen` or `genetic`"
        ), 
        partial_check=True,
        condition=lambda *args, **kwargs: kwargs.get(
            "strategy") in ('tpogen', 'genetic', 'topt')
    )
    def run(
        self, 
        estimator, 
        param_grid=None, 
        strategy='RSCV', 
        cv=5, 
        scoring=None, 
        n_jobs=-1,
        test_size=0.2, 
        random_state=42,
        max_iter=50,
        population_size=50,
        generations=50,
        verbose=2
    ):
        """
        Run the optimizer using the specified parameters.
    
        This method initializes the optimizer with the provided estimator 
        and parameter grid, splits the data into training and testing sets, 
        and executes the hyperparameter optimization process using different 
        strategies such as Randomized Search, Grid Search, Bayesian optimization, 
        and genetic algorithms.
    
        Parameters
        ----------
        estimator : estimator object
            The machine learning estimator to optimize. It should implement 
            the scikit-learn estimator interface (`fit`, `predict`, etc.).
        
        param_grid : dict, optional
            Dictionary with parameters names (`str`) as keys and lists of 
            parameter settings to try as values. This can also be a list of 
            such dictionaries, in which case the grids spanned by each 
            dictionary in the list are explored. This enables searching over 
            any sequence of parameter settings.
        
        strategy : str, optional, default='RSCV'
            The search strategy to apply for hyperparameter optimization. 
            Supported strategies include:
            - 'RSCV' for Randomized Search Cross Validation,
            - 'GSCV' for Grid Search Cross Validation,
            - 'bayesian' for Bayesian optimization,
            - 'TPOGEN' for genetic algorithms using TPOT.
        
        cv : int, default=5
            Determines the cross-validation splitting strategy. Possible 
            inputs for `cv` are:
            - None, to use the default 5-fold cross-validation,
            - int, to specify the number of folds in a (Stratified)KFold,
            - CV splitter,
            - An iterable yielding (train, test) splits as arrays of indices.
        
        scoring : str or callable, optional
            A string (see model evaluation documentation) or a scorer 
            callable object / function with signature 
            `scorer(estimator, X, y)`.
        
        n_jobs : int, default=-1
            The number of jobs to run in parallel. `-1` means using all 
            processors.
        
        test_size : float or int, default=0.2
            If float, should be between 0.0 and 1.0 and represent the 
            proportion of the dataset to include in the test split. If int, 
            represents the absolute number of test samples.
        
        random_state : int, RandomState instance or None, default=42
            Controls the shuffling applied to the data before applying the 
            split. Pass an int for reproducible output across multiple 
            function calls.
        
        max_iter : int, default=50
            Maximum number of iterations for optimization strategies that 
            support it, such as Bayesian optimization.
        
        population_size : int, default=50
            Population size for genetic algorithm-based optimization.
        
        generations : int, default=50
            Number of generations for genetic algorithm-based optimization.
        
        verbose : int, default=2
            Controls the verbosity: the higher, the more messages.
    
        Returns
        -------
        None
        
        Notes
        -----
        This method leverages advanced optimization strategies to tune the 
        hyperparameters of the provided estimator. It ensures the necessary 
        packages are installed and appropriately configures the optimizer 
        based on the selected strategy.
    
        The goal of hyperparameter tuning is to find the best set of 
        hyperparameters :math:`\theta` that minimizes or maximizes a given 
        scoring function. Formally, given a set of hyperparameters :math:`\theta`, 
        the objective is to find:
    
        .. math::
            \theta^* = \arg\min_{\theta \in \Theta} \mathbb{E}_{(X, y) \sim D} 
            [L(f(X; \theta), y)]
    
        where :math:`\Theta` represents the hyperparameter space, :math:`D` 
        denotes the data distribution, :math:`L` is the loss function, and 
        :math:`f(X; \theta)` is the model's prediction function.
    
        Examples
        --------
        >>> from gofast.models.tune import AdvancedTuningManager
        >>> tuning_manager = AdvancedTuningManager(
                config_path='example.yaml',
                model_id_path='/content/mistral_models',
                run_dir='/content/test_ultra',
                train_data='/content/data/ultrachat_chunk_train.csv',
                eval_data='/content/data/ultrachat_chunk_eval.csv'
            )
    
        >>> from sklearn.svm import SVC
        >>> estimator = SVC()
        >>> param_grid = {'C': [1, 10], 'kernel': ['linear', 'rbf']}
    
        >>> tuning_manager.run(
                estimator, 
                param_grid=param_grid, 
                strategy='bayesian', 
                cv=5, 
                scoring='accuracy', 
                n_jobs=1,
                test_size=0.25,
                random_state=24,
                max_iter=30
            )
    
        See Also
        --------
        gofast.models.optimize.Optimizer : Class for hyperparameter optimization.
    
        References
        ----------
        .. [1] Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter 
               Optimization. Journal of Machine Learning Research, 13, 281-305.
        .. [2] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
               Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine 
               Learning in Python. Journal of Machine Learning Research, 12, 
               2825-2830.
        """
        # Load configuration from file
        config = self.load_config()
        
        # Validate that training data path is provided in the configuration
        if 'instruct_data' not in config['data'] or not config['data']['instruct_data']:
            raise ValueError("Training data path must be provided in the configuration.")
        
        # Load training data
        X, y = self.load_data(config['data']['instruct_data'])
        
        # Load evaluation data if provided and concatenate with training data
        if config['data']['eval_instruct_data']:
            X_eval, y_eval = self.load_data(config['data']['eval_instruct_data'])
            X = np.concatenate((X, X_eval), axis=0)
            y = np.concatenate((y, y_eval), axis=0)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Normalize the strategy names for consistency
        if str(strategy).lower() in ('tpogen', 'genetic', 'tpot'): 
            strategy = "TPOGEN"
        
        # Initialize the optimizer based on the strategy
        if strategy == 'TPOGEN':
            optimizer = TPOTClassifier(
                generations=generations, 
                population_size=population_size, 
                cv=cv, 
                scoring=scoring, 
                n_jobs=n_jobs, 
                random_state=random_state,
                verbosity=verbose
            )
        else:
            optimizer = Optimizer(
                estimators={type(estimator).__name__: estimator},
                param_grids={type(estimator).__name__: param_grid},
                strategy=strategy,  # strategy check will be handled by the Optimizer class
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs
            )

        # Fit the model using TPOT if strategy is TPOGEN
        if strategy == 'TPOGEN':
            optimizer.fit(X_train, y_train)
            print(f"Test Score: {optimizer.score(X_test, y_test)}")
            optimizer.export('tpot_pipeline.py')
            
            # If TPOTClassifier provides best estimator and other results
            try:
                results = {
                    "best_estimator_": optimizer.fitted_pipeline_,
                    "best_params_": optimizer.fitted_pipeline_.get_params(),
                    "best_score_": optimizer.score(X_test, y_test),
                    "cv_results_": optimizer.evaluated_individuals_
                }
                results = ModelSummary(
                    descriptor="AdvancedTuningManager", **results
                ).summary(results)
                print(results)
            except AttributeError:
                # If TPOTClassifier does not provide results, just export the pipeline
                print("TPOT optimization completed. Pipeline exported to 'tpot_pipeline.py'")
        else:
            # Fit the model using Optimizer for other strategies
            results = optimizer.fit(X_train, y_train)
            print(results)



