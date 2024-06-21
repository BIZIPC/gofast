# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import itertools
import numpy as np
import pandas as pd 
from .types import List, Optional, Union, DataFrame 

__all__=[
    "Bundle", 
    "KeyBox",
    "data2Box", 
    "BoxCategoricalEncoder", 
    "normalize_box_data", 
    "merge_boxes", 
    "deep_merge_keybox", 
    "save_to_box", 
    "filter_keybox", 
    "apply_to_keybox",
    "transform_keybox_attributes", 
    "keybox_to_dataframe",
    "dataframe_to_keybox",
    "list_dicts_to_keybox", 
    "flatten_keybox",
    "update_keybox_if",
    "DataToBox"
   ]

class Bundle:
    """
    A utility class for storing collections of results or data attributes.
    
    This class is designed to bundle various attributes and provide a 
    convenient way to access them using attribute-style access instead of 
    dictionary-style. It overrides the `__repr__` and `__str__`
    methods to offer informative and readable representations of its contents.
    
    Methods
    -------
    __repr__()
        Provides a basic representation of the object, indicating it contains 
        results.
    
    __str__()
        Returns a string representation of the object's contents in a nicely 
        formatted way, showing each attribute and its corresponding value.
    
    Examples
    --------
    >>> from gofast.tools.box import Bundle 
    >>> results = Bundle()
    >>> results.accuracy = 0.95
    >>> results.loss = 0.05
    >>> print(results)
    accuracy : 0.95
    loss     : 0.05
    
    >>> results
    <Bundle object containing results. Use print() to see contents>
    
    Note
    ----
    This class is intended to be a simple container for dynamically added attributes. 
    It provides a more intuitive and accessible way to handle grouped data without 
    the syntax of dictionaries, enhancing code readability and convenience.
    """
 
    def __init__(self, **kwargs):
        """
        Initialize a Bundle object with optional keyword arguments.

        Each keyword argument provided is set as an attribute of the Bundle object.

        Parameters
        ----------
        **kwargs : dict, optional
            Arbitrary keyword arguments which are set as attributes of the Bundle object.
        """
        self.__dict__.update(kwargs)
        
    def __repr__(self):
        """
        Returns a simple representation of the _Bundle object.
        
        This representation indicates that the object is a 'bunch' containing 
        results,suggesting to print the object to see its contents.
        
        Returns
        -------
        str
            A string indicating the object contains results.
        """
        return "<Bundle object containing results. Use print() to see contents>"

    def __str__(self):
        """
        Returns a detailed string representation of the _Bundle object's contents.
        
        This method lists all attributes of the _Bundle object in alphabetical order,
        along with their values in a formatted string. This provides a clear overview
        of all stored data for easy readability.
        
        Returns
        -------
        str
            A string representation of the _Bundle object's contents, including attribute
            names and their values, formatted in a tabular style.
        """
        if not self.__dict__:
            return "<empty Bundle>"
        keys = sorted(self.__dict__.keys())
        max_key_length = max(len(key) for key in keys)
        formatted_attrs = [
            f"{key:{max_key_length}} : {self.summarize(self.__dict__[key])}" 
            for key in keys]
        return "\n".join(formatted_attrs)
    
    def summarize(self, attr):
        """
        Provides a concise summary of iterable attributes within a Bundle object, 
        offering key statistical or structural information tailored to the 
        attribute's type.
    
        This method intelligently adapts to various iterable types including 
        lists, tuples, sets, NumPy arrays, pandas Series, and DataFrames.
        For numeric data, it calculates and presents statistics like minimum,
        maximum, mean values, and length. For NumPy  arrays, it also includes 
        shape and dtype. Pandas Series and DataFrames summaries account for 
        data type and, when relevant, provide a statistical overview for numeric
        data.
    
        Parameters
        ----------
        attr : iterable
            The iterable attribute of the Bundle object to be summarized. 
            Acceptable types are list, tuple, set, np.ndarray, pd.Series, 
            and pd.DataFrame.
    
        Returns
        -------
        str
            A string representation of the summary, formatted according to 
            the attribute's type and contents. This includes a concise summary
            of statistical information for numeric data and relevant structural
            information for all types.
    
        Notes
        -----
        - The method aims to provide quick insights into the data's structure
          and distribution without needing detailed exploratory data analysis.
        - Numeric summaries are rounded to 4 decimal places for clarity.
        - For pandas DataFrames with uniform data types, the dtype is mentioned
          directly; for mixed types, the unique dtypes are listed.
        
        Examples
        --------
        >>> from gofast.tools.box import Bundle
        >>> bundle = Bundle()
        >>> bundle.numeric_list = [1, 2, 3, 4, 5]
        >>> print(bundle.summarize(bundle.numeric_list))
        list (min=1, max=5, mean=3, len=5)
        
        >>> import numpy as np
        >>> bundle.numeric_array = np.array([1, 2, 3, 4, 5])
        >>> print(bundle.summarize(bundle.numeric_array))
        Array (min=1, max=5, mean=3, shape=5, dtype=int64)
        
        >>> import pandas as pd
        >>> bundle.data_frame = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> print(bundle.summarize(bundle.data_frame))
        DataFrame (min: {'A': 1, 'B': 3}, max: {'A': 2, 'B': 4},
                   mean: {'A': 1.5, 'B': 3.5}, rows=2, cols=2, dtypes=[int64])
        """

        def format_numeric(data, label=''):
            minval, maxval = round(min(data), 4), round(max(data), 4)
            meanval = round(np.mean(data), 4)
            return f"{label} (min={minval}, max={maxval}, mean={meanval}, len={len(data)})"
        
        def format_array(arr):
            shape = 'x'.join(map(str, arr.shape))
            stats = "min={}, max={}, mean={}".format(
                round(arr.min(), 4), round(arr.max(), 4), round(arr.mean(), 4))
            return f"Array ({stats}, shape={shape}, dtype={arr.dtype})"
        
        if isinstance(attr, (list, tuple, set)) and all(
                isinstance(item, (int, float)) for item in attr):
            return format_numeric(attr, type(attr).__name__)
        elif isinstance(attr, np.ndarray):
            if attr.dtype.kind in 'if':
                return format_array(attr)
            return f"Array (shape={attr.shape}, dtype={attr.dtype})"
        elif isinstance(attr, pd.Series):
            if attr.dtype.kind in 'if':
                return f"Series ({format_numeric(attr, '')}, dtype={attr.dtype})"
            return f"Series (len={attr.size}, dtype={attr.dtype})"
        elif isinstance(attr, pd.DataFrame):
            if 'object' not in attr.dtypes.values:
                numeric_cols = attr.select_dtypes(include=['number']).columns.tolist()
                aggregated_stats = attr[numeric_cols].agg(
                    ['min', 'max', 'mean']).round(4).to_dict()
                stats = ", ".join([f"{k}: {v}" for k, v in aggregated_stats.items()])
                return (f"DataFrame ({stats}, rows={attr.shape[0]},"
                        f" cols={attr.shape[1]}, dtypes={attr.dtypes.unique()})")
            return (f"DataFrame (rows={attr.shape[0]}, cols={attr.shape[1]},"
                    f" dtypes={attr.dtypes.unique()})")
    
        return str(attr)

    def __delattr__(self, name):
        """
        Delete an attribute from the Bundle object.

        Parameters
        ----------
        name : str
            The name of the attribute to delete.

        Raises
        ------
        AttributeError
            If the specified attribute does not exist.
        """
        if name in self.__dict__:
            del self.__dict__[name]
        else:
            raise AttributeError(f"'Bundle' object has no attribute '{name}'")

    def __contains__(self, name):
        """
        Check if an attribute exists in the Bundle object.

        Parameters
        ----------
        name : str
            The name of the attribute to check for existence.

        Returns
        -------
        bool
            True if the attribute exists, False otherwise.
        """
        return name in self.__dict__

    def __iter__(self):
        """
        Return an iterator over the Bundle object's attribute names and values.

        Yields
        ------
        tuple
            Pairs of attribute names and their corresponding values.
        """
        for item in self.__dict__.items():
            yield item

    def __len__(self):
        """
        Return the number of attributes stored in the Bundle object.

        Returns
        -------
        int
            The number of attributes.
        """
        return len(self.__dict__)

    def __getattr__(self, name):
        """
        Get an attribute's value, raising an AttributeError if it doesn't exist.

        Parameters
        ----------
        name : str
            The name of the attribute to access.

        Raises
        ------
        AttributeError
            If the specified attribute does not exist.
        """
        raise AttributeError(f"'Bundle' object has no attribute '{name}'")

    def __eq__(self, other):
        """
        Compare this Bundle object to another for equality.

        Parameters
        ----------
        other : Bundle
            Another Bundle object to compare against.

        Returns
        -------
        bool
            True if the other object is a Bundle with the same attributes and 
            values, False otherwise.
        """
        return isinstance(other, Bundle) and self.__dict__ == other.__dict__

class KeyBox(dict):
    """
    `KeyBox` is a flexible container that extends the standard dictionary to 
    support attribute-like access in addition to traditional key-based access. 
    This enhancement facilitates a more intuitive interaction with data structures, 
    particularly beneficial for configuration management and handling data with 
    dynamic schemas.

    Unlike a regular dictionary, `KeyBox` enables you to access the stored values 
    using dot notation, which can make your code cleaner and more readable. 
    However, care must be taken to avoid naming conflicts with existing methods 
    and attributes of the dictionary class.

    Examples
    --------
    Creating a `KeyBox` instance and accessing its items:
    
    >>> from gofast.tools.box import KeyBox
    >>> keybox = KeyBox(pkg='gofast', objective='give water', version='0.1.dev')
    >>> keybox['pkg']
    'gofast'
    >>> keybox.pkg
    'gofast'
    >>> keybox.objective
    'give water'
    >>> keybox.version
    '0.1.dev'
        
    >>> keybox = KeyBox(project='Hydra', version='1.0', contributors=4)
    >>> print(keybox.project)
    Hydra
    >>> print(keybox['version'])
    1.0

    Dynamically adding new items and accessing them:

    >>> keybox.status = 'active'
    >>> print(keybox['status'])
    active

    Iterating over items:

    >>> for key, value in keybox.items():
    ...     print(f"{key}: {value}")
    project: Hydra
    version: 1.0
    contributors: 4
    status: active

    Notes
    -----
    - While `KeyBox` enhances accessibility to dictionary items, users must ensure 
      that keys do not overlap with the names of dictionary methods (e.g., `items`, 
      `keys`, `update`). Such overlaps could obscure the method access, leading 
      to unexpected results or errors.
    - The implementation leverages Python's magic methods to seamlessly integrate 
      attribute-like access while retaining all functionalities of a standard 
      dictionary.
    - As with dictionaries, the `KeyBox` keys are case-sensitive and must be unique.
    - `KeyBox` objects can be nested, allowing for complex data structures with 
      convenient access patterns.

    Caution is advised when using reserved words or method names as keys, as 
    it may necessitate the use of bracket notation for access to avoid conflicts 
    or unexpected behavior.
    """
    def __init__(self, **kwargs):
        """
        Initializes a KeyBox object with optional keyword arguments.
        
        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments which are set as the initial 
            items of the dictionary.
        """
        super().__init__(**kwargs)

    def __getattr__(self, key):
        """
        Allows attribute-like access to dictionary items.
        
        Parameters
        ----------
        key : str
            The attribute name corresponding to the dictionary key.
        
        Returns
        -------
        The value associated with 'key' in the dictionary.
        
        Raises
        ------
        AttributeError
            If the key is not found in the dictionary.
        """
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'KeyBox' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        """
        Allows setting dictionary items as attributes.
        
        Parameters
        ----------
        key : str
            The attribute name to be added or updated in the dictionary.
        value : any
            The value to be associated with 'key'.
        """
        self[key] = value

    def __setstate__(self, state):
        """
        Overrides __setstate__ to ensure the object can be unpickled correctly.
        
        This method is a no-op, effectively ignoring the pickled __dict__, which is
        necessary because `KeyBox` objects use the dictionary itself for item storage.
        """
        pass

    def __dir__(self):
        """
        Ensures that autocompletion works in interactive environments.
        
        Returns
        -------
        list
            A list of keys in the dictionary, which are exposed as attributes.
        """
        return super().__dir__() + list(self.keys()) # self.keys()

    def to_frame(self, error='raise', force_df=False):
        """
        Converts the KeyBox object into a pandas DataFrame.
    
        This method intelligently handles the conversion of KeyBox contents to
        a DataFrame, accommodating various data structures such as lists, dictionaries,
        and Bundle objects within the KeyBox. It ensures consistency across the
        resulting DataFrame by standardizing lengths and filling missing values with
        `pd.NA` where necessary.
    
        Parameters
        ----------
        error : str, optional
            Specifies how to handle errors encountered during conversion. If set
            to 'raise', an error will be thrown for invalid data structures. If
            'ignore', an empty DataFrame will be returned instead. Default is 'raise'.
        force_df : bool, optional
            If `True`, forces the output to always be a DataFrame, even if the
            KeyBox contains a single attribute. Default is `False`, where single
            attribute KeyBoxes may return a pandas Series instead.
    
        Returns
        -------
        pd.DataFrame or pd.Series
            A DataFrame representing the KeyBox's contents. If `force_df` is `False`
            and the KeyBox contains a single attribute, a pandas Series is returned
            instead of a DataFrame.
    
        Raises
        ------
        ValueError
            Raised if `error` is set to 'raise' and the KeyBox's contents cannot
            be converted into a consistent DataFrame structure due to incompatible
            data types or structures.
    
        Examples
        --------
        Single attribute KeyBox, returning a Series by default:
            
        >>> from gofast.tools.box import Bundle, KeyBox
        >>> box = KeyBox(color=['red', 'blue', 'green'])
        >>> box.to_frame()
        0      red
        1     blue
        2    green
        Name: color, dtype: object
    
        Multiple attributes with consistent length, returning a DataFrame:
    
        >>> box = KeyBox(color=['red', 'blue', 'green'], size=['S', 'M', 'L'])
        >>> box.to_frame()
           color size
        0    red    S
        1   blue    M
        2  green    L
    
        Handling of inconsistent data lengths by filling with `pd.NA`:
    
        >>> box = KeyBox(color=['red', 'blue'], size=['S', 'M', 'L'])
        >>> box.to_frame()
           color size
        0    red    S
        1   blue    M
        2   <NA>    L
        
        >>> box = KeyBox(a=Bundle(name='Alice', age=30),
        ...                b=Bundle(name='Bob', age=25))
        >>> df = box.to_frame()
        >>> print(df)
           name  age
        a  Alice   30
        b    Bob   25
    
        >>> # Example with inconsistent data and error handling
        >>> box = KeyBox(a=Bundle(name='Alice', age=30),
        ...                b={'name': 'Bob'})
        >>> df = box.to_frame()
        >>> print(df)
            age   name
        a    30  Alice
        b  <NA>    Bob
    
        Notes
        -----
        The `to_frame` method is versatile, capable of handling various data structures
        within a KeyBox. It prioritizes maintaining the integrity and consistency of
        the dataset during the conversion process. Users should be aware that modifying
        the error handling behavior can significantly alter the method's response to
        incompatible data structures.
        """
        def ensure_list(item):
            """
            Ensure the input item is a list. If the input is not a list, it is
            encapsulated into a list. If the input is a list, it is returned 
            unchanged.
        
            Parameters:
            - item: The input item to check and possibly convert to a list.
        
            Returns:
            - A list containing the input item if it was not already a list, or the
              input item unchanged if it was already a list.
            """
            if not isinstance(item, list):
                return [item]
            return item
        
        # Return an empty DataFrame for an empty KeyBox
        if not self:
            return pd.DataFrame()
        
        # Wrap non-iterable items (except dict and Bundle) in lists
        items_list_wrapped={} 
        for key, value in self.items():
            if not isinstance(value, (list, dict, Bundle)):
                self[key] = ensure_list(value)
                items_list_wrapped[key]= self[key] # append wrapped keys
                
        # Function to handle lists within KeyBox
        def handle_lists():
            lengths = [len(value) for value in self.values()]
            if len(set(lengths)) > 1:  # Check for unequal lengths
                # Equalize lengths by filling with pd.NA
                for key, value in self.items():
                    if len(value) < max(lengths):
                        self[key] += [pd.NA] * (max(lengths) - len(value))
            return pd.DataFrame.from_dict(self)
    
        # Function to handle dict or Bundle within KeyBox
        def handle_dicts():
            # Ensure all values are dict or Bundle, convert to DataFrame
           if any(not isinstance(item, (dict, Bundle)) for item in self.values()):
               error_msg = "All items in the KeyBox must be dictionaries or Bundle objects."
               if error == 'raise':
                   raise ValueError(error_msg)
               return pd.DataFrame()  # Return an empty DataFrame on error='ignore'
           
           # Transform each item in the KeyBox into a dictionary format, 
           # accommodating both dict and Bundle types.
           data = [dict(item) if isinstance(item, Bundle) else item for item in self.values()]
       
           # Compile all keys from the dictionaries to establish a consistent 
           # column structure for the DataFrame.
           all_keys = set().union(*[d.keys() for d in data])
       
           # Create a comprehensive list of dictionaries, filling in any 
           # missing keys with pd.NA, to ensure data integrity.
           prepared_data = [{key: d.get(key, pd.NA) for key in all_keys} for d in data]
       
           try:
               # Construct the DataFrame using the prepared data. Attempt to 
               # use the KeyBox's keys as DataFrame index.
               df = pd.DataFrame(prepared_data, index=list(self.keys()))
           except Exception:
               # Fallback: Create the DataFrame without specifying the index, 
               # ensuring the operation's robustness.
               df = pd.DataFrame(prepared_data)
       
           return df

        # Determine data type and call the appropriate handler
        if all(isinstance(value, list) for value in self.values()):
            df = handle_lists()
        else:
            # Create DataFrame
            try:
                df = handle_dicts()
            except Exception as e:
                if error == 'raise':
                    raise ValueError(f"Failed to create DataFrame from KeyBox: {e}")
                return pd.DataFrame()
       
        # Check for single column DataFrame and 'force_df' flag
        if len(df.columns) == 1 and not force_df:
            return df.iloc[:, 0] if not force_df else df
    
        # fallback to original item type. 
        if items_list_wrapped: 
            for key, value in items_list_wrapped.items(): 
                self[key]= value[0] # remove item from list  
                
        return df
    
    def __repr__(self):
        """
        Provides an enhanced representation of the KeyBox object, displaying
        its contents in a more readable and nicely formatted manner.
        
        Returns
        -------
        str
            A string representation of the KeyBox object, showcasing its
            keys and values in a well-organized layout.
        """
        if not self:
            return f"{self.__class__.__name__}()"

        max_key_length = max(len(str(key)) for key in self.keys()) + 1
        items_repr = []

        for key, value in self.items():
            key_repr = f"{str(key):<{max_key_length}}"
            value_repr = repr(value).replace('\n', '\n' + ' ' * (max_key_length + 4))
            items_repr.append(f"{key_repr} : {value_repr}")

        items_str = "\n    ".join(items_repr)
        return f"{self.__class__.__name__}(\n    {items_str}\n)"
    
def DataToBox(
    data: Union[DataFrame, dict, List], 
    entity_name: Optional[str] = None, 
    use_column_as_name: bool = False, 
    keep_name_column: bool = True, 
    columns: Optional[List[str]] = None):
    """
    Transforms each row of a DataFrame or a similar iterable structure into 
    :class:`KeyBox` objects.
    
    Parameters
    ----------
    data : DataFrame, dict, or List
        The data to transform. If not a DataFrame, an attempt is made to 
        convert it into one.
    
    entity_name : str, optional
        Base name for the entities in the BoxSpace. If `use_column_as_name` is 
        True, this refersto a column name whose values are used as entity names.
    
    use_column_as_name : bool, default False
        If True, `entity_name` must be a column name in `data`, and its values 
        are used as names
        for the entities.
    
    keep_name_column : bool, default True
        Whether to keep the column used for naming entities in the data.
    
    columns : list of str, optional
        Specifies column names when `data` is a list or an array-like structure. 
        Ignored if `data` is already a DataFrame.
    
    Returns
    -------
    BoxSpace
        A BoxSpace object containing named entities corresponding to each 
        row of the input `data`.
        
    Raises
    ------
    ValueError
        If `use_column_as_name` is True but `entity_name` is not a 
        column in `data`.
    
    TypeError
        If the input `data` cannot be converted into a DataFrame.
    
    Examples
    --------
    >>> from gofast.tools.box import DataToBox 
    >>> DataToBox([2, 3, 4], entity_name='borehole')
    >>> DataToBox({"x": [2, 3, 4], "y": [8, 7, 5]}, entity_name='borehole')
    >>> DataToBox([2, 3, 4], entity_name='borehole', columns=['id'])
    >>> DataToBox({"x": [2, 3, 4], "y": [8, 7, 5], "code": ['h2', 'h7', 'h12']},
                            entity_name='code', use_column_as_name=True)
    """
    if not isinstance(data, pd.DataFrame):
        if columns is None and isinstance(data, dict):
            columns = list(data.keys())
        data = pd.DataFrame(data, columns=columns)

    if entity_name is not None and use_column_as_name:
        if entity_name not in data.columns:
            raise ValueError(f"Column '{entity_name}' must exist in the data.")
        names = data[entity_name].astype(str)
        if not keep_name_column:
            data = data.drop(columns=[entity_name])
    else:
        if entity_name is None:
            entity_name = 'object'
        names = [f"{entity_name}{i}" for i in range(len(data))]

    box_space_objects = {
        name: KeyBox(**data.iloc[i].to_dict()) for i, name in enumerate(names)}

    return KeyBox(**box_space_objects)

def data2Box(
    data, /,  
    name: str = None, 
    use_colname: bool =False, 
    keep_col_data: bool =True, 
    columns: List [str] =None 
    ): 
    """ Transform each data rows as KeyBox object. 
    
    Parameters 
    -----------
    data: DataFrame 
      Data to transform as an object 
      
    columns: list of str, 
      List of str item used to construct the dataframe if tuple or list 
      is passed. 
      
    name: str, optional 
       The object name. When string argument is given, the index value of 
       the data is is used to prefix the name data unless the `use_column_name`
       is set to ``True``. 
       
    use_colname: bool, default=False 
       If ``True`` the name must be in columns. Otherwise an error raises. 
       However, when ``use_colname=true``, It is recommended to make sure 
       whether each item in column data is distinct i.e. is unique, otherwise, 
       some data will be erased. The number of object should be less than 
       the data size along rows axis. 
       
    keep_col_data: bool, default=True 
      Keep in the data the column that is used to construct the object name.
      Otherwise, column data whom object created from column name should 
      be dropped. 
      
    Return
    --------
    Object: :class:`.BoxSpace`, n_objects = data.size 
       Object that composed of many other objects where the number is equals 
       to data size. 
       
    Examples
    --------- 
    >>> from gofast.tools.box import data2Box 
    >>> o = data2Box ([2, 3, 4], name = 'borehole')
    >>> o.borehole0
    {'0': 2}
    >>> o = data2Box ({"x": [2, 3, 4], "y":[8, 7, 5]}, name = 'borehole')
    >>> o.borehole0.y
    8
    >>> from gofast.tools.box import data2Box 
    >>> o = data2Box ([2, 3, 4], name = 'borehole', columns ='id') 
    >>> o.borehole0.id
    2
    >>> o = data2Box ({"x": [2, 3, 4], "y":[8, 7, 5], 
                       "code": ['h2', 'h7', 'h12'] }, name = 'borehole')
    >>> o.borehole1.code
    'h7'
    >>> o = data2Box ({"x": [2, 3, 4], "y":[8, 7, 5], "code": ['h2', 'h7', 'h12'] }, 
                      name = 'code', use_colname= True )
    >>> o.h7.code
    'h7'
    >>> o = data2Box ({"x": [2, 3, 4], "y":[8, 7, 5], "code": ['h2', 'h7', 'h12'] 
                       }, name = 'code', use_colname= True, keep_col_data= False  )
    >>> o.h7.code # code attribute does no longer exist 
    AttributeError: code
    """
    from .validator import _is_numeric_dtype 
    from .coreutils import is_iterable 
    
    if columns is not None: 
        columns = is_iterable (
            columns, exclude_string= True , transform =True  )
   
    if ( 
            not hasattr ( data , 'columns') 
            or hasattr ( data, '__iter__')
            ): 
        data = pd.DataFrame ( data, columns = columns )
        
    if not hasattr(data, '__array__'): 
            raise TypeError (
                f"Object accepts only DataFrame. Got {type(data).__name__}")

    if columns is not None: 
        # rename columns if given 
        data = pd.DataFrame(np.array( data), columns = columns )
        
    if name is not None: 
        # Name must be exists in the dataframe. 
        if use_colname:  
            if name not in data.columns:  
                raise ValueError (
                    f"Name {name!r} must exist in the data columns.")
            
            name =  data [name] if keep_col_data else data.pop ( name )
            
    # make name column if not series 
    if not hasattr ( name, 'name'):
        # check whether index is numeric then prefix with index 
        index = data.index 
        if _is_numeric_dtype(index, to_array= True ): 
            index = index.astype (str)
            if name is None:
                name ='obj'
        
        name = list(map(''.join, itertools.zip_longest(
            [name  for i in range ( len(index ))], index)))
        
    # for consistency # reconvert name to str 
    name = np.array (name ).astype ( str )
    
    obj = dict() 
    for i in range ( len(data)): 
        v = KeyBox( **dict ( zip ( data.columns.astype (str),
                                    data.iloc [i].values )))
        obj [ name[i]] = v 
        
    return KeyBox( **obj )

class BoxCategoricalEncoder:
    """
    Encodes categorical features in a dataset into numerical formats. This encoder 
    supports both label encoding and one-hot encoding, allowing for the conversion 
    of categorical variables into a format suitable for machine learning models.
    
    The class is compatible with pandas DataFrames and KeyBox objects, making it 
    versatile for different data structures. It can be particularly useful in 
    preprocessing steps of a data pipeline where categorical data needs to be 
    transformed into numerical representations.

    Attributes
    ----------
    encoding_type : str
        Specifies the type of encoding to apply. Options include:
        - 'label': Maps each category to a unique integer. Useful for ordinal 
          data or as a general encoding method.
        - 'one-hot': Encodes categories as binary vectors with only one high (1)
          indicating the presence of a feature. This method is useful for nominal
          data where no ordinal relationship exists.

    encoders : dict
        Stores the encoders for each categorical column after fitting. This is 
        used for transforming data and inverse transforming data back to original 
        categories.

    Methods
    -------
    fit(data)
        Prepares the encoder based on the data, identifying categorical columns 
        and initializing encoders according to the `encoding_type`.

    transform(data=None)
        Transforms the categorical features of the data using the fitted encoders.
        Can be used directly on new data if the encoder has been previously fitted.

    fit_transform(data)
        A convenience method that first fits the encoder with the data and then 
        transforms the data in a single step.

    inverse_transform(data)
        Converts numerical features back to their original categorical values based 
        on the fitted encoders. Note that inverse transformation might not be 
        meaningful for one-hot encoded data.

    Examples
    --------
    Label encoding example:
    
    >>> from gofast.tools.box import KeyBox, BoxCategoricalEncoder
    >>> data = KeyBox(color=['red', 'blue', 'green'], size=['S', 'M', 'L'])
    >>> encoder = BoxCategoricalEncoder(encoding_type='label')
    >>> transformed_data = encoder.fit_transform(data)
    >>> print(transformed_data.color)
    [0, 1, 2]

    One-hot encoding example:
    
    >>> data = pd.DataFrame({'color': ['red', 'blue', 'green'], 'size': ['S', 'M', 'L']})
    >>> encoder = BoxCategoricalEncoder(encoding_type='one-hot')
    >>> transformed_data = encoder.fit_transform(data)
    >>> print(transformed_data)
       color_blue  color_green  color_red  size_L  size_M  size_S
    0           0            0          1       0       0       1
    1           1            0          0       0       1       0
    2           0            1          0       1       0       0

    Inverse transform example (label encoding):
    
    >>> inverse_data = encoder.inverse_transform(transformed_data)
    >>> print(inverse_data.color)
    ['red', 'blue', 'green']

    Note: The inverse_transform method for 'one-hot' encoding is not implemented, 
    as reversing one-hot encoding to exact original categories can be ambiguous.
    """

    def __init__(self, encoding_type='label'):
        self.encoding_type = encoding_type
        self.encoders = {}


    def fit(self, data):
        """
        Fits the encoder to the data, preparing it for transforming categorical
        features according to the specified `encoding_type`. This method identifies
        the categorical columns and initializes encoders for them.
    
        Parameters
        ----------
        data : pd.DataFrame, dict, or KeyBox
            The dataset containing categorical features to encode. It can be 
            a pandas DataFrame, a dictionary (which will be converted to a DataFrame),
            or a KeyBox object.
    
        Returns
        -------
        self : BoxCategoricalEncoder
            The fitted encoder instance ready for transforming data.
    
        Raises
        ------
        ValueError
            If the input `data` type is not supported or if no categorical columns
            are identified in the `data`.
    
        Example
        -------
        >>> from gofast.tools.box import KeyBox, BoxCategoricalEncoder
        >>> data = KeyBox(a='red', b='blue', c='green')
        >>> encoder = BoxCategoricalEncoder(encoding_type='label')
        >>> encoder.fit(data)
        <BoxCategoricalEncoder fitted with label encoding for ['a', 'b', 'c']>
        """
        # Ensure data is a DataFrame for processing
        if isinstance(data, KeyBox):
            data = data.to_frame()
        elif isinstance(data, dict):
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            raise ValueError(
                "Data must be a pandas DataFrame, a KeyBox object, or a dictionary.")
    
        # Identify categorical columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) == 0:
            raise ValueError("No categorical columns identified in the data.")
    
        # Initialize encoders for each categorical column
        self.encoders = {}
        if self.encoding_type == 'label':
            from sklearn.preprocessing import LabelEncoder
            for col in categorical_cols:
                le = LabelEncoder()
                le.fit(data[col])
                self.encoders[col] = le
        elif self.encoding_type == 'one-hot':
            # For one-hot encoding, we don't need to fit individual encoders
            pass
        else:
            raise ValueError(f"Unsupported encoding type '{self.encoding_type}'.")
    
        return self

    def fit_transform(self, data):
        """
        Fits the encoder to the data and transforms the categorical features. This
        method combines the fitting and transforming steps for convenience.
    
        Parameters
        ----------
        data : pd.DataFrame, dict, or KeyBox
            The dataset containing categorical features to encode. It can be a
            pandas DataFrame, a dictionary of lists (which will be converted to DataFrame), 
            or a KeyBox object.
    
        Returns
        -------
        pd.DataFrame or KeyBox
            A new dataset with categorical features encoded according to the 
            `encoding_type`. The type of the returned object matches the type 
            of the input `data`.
    
        Raises
        ------
        ValueError
            If the input `data` type is not supported or if `encoding_type` 
            is not recognized.
    
        Example
        -------
        >>> from gofast.tools.box import KeyBox, BoxCategoricalEncoder
        >>> data = KeyBox(a='red', b='blue', c='red')
        >>> encoder = BoxCategoricalEncoder(encoding_type='label')
        >>> encoded_data = encoder.fit_transform(data)
        >>> print(encoded_data)
        {'a': 0, 'b': 1, 'c': 0}
        """
        self.fit(data)
        return self.transform()
    
    def inverse_transform(self, data):
        """
        Converts numerical features back to their original categorical values. 
        
        This method is primarily applicable for label encoded data.
    
        Parameters
        ----------
        data : pd.DataFrame or KeyBox
            The dataset with numerical features previously encoded by this 
            encoder, which are to be converted back to their original 
            categorical format.
    
        Returns
        -------
        pd.DataFrame or KeyBox
            The dataset with encoded features converted back to their original
            categorical values. The type of the returned object matches the 
            type of the input `data`.
    
        Raises
        ------
        NotImplementedError
            If attempting to inverse-transform data encoded with 'one-hot', as this
            operation is not straightforward with the current implementation.
    
        Example
        -------
        >>> from gofast.tools.box import KeyBox, BoxCategoricalEncoder
        >>> data = KeyBox(a=0, b=1, c=0)
        >>> encoder = BoxCategoricalEncoder(encoding_type='label')
        >>> original_data = encoder.inverse_transform(data)
        >>> print(original_data)
        {'a': 'red', 'b': 'blue', 'c': 'red'}
        """
        if not isinstance(data, (pd.DataFrame, KeyBox)):
            raise ValueError("Data must be a pandas DataFrame or a KeyBox object.")
        original_type= type (data)
        if original_type==KeyBox: 
            data = data.to_frame(error='ignore')
            
        if self.encoding_type == 'label':
            for col, le in self.encoders.items():
                if col in data.columns:
                    data[col] = le.inverse_transform(data[col])
        else:
            raise NotImplementedError(
                "Inverse transformation for 'one-hot' encoding is not implemented.")
        if original_type==KeyBox: 
            return KeyBox ( **data.to_dict (orient='series'))
        return data

def normalize_box_data(data, method='z-score'):
    """
    Normalizes the numerical features in a dataset using specified normalization
    techniques.
    
    This utility function is versatile and can handle both pandas DataFrame 
    and KeyBox objects,
    making it suitable for a wide range of data preprocessing tasks in data
    analysis and machine learning.
    
    Supported normalization methods include 'z-score' for standardization and
    'min-max' for scaling data to a fixed range [0, 1]. Z-score normalization 
    transforms the data to have a mean of 0 and a standard deviation of 1, 
    while min-max scaling adjusts the scale without distorting the differences
    in the ranges of values.
    
    Parameters
    ----------
    data : pd.DataFrame or KeyBox
        The input dataset containing numerical features to be normalized. 
        This can be a pandas DataFrame or a KeyBox object, providing 
        flexibility in data structures for normalization.
    method : str, optional
        The normalization method to apply to the numerical features in `data`.
        Supported values are:
        - 'z-score': Applies Z-score normalization.
        - 'min-max': Applies min-max scaling.
        Default value is 'z-score'.
    
    Returns
    -------
    pd.DataFrame or KeyBox
        The resulting dataset after applying the specified normalization 
        technique. The return type matches the input type (`data`), ensuring 
        consistency and ease of integration into data pipelines.
    
    Examples
    --------
    Using a pandas DataFrame:

    >>> import pandas as pd
    >>> from gofast.tools.box import normalize_box_data
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> normalized_df = normalize_box_data(df, method='min-max')
    >>> print(normalized_df)
           A         B
    0  0.0  0.0
    1  0.5  0.5
    2  1.0  1.0

    Using a KeyBox object:

    >>> from gofast.tools.box import KeyBox, normalize_box_data
    >>> box = KeyBox(item1={'A': 1, 'B': 4}, item2={'A': 2, 'B': 5}, item3={'A': 3, 'B': 6})
    >>> normalized_box = normalize_box_data(box, method='z-score')
    >>> print(normalized_box.item1)
    {'A': -1.224744871391589, 'B': -1.224744871391589}
    
    Notes
    -----
    - When applying 'min-max' scaling, the function uses the range [0, 1]. 
      This behavior can be customized by modifying the function to accept range
      parameters if needed.
    - For KeyBox objects, ensure that all items are dictionary-like with 
      numerical values for the normalization to work properly.
    """

    if method not in ['z-score', 'min-max']:
        raise ValueError(f"Unsupported normalization method: {method}")

    original_type = type(data)

    if original_type not in [pd.DataFrame, KeyBox]:
        raise TypeError("Data must be either a pd.DataFrame or KeyBox object.")

    # Convert KeyBox to DataFrame if necessary
    if original_type == KeyBox:
        data = data.to_frame(error='ignore')

    numeric_cols = data.select_dtypes(include=np.number).columns

    if not numeric_cols.any():
        raise ValueError("No numeric columns found in data for normalization.")

    try:
        if method == 'z-score':
            from scipy.stats import zscore
            data[numeric_cols] = data[numeric_cols].apply(zscore)
        elif method == 'min-max':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    except Exception as e:
        raise RuntimeError(f"Normalization failed: {e}")

    # Convert back to original data type if necessary
    if original_type == KeyBox:
        return KeyBox(**data.to_dict(orient='series'))
    
    return data

def merge_boxes(*boxes):
    """
    Merges multiple KeyBox objects into a single KeyBox object, ensuring that
    all inputs are KeyBox instances and handling duplicate keys gracefully.

    Parameters
    ----------
    *boxes : KeyBox
        An arbitrary number of KeyBox objects to merge.

    Returns
    -------
    KeyBox
        A new KeyBox object containing the merged contents of all input KeyBoxs.

    Raises
    ------
    TypeError
        If any of the arguments are not instances of KeyBox.

    Warning
    -------
    UserWarning
        If duplicate keys are found, the function keeps the value from the 
        first occurrence and issues a warning about the duplicate keys ignored.

    Example
    -------
    >>> from gofast.tools.box import merge_boxes, KeyBox
    >>> box1 = KeyBox(a=1, b=2)
    >>> box2 = KeyBox(c=3, d=4, a=5)  # Note the duplicate key 'a'
    >>> merged_box = merge_boxes(box1, box2)
    UserWarning: Duplicate keys were found and ignored; kept the first occurrence: ['a']
    >>> print(merged_box.a, merged_box.c)
    1 3
    """
    if not all(isinstance(box, KeyBox) for box in boxes):
        raise TypeError("All arguments must be instances of KeyBox.")

    merged_dict = {}
    duplicates = []
    for box in boxes:
        for key, value in box.items():
            if key not in merged_dict:
                merged_dict[key] = value
            else:
                duplicates.append(key)

    if duplicates:
        import warnings
        duplicates = list(set(duplicates))  # Removing duplicates from the list of duplicates
        warnings.warn(
            f"Duplicate keys were found and ignored; kept the first occurrence: {duplicates}",
            UserWarning
        )

    return KeyBox(**merged_dict)
        
    
def deep_merge_keybox(*boxes):
    """
    Deeply merges multiple KeyBox objects into one.

    Parameters
    ----------
    *boxes : KeyBox
        Variable number of KeyBox objects to merge.

    Returns
    -------
    KeyBox
        A new KeyBox object containing the deeply merged contents of all
        input KeyBoxs.

    Example
    -------
    >>> from gofast.tools.box import deep_merge_keybox, KeyBox
    >>> box1 = KeyBox(a=1, b=KeyBox(x=2, y=3))
    >>> box2 = KeyBox(b=KeyBox(y=4, z=5), c=6)
    >>> merged_box = deep_merge_keybox(box1, box2)
    >>> print(merged_box.b.y, merged_box.b.z, merged_box.c)
    4 5 6
    """
    if not all(isinstance(box, KeyBox) for box in boxes):
        raise TypeError("All arguments must be instances of KeyBox.")
        
    def merge(dict1, dict2):
        for k, v in dict2.items():
            if k in dict1 and isinstance(dict1[k], KeyBox) and isinstance(v, KeyBox):
                merge(dict1[k], v)
            else:
                dict1[k] = v

    merged = KeyBox()
    for box in boxes:
        merge(merged, box)
    
    return merged

def save_to_box(data, key_column, value_columns=None):
    """
    Saves data from a DataFrame to a KeyBox, using one column's 
    values as keys.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing data to be saved in a KeyBox. 
        Must be an instance of pd.DataFrame.
    key_column : str
        The column in `data` whose values will be used as keys in the KeyBox. 
        Must exist in `data`.
    value_columns : list of str, optional
        The columns in `data` whose values will be saved under each key in 
        the KeyBox.
        If None, all other columns except `key_column` are included.

    Returns
    -------
    KeyBox
        A KeyBox object containing data indexed by `key_column` values.

    Raises
    ------
    TypeError
        If `data` is not an instance of pd.DataFrame.
    KeyError
        If `key_column` is not found in `data` or any `value_columns` do not 
        exist in `data`.

    Example
    -------
    >>> import pandas as pd 
    >>> from gofast.tools.box import save_to_box
    >>> df = pd.DataFrame({'id': ['item1', 'item2'], 'value1': [10, 20], 'value2': [30, 40]})
    >>> box = save_to_box(df, 'id')
    >>> print(box.item1.value1, box.item2.value2)
    10 40
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("The 'data' parameter must be an instance of pd.DataFrame.")

    if key_column not in data.columns:
        raise KeyError(f"The key column '{key_column}' is not found in the DataFrame.")
    
    if value_columns is not None:
        missing_cols = [col for col in value_columns if col not in data.columns]
        if missing_cols:
            raise KeyError("The following value columns are not found in"
                           f" the DataFrame: {missing_cols}")
    else:
        value_columns = [col for col in data.columns if col != key_column]
    
    try:
        box_data = {
            getattr(row, key_column): Bundle(**row[value_columns].to_dict())
            for _, row in data.iterrows()
        }
    except Exception as e:
        raise ValueError(f"An error occurred while processing the DataFrame: {e}")
    
    return KeyBox(**box_data)


def filter_keybox(box, condition):
    """
    Filters attributes in a KeyBox object based on a given condition.

    Parameters
    ----------
    box : KeyBox
        The KeyBox object to be filtered. Must be an instance of KeyBox.
    condition : callable
        A function that takes a key-value pair (attribute name and value) and 
        returns True if the attribute should be included in the filtered 
        KeyBox. Must be a callable object.

    Returns
    -------
    KeyBox
        A new KeyBox object containing only the attributes that meet the 
        condition.

    Raises
    ------
    TypeError
        If `box` is not an instance of KeyBox or if `condition` is not callable.

    Example
    -------
    >>> from gofast.tools.box import filter_keybox, KeyBox 
    >>> box = KeyBox(a=1, b=2, c=3)
    >>> filtered_box = filter_keybox(box, lambda k, v: v > 1)
    >>> print(filtered_box)
    b : 2
    c : 3
    """
    if not isinstance(box, KeyBox):
        raise TypeError("The 'box' parameter must be an instance of KeyBox.")
    if not callable(condition):
        raise TypeError("The 'condition' parameter must be callable.")
    
    try:
        filtered_items = {k: v for k, v in box.items() if condition(k, v)}
    except Exception as e:
        raise ValueError(f"An error occurred while applying the condition: {e}")

    return KeyBox(**filtered_items)


def apply_to_keybox(box, func):
    """
    Applies a function to each value in a KeyBox object.

    Parameters
    ----------
    box : KeyBox
        The KeyBox object whose values will be transformed. 
        Must be an instance of KeyBox.
    func : callable
        A function to apply to each value in the KeyBox. 
        Must be a callable object.

    Returns
    -------
    KeyBox
        A new KeyBox object with the function applied to each original value.

    Raises
    ------
    TypeError
        If `box` is not an instance of KeyBox or if `func` is not callable.

    Example
    -------
    >>> from gofast.tools.box import apply_to_keybox, KeyBox
    >>> box = KeyBox(a=1, b=2, c=3)
    >>> squared_box = apply_to_keybox(box, lambda x: x**2)
    >>> print(squared_box)
    a : 1
    b : 4
    c : 9
    """
    if not isinstance(box, KeyBox):
        raise TypeError("The 'box' parameter must be an instance of KeyBox.")
    if not callable(func):
        raise TypeError("The 'func' parameter must be callable.")
    
    try:
        transformed_items = {k: func(v) for k, v in box.items()}
    except Exception as e:
        raise ValueError(f"An error occurred while applying the function: {e}")

    return KeyBox(**transformed_items)

            
def transform_keybox_attributes(box, func, attributes=None):
    """
    Applies a function selectively to attributes of a KeyBox object.

    Parameters
    ----------
    box : KeyBox
        The KeyBox object whose attributes are to be transformed.
    func : callable
        The function to apply to selected attributes of the KeyBox. 
        The function must take a single argument and return a value.
    attributes : list of str, optional
        A list of specific attribute names within the KeyBox to which the 
        function will be applied. If None, the function is applied to all 
        attributes.

    Returns
    -------
    None
        The function modifies the KeyBox object in place, transforming 
        the values of specified attributes according to the provided function.

    Raises
    ------
    TypeError
        If `box` is not an instance of KeyBox or if `func` is not callable.
    KeyError
        If any of the specified attributes do not exist in the KeyBox.

    Example
    -------
    >>> from gofast.tools.box import transform_keybox_attributes, KeyBox
    >>> box = KeyBox(a=1, b=2, c=3)
    >>> def increment(x): return x + 1
    >>> transform_keybox_attributes(box, increment, attributes=['a', 'c'])
    >>> print(box.a, box.b, box.c)
    2 2 4
    """
    if not isinstance(box, KeyBox):
        raise TypeError("The 'box' parameter must be an instance of KeyBox.")
    if not callable(func):
        raise TypeError("The 'func' parameter must be callable.")
    
    target_keys = attributes if attributes is not None else box.keys()
    
    for key in target_keys:
        if key in box:
            box[key] = func(box[key])
        else:
            raise KeyError(f"Attribute '{key}' not found in KeyBox.")
  
def keybox_to_dataframe(keybox):
    """
    Converts a KeyBox object containing multiple Bundle objects into a 
    pandas DataFrame.

    Parameters
    ----------
    keybox : KeyBox
        The KeyBox object to convert.

    Returns
    -------
    pd.DataFrame
        A DataFrame representing the data contained in the KeyBox.

    Raises
    ------
    ValueError
        If the input is not an instance of KeyBox or if the KeyBox 
        does not contain Bundle objects.

    Example
    -------
    >>> import pandas as pd
    >>> from gofast.tools.box import KeyBox, Bundle
    >>> box = KeyBox(item1=Bundle(a=1, b=2), item2=Bundle(a=3, b=4))
    >>> df = keybox_to_dataframe(box)
    >>> print(df)
       a  b
    0  1  2
    1  3  4
    """
    if not isinstance(keybox, KeyBox):
        raise ValueError("Input must be an instance of KeyBox.")

    records = []
    for item in keybox.values():
        if not isinstance(item, (Bundle, KeyBox)):
            raise ValueError("All items in the KeyBox must be Bundle objects.")
        records.append(item.__dict__)

    return pd.DataFrame(records)

def dataframe_to_keybox(df, index_as_key='auto'):
    """
    Converts a pandas DataFrame to a KeyBox object, with each row represented
    as a Bundle object. Offers flexible handling of DataFrame's index for naming
    the Bundle objects within the KeyBox.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to convert.
    index_as_key : bool or 'auto', default 'auto'
        Determines how the DataFrame's index is used as keys in the KeyBox.
        If False, keys are generated as 'row_{index}'. If True, the index values
        are used as keys. If 'auto', index values are used if they are unique
        and of string type, otherwise 'row_{index}' pattern is used.

    Returns
    -------
    KeyBox
        A KeyBox object containing the DataFrame's data, with each row 
        as a Bundle object.

    Raises
    ------
    TypeError
        If the input is not an instance of pd.DataFrame.

    Example
    -------
    >>> import pandas as pd
    >>> from gofast.tools.box import KeyBox, Bundle, dataframe_to_keybox
    >>> df = pd.DataFrame({'a': [1, 3], 'b': [2, 4]})
    >>> box = dataframe_to_keybox(df)
    >>> print(box.row_0.a, box.row_1.b)
    1 4
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be an instance of pd.DataFrame.")

    box = KeyBox()
    use_index_as_key = index_as_key

    # Auto-determine if index can be used as keys
    if index_as_key == 'auto':
        unique_index = df.index.is_unique
        string_index = df.index.map(type).eq(str).all()
        use_index_as_key = unique_index and string_index

    for index, row in df.iterrows():
        key = str(index) if use_index_as_key else f"row_{index}"
        box[key] = Bundle(**row.to_dict())

    return box

def list_dicts_to_keybox(list_dicts, key_func=lambda x, i: f"item_{i}"):
    """
    Converts a list of dictionaries into a KeyBox object, with each dictionary
    converted into a Bundle object.

    Parameters
    ----------
    list_dicts : list of dict
        A list where each element is a dictionary representing an item to be 
        included in the KeyBox. Must be a list of dictionaries.
    key_func : callable, optional
        A function that generates a key for each Bundle object in the KeyBox,
        given the dictionary and its index in the list. Must be a callable object.
        By default, keys are generated as 'item_{index}'.

    Returns
    -------
    KeyBox
        A KeyBox object containing Bundle objects for each dictionary in 
        the input list.

    Raises
    ------
    TypeError
        If `list_dicts` is not a list of dictionaries or if `key_func` is not callable.

    Example
    -------
    >>> list_dicts = [{'name': 'John', 'age': 30}, {'name': 'Jane', 'age': 25}]
    >>> box = list_dicts_to_keybox(list_dicts)
    >>> print(box.item_0.name, box.item_1.age)
    John 25
    """
    if not isinstance(list_dicts, list) or not all(
            isinstance(item, dict) for item in list_dicts):
        raise TypeError("The 'list_dicts' parameter must be a list of dictionaries.")
    if not callable(key_func):
        raise TypeError("The 'key_func' parameter must be callable.")
    
    box = KeyBox()
    try:
        for i, dict_item in enumerate(list_dicts):
            key = key_func(dict_item, i)
            box[key] = Bundle(**dict_item)
    except Exception as e:
        raise ValueError("An error occurred while converting the"
                         f" list of dictionaries: {e}")

    return box

def flatten_keybox(box, parent_key='', sep='_'):
    """
    Flattens a nested KeyBox object into a single-level KeyBox with keys
    representing the path to nested attributes.

    Parameters
    ----------
    box : KeyBox
        The KeyBox object to be flattened.
    parent_key : str, optional
        The prefix to append to child keys. Used internally for recursive calls.
    sep : str, optional
        The separator used between nested keys. Default is '_'.

    Returns
    -------
    KeyBox
        A flattened KeyBox object.

    Example
    -------
    >>> from gofast.tools.box import KeyBox, flatten_keybox
    >>> nested_box = KeyBox(a=KeyBox(b=1, c=2), d=3)
    >>> flat_box = flatten_keybox(nested_box)
    >>> print(flat_box.a_b, flat_box.a_c, flat_box.d)
    1 2 3
    """
    if not isinstance(box, ( KeyBox, Bundle)):
        raise TypeError("The 'box' parameter must be an instance of KeyBox.")
        
    items = []
    for k, v in box.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, KeyBox):
            items.extend(flatten_keybox(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return KeyBox(**dict(items))

         
def update_keybox_if(box, condition, update_func):
    """
    Updates attributes in a KeyBox object if they meet a specified 
    condition.

    Parameters
    ----------
    box : KeyBox
        The KeyBox object to be updated. Must be an instance of KeyBox.
    condition : callable
        A function that takes a key-value pair (attribute name and value) and 
        returns True if the update should be applied. Must be a callable object.
    update_func : callable
        A function to update the value of attributes that meet the condition.
        Must be a callable object.

    Returns
    -------
    None
        The KeyBox object is modified in place.

    Raises
    ------
    TypeError
        If `box` is not an instance of KeyBox or if either `condition` or 
        `update_func` is not callable.

    Example
    -------
    >>> from gofast.tools.box import KeyBox, update_keybox_if
    >>> box = KeyBox(x=10, y=20, z=5)
    >>> update_keybox_if(box, lambda k, v: v > 10, lambda x: x + 5)
    >>> print(box.x, box.y, box.z)
    10 25 5
    """
    if not isinstance(box, ( KeyBox, Bundle)):
        raise TypeError("The 'box' parameter must be an instance of KeyBox.")
    if not callable(condition):
        raise TypeError("The 'condition' parameter must be callable.")
    if not callable(update_func):
        raise TypeError("The 'update_func' parameter must be callable.")
    
    try:
        for k, v in box.items():
            if condition(k, v):
                box[k] = update_func(v)
    except Exception as e:
        raise ValueError(f"An error occurred while applying the update function: {e}")

if __name__=='__main__':
    # from gofast.tools.box import Bundle, KeyBox, save_to_box
    # import pandas as pd
    
    # Simulating some data
    df = pd.DataFrame({
        'user_id': ['user1', 'user2', 'user3'],
        'age': [25, 32, 18],
        'score': [88, 92, 95]
    })
    
    # Saving data into a KeyBox for easy access
    user_data_box = save_to_box(df, 'user_id')
    
    # Accessing data
    print(f"User2's age: {user_data_box.user2.age}")
    print(f"User3's score: {user_data_box.user3.score}")
    
    # Creating a Bundle for summary statistics
    summary_stats = Bundle(mean_age=df['age'].mean(), mean_score=df['score'].mean())
    print(f"Mean age: {summary_stats.mean_age}")
    print(f"Mean score: {summary_stats.mean_score}")
    
    # from gofast.tools.box import KeyBox, filter_keybox, apply_to_keybox
    
    # Example dataset of users and their attributes
    users = KeyBox(
        john=Bundle(age=28, score=85),
        doe=Bundle(age=34, score=90),
        jane=Bundle(age=22, score=95)
    )
    
    # Filtering users based on score
    high_scorers = filter_keybox(users, lambda k, v: v.score > 90)
    print("High scorers:", high_scorers.keys())
    
    # Applying a transformation to the age attribute of each user
    aged_users = apply_to_keybox(users, lambda v: Bundle(age=v.age + 10, score=v.score))
    print("Aged users:")
    for user, attrs in aged_users.items():
        print(f"{user}: age={attrs.age}, score={attrs.score}")
    
    # Output:
    # High scorers: dict_keys(['jane'])
    # Aged users:
    # john: age=38, score=85
    # doe: age=44, score=90
    # jane: age=32, score=95
                   
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
