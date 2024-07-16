# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

""" 
`GoFast`_ Type variables.

The `GoFast` library employs specialized type variables to enhance the clarity 
and precision of type hints throughout the package. These custom type hints 
facilitate the accurate definition of function arguments and return types.

Some customized type variables are used throughout the GoFast package for type 
hinting. These type hints define the expected type of arguments in various 
functions and methods.

M
---
Represents an integer variable used to denote the number of rows in an ``Array``.

N
---
Similar to ``M``, ``N`` indicates the number of columns in an ``Array``. It 
is bound to an integer variable.

_T
---
Generic type variable representing `Any` type. It is kept as the default type 
variable for generic purposes.

U
---
Unlike `_T`, `U` stands for an undefined dimension, typically used to specify a
one-dimensional array. For example:

.. code-block:: python

    import numpy as np
    array = np.arange(4).shape   # Output: (4,)

Shape
-----
Indicates the `Shape` of an array. It is bound by `M`, `U`, `N`. `U` represents
a one-dimensional array. While `Shape` is typically used for two-dimensional 
arrays, it can be extended to more dimensions using the :class:`AddShape`.

DType
-----
Denotes a `DType` object. It is bound to :class:`DType`.

Array1D
--------
Defined for one-dimensional arrays with an optional `DType`. For example:

.. code-block:: python

    import numpy as np
    from gofast._typing import TypeVar, Array, DType
    _T = TypeVar('_T', float)
    A = TypeVar('A', str, bytes)
    arr1: Array1D[_T, DType[_T]] = np.arange(21)
    arr2: Array1D[A, DType[A]] = arr1.astype('str')

NDArray
-------
Represents multi-dimensional arrays. Unlike `Array`, `NDArray` can have multiple
dimensions. Example usage:

.. code-block:: python

    import numpy as np
    from gofast.typing import TypeVar, Array, NDArray, DType
    _T = TypeVar('_T', int)
    U = TypeVar('U')
    multidarray = np.arange(7, 7).astype(np.int32)
    def accept_multid(arrays: NDArray[Array[_T, U], DType[_T]]=multidarray): ...

_Sub
----
Represents a subset of an `Array`. For example, extracting a specific zone 
from an array:

.. code-block:: python

    import numpy as np
    from gofast.api.types import TypeVar, DType, Array, _Sub
    from gofast.tools.mathex import _define_conductive_zone
    _T = TypeVar('_T', float)
    erp_array: Array[_T, DType[_T]] = np.random.randn(21)
    select_zone, _ = _define_conductive_zone(erp=erp_array, auto=True)
    def check_cz(select_zone: _Sub[Array]): ...

_SP
----
Stands for Station Positions, typically used in electrical resistivity 
profiling. Example:

.. code-block:: python

    import numpy as np
    from gofast.api.types import TypeVar, DType, _SP, _Sub
    _T = TypeVar('_T', bound=int)
    surveyL: _SP = np.arange(0, 50 * 121, 50.).astype(np.int32)

Series
------
Represents a `pandas Series`_ object. It is used in place of `pandas.Series` 
for type hinting throughout the package.

DataFrame
---------
Similarly, `DataFrame` represents a `pandas DataFrame`_ object. It is used 
instead of `pandas.DataFrame` for consistency in type hinting.

Examples of Series and DataFrame usage:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast.api.types import TypeVar, Any, DType, Series, DataFrame
    _T = TypeVar('_T')
    seriesStr = pd.Series([f'obx{s}' for s in range(21)], name='stringobj')
    seriesFloat = pd.Series(np.arange(7).astype(np.float32), name='floatobj')
    SERs = Series[DType[str]]
    SERf = Series[DType[float]]
    dfStr = pd.DataFrame({'ser1': seriesStr, 'obj2': [f'none' for i in range(21)]})
    dfFloat = pd.DataFrame({'ser1': seriesFloat, 'obj2': np.linspace(3, 28, 7)})
    dfAny = pd.DataFrame({'ser1': seriesStr, 'ser2': seriesFloat})
    DFs = DataFrame[SERs] or DataFrame[DType[str]]
    DFf = DataFrame[SERf] or DataFrame[DType[float]]
    DFa = DataFrame[Series[Any]] or DataFrame[DType[_T]]
"""
from __future__ import annotations 
# from typing import TYPE_CHECKING

# if TYPE_CHECKING:
from typing import (
    List,
    Tuple,
    Sequence,
    Dict,
    Iterable,
    Callable,
    Union,
    Any,
    Generic,
    Optional,
    Type,
    Mapping,
    Text, 
    TypeVar,
    Iterator,
    SupportsInt,
    Set,
    ContextManager,
    Deque, 
    FrozenSet,
    NamedTuple,
    NewType, 
    TypeGuard, 
    TypedDict, 
    Generator
)


__all__ = [
    "List",
    "Tuple", 
    "Sequence", 
    "Dict", 
    "Iterable",
    "Callable",
    "Any",
    "Generic", 
    "Optional", 
    "Union", 
    "Type", 
    "Mapping",
    "Text", 
    "Shape",
    "DType", 
    "NDArray",
    "ArrayLike",
    "_Sub",
    "_SP",
    "_F",
    "_T", 
    "_V",
    "Series",
    "Iterator",
    "SupportsInt",
    "Set", 
    "_ContextManager",
    "_Deque",
    "_FrozenSet",
    "_NamedTuple",
    "_NewType",
    "_TypedDict",
    "TypeGuard",  
    "TypedDict", 
    "Generator", 
    "Array1D", 
    "TypedCallable", 
    "LambdaType", 
    "NumPyFunction",
    "_Tensor",
    "_Dataset",
    "_Loss",
    "_Regularizer",
    "_Optimizer",
    "_Sequential", 
    "_History",
    "_Callback",
    "_Model", 
    "_Metric", 
    
]

_T = TypeVar('_T')
_V = TypeVar('_V')
K = TypeVar('K')
M = TypeVar('M', bound=int)
N = TypeVar('N', bound=int)
U = TypeVar('U')
D = TypeVar('D', bound='DType')
S = TypeVar('S', bound='Shape')
# Define type variables for the arguments and return type
A = TypeVar('A')  # Argument type
R = TypeVar('R')  # Return type

# ArrayLike = Union[Sequence, np.ndarray, pd.Series, pd.DataFrame]
class AddShape(Generic[S]): 
    """
    Represents an additional dimension for shapes beyond two dimensions.
    Useful for defining shapes with more than two dimensions.
    """

class Shape(Generic[M, N, S]):  
    """
    Generic type for constructing a tuple shape for NDArray. It is 
    specifically designed for two-dimensional arrays with M rows and N columns.

    Example:
        >>> import numpy as np 
        >>> array = np.random.randn(7, 3, 3) 
        >>> def check_valid_type(array: NDArray[Array[float], Shape[M, N, S]]): ...
    """
    
class DType(Generic[D]): 
    """
    Represents data types of elements in arrays. DType can be any type,
    allowing for the specification of array data types.

    Example:
        >>> import numpy as np
        >>> array = np.random.randn(7)
        >>> def check_dtype(array: NDArray[Array[float], DType[float]]): ...
    """

class ArrayLike(Generic[_T]):
    """
    Represents a 2-dimensional array-like structure, suitable for data 
    structures like Pandas DataFrame and Series.
    For N-dimensional arrays, NDArray should be used instead.

    Example:
        >>> import pandas as pd
        >>> def check_dataframe(array: ArrayLike[pd.DataFrame]): ...
        >>> def check_series(array: ArrayLike[pd.Series]): ...
    """

class NDArray(Generic[_T]):
    """
    Represents a generic N-dimensional array with specified data types, 
    suitable for N-dimensional structures,
    including but not limited to NumPy arrays.

    Example:
        >>> import numpy as np
        >>> def check_ndarray(array: NDArray[np.ndarray]): ...
    """

class _F(Generic[_T]):
    """
    Represents a generic callable type, including functions, methods, and classes.
    It allows for the specification of return types and argument types of callables.

    Example:
        >>> def my_function(x: int) -> str: ...
        >>> def check_callable(func: _F[Callable[[int], str]]): ...
    """

class LambdaType(Generic[A, R]):
    """
    Represents a type hint for lambda functions, specifying their argument 
    and return types.
    
    This enhances type checking and clarity when using lambda functions in 
    codebases, especially in contexts where the function signature is crucial 
    for understanding the code's intent.

    LambdaType can be used to explicitly define the expected argument types 
    and the return type  of a lambda, providing a clearer interface 
    contract within codebases.

    Example:
        >>> my_lambda: LambdaType[int, str] = lambda x: str(x)
        >>> def process_lambda(func: LambdaType[int, str], value: int) -> str:
        ...     return func(value)
    """

class NumPyFunction(Generic[R]):
    """
    A type hint for representing NumPy functions that operate on array-like
    structures and return either a scalar or an array. This class captures the
    signature of such functions, including their input types and return type.

    This type hint is particularly useful for annotating functions that accept 
    other NumPy functions as arguments, or for specifying the expected behavior
    of user-defined functions that mimic NumPy reduction operations.

    Example:
        >>> from numpy.typing import ArrayLike
        >>> def apply_numpy_func(func: NumPyFunction[float], data: ArrayLike) -> float:
        ...     return func(data)
    """

    def __init__(self, func: Callable[[ArrayLike, Union[None, int, Sequence[int]]], R]):
        self.func = func
    
    def __call__(self, *args, **kwargs) -> R:
        return self.func(*args, **kwargs)

class TypedCallable(Generic[_T]):
    """
    Represents a generic callable type, including functions, methods, and classes.
    It allows for the specification of return types and argument types of 
    callables, enhancing type checking and documentation.

    The type hint can be used to explicitly define the expected argument types
    and the return type
    of a callable, providing clearer interface contracts within codebases.

    Example:
        >>> def my_function(x: int) -> str: ...
        >>> def check_callable(func: TypedCallable[Callable[[int], str]]): ...
    """

class _Sub(Generic[_T]):
    """
    Represents a subset of an array or collection.
    """

class _SP(Generic[_T, D]):
    """
    Represents station position arrays in geophysical surveys.
    Typically holds integer values representing station positions.

    Example:
        >>> import numpy as np
        >>> positions: _SP = np.arange(0, 21 * 10, 10).astype(np.int32)
        >>> def check_station_positions(positions: _SP[ArrayLike[int], DType[int]]): ...
    """

class Series(Generic[_T]):
    """
    Represents a pandas Series object.

    Example:
        >>> import pandas as pd
        >>> series = pd.Series([1, 2, 3])
        >>> def check_series(series: Series[int]): ...
    """

class DataFrame(Generic[_T]):
    """
    Represents a pandas DataFrame object.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
        >>> def check_dataframe(df: DataFrame[object]): ...
    """
    
class _ContextManager(ContextManager, Generic[_T]):
    """
    Represents a context manager, a resource that is specifically designed
    to be used with the 'with' statement.

    Example:
        >>> from contextlib import contextmanager
        >>> @contextmanager
        ... def my_context_manager() -> Generator[None, None, None]:
        ...     print("Enter context")
        ...     yield
        ...     print("Exit context")
        >>> def check_context(context: _ContextManager[None]): ...
    """

class _Deque(Deque, Generic[_T]):
    """
    Represents a double-ended queue, allowing for efficient addition
    and removal of elements from either end.

    Example:
        >>> from collections import deque
        >>> my_deque: Deque[int] = deque([1, 2, 3])
        >>> def check_deque(dq: _Deque[int]): ...
    """

class _FrozenSet(FrozenSet, Generic[_T]):
    """
    Represents an immutable set.

    Example:
        >>> my_frozen_set: FrozenSet[int] = frozenset([1, 2, 3])
        >>> def check_frozen_set(fs: _FrozenSet[int]): ...
    """

class _NamedTuple(NamedTuple):
    """
    Represents a named tuple, combining the benefits of tuples and dictionaries.

    Example:
        >>> from typing import NamedTuple
        >>> class MyNamedTuple(NamedTuple):
        ...     name: str
        ...     age: int
        >>> def check_named_tuple(nt: _NamedTuple[str, int]): ...
    """

class _NewType(NewType, Generic[_T]):
    """
    Represents a new type, created from an existing type, used for type checking.

    Example:
        >>> from typing import NewType
        >>> UserId = _NewType('UserId', int)
        >>> def check_user_id(user_id: UserId): ...
    """

class _TypedDict(TypedDict):
    """
    Represents a dictionary with fixed keys and specified value types.
    
    `_TypedDict` creates a dictionary type with specific types for each key.
    It helps to ensure that each field in the dictionary adheres to the
    specified type, providing better type checking at the code analysis stage.
    
    Example:
        >>> from typing import TypedDict
        >>> class MyDict(_TypedDict):
        ...     name: str
        ...     age: int
        >>> def check_typed_dict(d: MyDict): ...
    """
    

class BeautifulSoupTag(Generic[_T]):
    """
    A type hint for BeautifulSoup Tag objects, meant to provide clearer 
    documentation and usage within type-checked Python code. This class does 
    not implement functionality but serves as a type hint for functions or 
    methods that expect or return BeautifulSoup Tag objects.

    The generic type T is illustrative, representing the type of data expected
    to be extracted from the Tag, such as strings for text, although
    BeautifulSoup does not enforce this directly.

    Example:
        >>> from bs4 import BeautifulSoup
        >>> def get_tag_text(tag: BeautifulSoupTag[str]) -> str:
        ...     return tag.get_text()

        >>> soup = BeautifulSoup('<p>Hello, world!</p>', 'html.parser')
        >>> p_tag = soup.find('p')
        >>> text = get_tag_text(p_tag)
        >>> print(text)
        Hello, world!
    """
    
class Array1D(Generic[_T]):
    """
    A generic class for representing and manipulating one-dimensional arrays 
    of any type.

    This class provides basic functionality similar to a Python list but
    ensures type safety, so all elements in the array are of the 
    specified type `T`.

    Attributes:
        elements (List[T]): The list of elements stored in the array.

    Type Parameters:
        T: The type of elements in the `Array1D`. This can be any valid Python type.

    Example Usage:
        >>> numbers = Array1D[int]([1, 2, 3, 4])
        >>> print(numbers)
        Array1D([1, 2, 3, 4])
        
        >>> words = Array1D[str](["hello", "world"])
        >>> words[1] = "python"
        >>> print(words)
        Array1D(['hello', 'python'])
    """

    def __init__(self, elements: List[_T]):
        """
        Initializes a new instance of the Array1D class with the provided elements.

        Parameters:
            elements (List[T]): A list of elements of type `T` to initialize the array.
        """
        self.elements = elements
    
    def __getitem__(self, index: int) -> _T:
        """
        Allows indexing into the array to get an element.

        Parameters:
            index (int): The index of the element to retrieve.

        Returns:
            T: The element at the specified index.

        Raises:
            IndexError: If the index is out of bounds.
        """
        return self.elements[index]
    
    def __setitem__(self, index: int, value: _T):
        """
        Allows setting the value of an element at a specific index.

        Parameters:
            index (int): The index of the element to modify.
            value (T): The new value to set at the specified index.

        Raises:
            IndexError: If the index is out of bounds.
        """
        self.elements[index] = value
    
    def __len__(self) -> int:
        """
        Returns the number of elements in the array.

        Returns:
            int: The length of the array.
        """
        return len(self.elements)
    
    def add(self, value: _T):
        """
        Appends a new element to the end of the array.

        Parameters:
            value (T): The element to append to the array.
        """
        self.elements.append(value)
    
    def __repr__(self) -> str:
        """
        Provides a string representation of the array instance, useful for debugging.

        Returns:
            str: A string representation of the `Array1D` instance.
        """
        return f"Array1D({self.elements})"

class _Tensor:
    """
    Represents a mathematical tensor, which could be used for operations
    in a neural network.

    Parameters
    ----------
    data : Any
        The underlying data of the tensor, typically an array or a matrix.

    Examples
    --------
    >>> import numpy as np
    >>> t = Tensor(np.array([1, 2, 3]))
    >>> t.data
    array([1, 2, 3])
    """
    def __init__(self, data: Any):
        self.data = data

class _Dataset:
    """
    Dataset class for handling batches of data for training or testing
    a model.

    Notes
    -----
    This class needs to be extended to implement custom loading and
    batching mechanisms.

    Examples
    --------
    >>> class SimpleDataset(Dataset):
    ...     def __init__(self, data):
    ...         self.data = data
    ...     def __getitem__(self, index):
    ...         return Tensor(self.data[index])
    ...     def __len__(self):
    ...         return len(self.data)
    >>> ds = SimpleDataset([1, 2, 3])
    >>> ds[0].data
    1
    """
    def __getitem__(self, index: int) -> _Tensor:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

class _Loss:
    """
    Base class for loss functions.

    Notes
    -----
    Implement specific loss functions by subclassing this class.

    Examples
    --------
    >>> class MSE(Loss):
    ...     def __call__(self, predictions, targets):
    ...         return Tensor(((predictions.data - targets.data) ** 2).mean())
    >>> loss = MSE()
    >>> predictions = Tensor(np.array([2, 3]))
    >>> targets = Tensor(np.array([1, 1]))
    >>> loss(predictions, targets).data
    2.5
    """
    def __call__(self, predictions: _Tensor, targets: _Tensor) -> _Tensor:
        pass

class _Regularizer:
    """
    Base class for regularization techniques.

    Notes
    -----
    Regularizers add a penalty to the loss function, based on model complexity.

    Examples
    --------
    >>> class L2(Regularizer):
    ...     def __call__(self, parameter):
    ...         return Tensor(0.01 * np.sum(parameter.data ** 2))
    >>> reg = L2()
    >>> parameter = Tensor(np.array([2, 3]))
    >>> reg(parameter).data
    0.13
    """
    def __call__(self, parameter: _Tensor) -> _Tensor:
        pass

class _Optimizer:
    """
    Base class for optimization algorithms used in training models.

    Notes
    -----
    Optimizers adjust the parameters of the model in response to the
    gradient calculated by the backward pass.

    Examples
    --------
    >>> class SGD(Optimizer):
    ...     def step(self, gradients):
    ...         for grad in gradients:
    ...             grad.data -= 0.01 * grad.data
    >>> optimizer = SGD()
    >>> grads = [Tensor(np.array([0.1, 0.2])), Tensor(np.array([0.3]))]
    >>> optimizer.step(grads)
    >>> grads[0].data
    array([0.099, 0.198])
    """
    def step(self, gradients: List[_Tensor]) -> None:
        pass

class _History:
    """
    Class for recording and retrieving the training history, such as
    loss over epochs.

    Attributes
    ----------
    records : dict
        A dictionary to store metric values by their names.

    Examples
    --------
    >>> history = History()
    >>> history.log('loss', 0.5)
    >>> history.records['loss']
    [0.5]
    """
    records: dict

    def __init__(self):
        self.records = {}

    def log(self, metric: str, value: Any) -> None:
        self.records.setdefault(metric, []).append(value)

class _Callback:
    """
    Base class for creating callbacks to monitor and take actions during
    training.

    Notes
    -----
    Callbacks can be used to implement behaviors like early stopping,
    learning rate adjustments, or model checkpointing during training.

    Examples
    --------
    >>> class PrintEpochNumber(Callback):
    ...     def on_epoch_end(self, epoch, logs):
    ...         print(f'Epoch {epoch} ended')
    >>> callback = PrintEpochNumber()
    >>> callback.on_epoch_end(1, None)
    Epoch 1 ended
    """
    def on_epoch_end(self, epoch: int, logs: Optional[_History]) -> None:
        pass

class _Model:
    """
    Class representing a machine learning model.

    Parameters
    ----------
    layers : List[Any]
        A list of layers within the model.

    Examples
    --------
    >>> model = Model(layers=[Layer(), Layer()])
    >>> input_tensor = Tensor(np.array([1, 2, 3]))
    >>> output = model.forward(input_tensor)
    >>> model.predict(input_tensor).data
    array([output.data])

    Notes
    -----
    This class should be extended with specific layer handling, forward,
    and backward passes.
    """
    def __init__(self, layers: List[Any]):
        self.layers = layers

    def forward(self, input: _Tensor) -> _Tensor:
        pass

    def backward(self, loss_grad: _Tensor) -> None:
        pass

    def predict(self, input: _Tensor) -> _Tensor:
        pass

    def train(self, dataset: _Dataset, loss_fn: _Loss, optimizer: _Optimizer,
              callbacks: List[_Callback] = []) -> _History:
        pass

class _Sequential: 
    """ Base class for Sequential used to create  neural networks models."""
    
    def __init__( self, model: _Model ): 
        self.model= model  
    
class _Metric:
    """
    Base class for metrics used to evaluate machine learning models.

    Notes
    -----
    This class is intended as a foundation. Specific metrics should 
    inherit from this class and implement the `update` and `compute` 
    methods according to their specific metric calculations (e.g., accuracy,
    precision, recall).

    Examples
    --------
    >>> class Accuracy(Metric):
    ...     def __init__(self):
    ...         super().__init__('accuracy')
    ...         self.correct = 0
    ...         self.total = 0
    ...     def update(self, predictions, targets):
    ...         self.correct += (predictions == targets).sum()
    ...         self.total += len(predictions)
    ...     def compute(self):
    ...         return self.correct / self.total if self.total > 0 else 0
    >>> predictions = [1, 2, 3, 4]
    >>> targets = [1, 2, 2, 4]
    >>> acc = Accuracy()
    >>> acc.update(predictions, targets)
    >>> acc.compute()
    0.75
    """
    def __init__(self, name: str):
        """
        Initializes the Metric with a given name.

        Parameters
        ----------
        name : str
            The name of the metric.
        """
        self.name: str = name
        self.value: float = 0.0

    def reset(self) -> None:
        """
        Resets the metric's calculated value to its initial state.
        """
        self.value = 0.0

    def update(self, predictions: Any, targets: Any) -> None:
        """
        Updates the metric based on the provided predictions and targets.

        Parameters
        ----------
        predictions : Any
            The predictions made by the model, expected to be a list or array.
        targets : Any
            The actual target values, expected to be in the same format as predictions.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def compute(self) -> float:
        """
        Computes the final value of the metric after all updates.

        Returns
        -------
        float
            The computed metric value.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


if __name__ == '__main__':
    # Test cases demonstrating the usage of defined generic types.
    ...

























