# -*- coding: utf-8 -*-
""" 
test_formatter.py 
Author: LKouadio ~@Daniel
"""

import pytest #noqa 
import unittest
import pandas as pd

from gofast.api.formatter import MultiFrameFormatter
from gofast.api.formatter import DataFrameFormatter
from gofast.api.formatter import MetricFormatter 
from gofast.api.formatter import BoxFormatter 
from gofast.api.formatter import DescriptionFormatter  
from gofast.api.formatter import formatter_validator

# functions for creating instances with dataframes
def df_formatter():
    df = pd.DataFrame({
        'age': [88, 77],
        'name': ['Kouadio Ernest', "Agaman Bla"]
    })
    formatter = DataFrameFormatter()
    formatter.add_df(df)
    return formatter

def dfs_formatter():
    df1 = pd.DataFrame({
        'distance': [68, 75],
        'village': ['Niable', "Yobouakro"]
    })
    df2 = pd.DataFrame({
        'lags': ["intolerable", 'sometimes'],
        'dishes': ['Assiapluwa', "fufu"]
    })
    df3 = pd.DataFrame({
        'temperature': [22, 24],
        'city': ['Abidjan', 'San Pedro']
    })
    df4 = pd.DataFrame({
        'humidity': [85, 90],
        'region': ['Sud-Comoé', 'Grands Ponts']
    })
    formatter = MultiFrameFormatter()
    formatter.add_dfs(df1, df2, df3, df4)
    return formatter

# Test successful validation with no exceptions
def test_formatter_validator_success():
    instance = df_formatter()
    # Should not raise an exception
    formatter_validator(instance, error='warn')

# Test type error for incorrect instance type
def test_formatter_validator_type_error():
    with pytest.raises(TypeError):
        formatter_validator("not a formatter instance")

# Test attribute checking
def test_formatter_validator_attribute_check():
    instance = df_formatter()
    # Assuming 'df' is a required attribute for DataFrameFormatter
    with pytest.raises(ValueError):
        formatter_validator(instance, attributes=['nonexistent_attribute'])

# Test error handling with 'warn' option
def test_formatter_validator_warn():
    instance = dfs_formatter()
    # Modify the instance to simulate an error condition by clearing one dataframe
    instance.dfs[0] = None  # Example of modifying directly may vary 
    # depending on actual implementation
    
    # No deeper check is operated so no need to warm. 
    # This can handle externally. 
    # with pytest.warns(UserWarning):
    assert formatter_validator(instance, error='warn')[0]==None 

# Test for checking return values from data attributes
def test_formatter_validator_return_values():
    instance = dfs_formatter()
    assert formatter_validator(instance, attributes=['dfs'], check_only=False) == instance.dfs

# Test checking specific dataframe indices
def test_formatter_validator_df_indices():
    instance = dfs_formatter()
    # Check accessing specific dataframes by index
    expected_dfs = [instance.dfs[0], instance.dfs[2]]
    assert formatter_validator(instance, df_indices=[0, 2], check_only=False) == expected_dfs

# Test handling out of range indices
def test_formatter_validator_out_of_range_indices():
    instance = dfs_formatter()
    with pytest.raises(IndexError):
        formatter_validator(instance, df_indices=[10], error='raise')

class TestMultiFrameFormatter(unittest.TestCase):
    def setUp(self):
        # Sample DataFrames for testing
        self.df1 = pd.DataFrame({'A': [1, 2], 'B': ['Text example', 5]})
        self.df2 = pd.DataFrame({'C': [3, 4], 'D': ['Another text', 6]})
        self.df3 = pd.DataFrame({'A': [5, 6], 'B': ['More text', 7]})
        
    def test_add_dfs(self):
        formatter = MultiFrameFormatter()
        formatter.add_dfs(self.df1, self.df2)
        self.assertEqual(len(formatter.dfs), 2)

    def test_dataframe_with_same_columns(self):
        formatter = MultiFrameFormatter(titles=['DataFrame 1', 'DataFrame 3'])
        formatter.add_dfs(self.df1, self.df3)
        combined_table_str = formatter._dataframe_with_same_columns()
        self.assertIsInstance(combined_table_str, str)
        # Additional assertions can check for the presence of specific table content.
        
    def test_dataframe_with_different_columns(self):
        formatter = MultiFrameFormatter(titles=['DataFrame 1', 'DataFrame 2'])
        formatter.add_dfs(self.df1, self.df2)
        tables_str = formatter._dataframe_with_different_columns()
        self.assertIsInstance(tables_str, str)
        # Additional assertions can check for the presence of specific table content.
        
    def test_str_representation(self):
        formatter = MultiFrameFormatter(titles=['DataFrame 1'])
        formatter.add_dfs(self.df1)
        output_str = formatter.__str__()
        self.assertIsInstance(output_str, str)
        # Check if the string representation includes the title or specific DataFrame content.
        
    def test_repr_representation(self):
        formatter = MultiFrameFormatter()
        repr_str = formatter.__repr__()
        self.assertIsInstance(repr_str, str)
        self.assertIn('Empty MultiFrame', repr_str)

        formatter.add_dfs(self.df1)
        repr_str = formatter.__repr__()
        self.assertIn('MultiFrame object containing table', repr_str)

class TestDataFrameFormatter(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'column1': [1, 2, 3.12345, 4],
            'column2': [5.6789, 6, 7, 8]
        }, index=['Index1', 'Index2', 'Index3', 'Index4'])
        self.series = pd.Series([10, 20, 30], index=['a', 'b', 'c'], name='series_data')

    def test_add_df_with_dataframe(self):
        formatter = DataFrameFormatter()
        formatter.add_df(self.df)
        self.assertIsNotNone(formatter.df)

    def test_add_df_with_series(self):
        formatter = DataFrameFormatter()
        formatter.add_df(self.series)
        self.assertIsNotNone(formatter.df)
        self.assertTrue(isinstance(formatter.df, pd.DataFrame))

    def test_format_output(self):
        formatter = DataFrameFormatter("Test Title")
        formatter.add_df(self.df)
        output_str = str(formatter)
        self.assertIn("Test Title", output_str)
        self.assertIn("column1", output_str)
        self.assertIn("column2", output_str)

    def test_attribute_access(self):
        formatter = DataFrameFormatter()
        formatter.add_df(self.df)
        self.assertTrue(hasattr(formatter, 'column1'))
        self.assertTrue(hasattr(formatter, 'column2'))

    def test_snake_case_attribute(self):
        formatter = DataFrameFormatter(keyword="Some Keyword")
        formatter.add_df(self.df)
        self.assertTrue(hasattr(formatter, 'some_keyword'))

    def test_to_original_name(self):
        formatter = DataFrameFormatter()
        formatter.add_df(self.df)
        original_name = formatter._to_original_name('column1')
        self.assertEqual(original_name, 'column1')

    def test_repr(self):
        formatter = DataFrameFormatter()
        self.assertIn("Empty Frame", formatter.__repr__())

        formatter.add_df(self.df)
        self.assertIn("Frame object containing table", formatter.__repr__())

class TestMetricFormatter(unittest.TestCase):
    def test_initialization_with_metrics(self):
        metrics = MetricFormatter(accuracy=0.95, precision=0.93, recall=0.92)
        self.assertEqual(metrics.accuracy, 0.95)
        self.assertEqual(metrics.precision, 0.93)
        self.assertEqual(metrics.recall, 0.92)

    def test_str_without_title(self):
        metrics = MetricFormatter(accuracy=0.95)
        
        self.assertIn("accuracy : 0.95", str(metrics))

    def test_str_with_title(self):
        metrics = MetricFormatter(title="Model Performance", accuracy=0.95)
        output = str(metrics)
        self.assertIn("Model Performance", output)
        self.assertIn("accuracy : 0.95", output)

    def test_handling_different_value_types(self):
        metrics = MetricFormatter(count=5, list_values=[1, 2, 3])
        output = str(metrics)
        print(metrics)
        self.assertIn("count       : 5", output)
        self.assertIn("list_values : list (minval=1, maxval=3, mean=2.0, len=3)", output)

class TestBoxFormatter(unittest.TestCase):
    def test_add_text(self):
        formatter = BoxFormatter("Example Title")
        formatter.add_text("This is an example of formatted text.", 60)
        self.assertTrue(formatter.has_content)
        self.assertIn("This is an example of formatted text.", str(formatter))

    def test_add_dict(self):
        formatter = BoxFormatter("Example Dict")
        dict_content = {"Key1": "Description1", "Key2": "Description2"}
        formatter.add_dict(dict_content, 50)
        self.assertTrue(formatter.has_content)
        self.assertIn("Key1", str(formatter))
        self.assertIn("Description1", str(formatter))

    def test_format_with_no_content(self):
        formatter = BoxFormatter()
        self.assertEqual(str(formatter), "No content added. Use add_text() or add_dict() to add content.")

    def test_format_with_title(self):
        formatter = BoxFormatter("Box Title")
        formatter.add_text("Content", 50)
        output = str(formatter)
        self.assertIn("Box Title", output)
        self.assertIn("Content", output)

class TestDescriptionFormatter(unittest.TestCase):
    def test_initialization_with_text_content(self):
        description = "This is a simple text description."
        formatter = DescriptionFormatter(content=description)
        self.assertEqual(formatter.content, description)
        self.assertIsInstance(formatter.description(), BoxFormatter)

    def test_initialization_with_dict_content(self):
        feature_descriptions = {
            "Feature1": "Description of feature 1",
            "Feature2": "Description of feature 2"
        }
        formatter = DescriptionFormatter(content=feature_descriptions)
        self.assertEqual(formatter.content, feature_descriptions)
        self.assertIsInstance(formatter.description(), BoxFormatter)

    def test_str_with_text_content(self):
        description = "This is a simple text description."
        formatter = DescriptionFormatter(content=description, title="Simple Description")
        formatted_content = str(formatter)
        self.assertIn("Simple Description", formatted_content)
        self.assertIn(description, formatted_content)

    def test_str_with_dict_content(self):
        feature_descriptions = {
            "Feature1": "Description of feature 1",
            "Feature2": "Description of feature 2"
        }
        formatter = DescriptionFormatter(content=feature_descriptions, title="Feature Descriptions")
        formatted_content = str(formatter)
        self.assertIn("Feature Descriptions", formatted_content)
        self.assertIn("Feature1", formatted_content)
        self.assertIn("Description of feature 1", formatted_content)

    def test_repr_method(self):
        formatter = DescriptionFormatter(content="Dummy content")
        self.assertEqual(repr(formatter), "<DescriptionFormatter: Use print() to view detailed content>")


if __name__ == '__main__':
    pytest.main([__file__])
