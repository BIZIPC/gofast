# -*- coding: utf-8 -*-
"""
test_ts.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock # noqa 
import pytest
import seaborn as sns 
from gofast.tools.coreutils import is_module_installed 
from gofast.tools.funcutils import install_package 
from gofast.plot.ts import TimeSeriesPlotter  

try:
    import pytest_mock # noqa 
except ImportError:
    if not is_module_installed('pytest_mock'): 
        install_package(
            'pytest_mock', dist_name="pytest-mock", infer_dist_name=True ) 
 
# def test_fit_method():
#     # Create a sample DataFrame
#     dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
#     values = np.random.rand(len(dates))
#     df = pd.DataFrame({'Date': dates, 'Value': values})

#     # Instantiate the TimeSeriesPlotter
#     plotter = TimeSeriesPlotter()

#     # Test the fit method
#     returned_object = plotter.fit(df, date_col='Date', value_col='Value')

#     # Assert that the method returns self for chaining
#     assert returned_object is plotter, "Fit method should return self for method chaining."

#     # Assert that the internal state is set correctly
#     assert plotter.data is not None, "Data should be properly assigned."
#     assert plotter.date_col == 'Date', "Date column should be set correctly."
#     assert plotter.value_col == 'Value', "Value column should be set correctly."

#     # Optionally, check if date conversion and column extraction are correct
#     # This might require mocking or further setup to validate properly
#     assert pd.to_datetime(plotter.data['Date']).equals(df['Date']), ( 
#         "Dates should be correctly converted to datetime."
#         )
    
# def test_plotRollingMean(mocker):
#     mock_figure = mocker.patch('matplotlib.pyplot.figure')
#     mocker.patch('matplotlib.pyplot.show')
#     mocker.patch('matplotlib.pyplot.plot')
    
#     # Create a sample DataFrame
#     dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='D')
#     values = np.random.rand(len(dates))
#     df = pd.DataFrame({'Date': dates, 'Value': values})

#     # Instantiate the TimeSeriesPlotter and fit data
#     plotter = TimeSeriesPlotter()
#     plotter.fit(data=df, date_col='Date', value_col='Value')

#     # Test plotRollingMeanStd
#     plotter.plotRollingMeanStd(window=10)

#     # Check if figure functions are called the expected number of times
#     assert mock_figure.call_count == 5, ( 
#         f"Expected 'figure' to have been called 5 times. Called {mock_figure.call_count} times."
#         )
#     plt.show.assert_called_once()

# def test_plotAutocorrelation(mocker):

#     mock_figure = mocker.patch('matplotlib.pyplot.figure')
#     mocker.patch('matplotlib.pyplot.show')
#     mocker.patch('pandas.plotting.autocorrelation_plot', return_value=mocker.Mock())

#     # Prepare data
#     dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='D')
#     values = np.random.rand(len(dates))
#     df = pd.DataFrame({'Date': dates, 'Value': values})

#     plotter = TimeSeriesPlotter()
#     plotter.fit(data=df, date_col='Date', value_col='Value')

#     # Test plotAutocorrelation
#     plotter.plotAutoCorrelation()
#     #print(mock_figure.call_count )
#     assert mock_figure.call_count == 3, ( 
#         f"Expected 'figure' to have been called 3 times. Called {mock_figure.call_count} times."
#         )
#     plt.show.assert_called_once()
    
# @pytest.mark.skipif (not is_module_installed('statsmodels'),
#                       reason= 'Need statsmodels to be installed for running this code.')
# def test_plotPACF(mocker):
#     mocker.patch('matplotlib.pyplot.figure')
#     mocker.patch('matplotlib.pyplot.show')
#     mocker.patch('statsmodels.graphics.tsaplots.plot_pacf', return_value=mocker.Mock())

#     # Data setup
#     dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='D')
#     values = np.random.rand(len(dates))
#     df = pd.DataFrame({'Date': dates, 'Value': values})

#     plotter = TimeSeriesPlotter()
#     plotter.fit(data=df, date_col='Date', value_col='Value')

#     # Test plotPACF
#     plotter.plotPACF()

#     plt.show.assert_called_once()
    
  
# def test_plotDecomposition(mocker):
#     mocker.patch('matplotlib.pyplot.figure')
#     mocker.patch('matplotlib.pyplot.show')
#     mocker.patch('statsmodels.tsa.seasonal.seasonal_decompose')
    
#     # Create a sample DataFrame with sufficient data
#     dates = pd.date_range(start="2020-01-01", end="2021-12-31", freq='M')  # 24 months
#     values = np.random.rand(len(dates))
#     df = pd.DataFrame({'Date': dates, 'Value': values})

#     # Instantiate and fit
#     plotter = TimeSeriesPlotter()
#     plotter.fit(data=df, date_col='Date', value_col='Value')

#     # Test plotDecomposition
#     plotter.plotDecomposition()

#     # statsmodels.tsa.seasonal.seasonal_decompose.assert_called_once()
#     # plt.figure.assert_called_once()
#     plt.show.assert_called_once()


# @pytest.mark.parametrize("freq, expected_call_count", [
#     (12, 24),  # 24 observations, frequency 12
#     (6, 12),   # 12 observations, frequency 6
# ])
# def test_plotDecomposition_dynamic(mocker, freq, expected_call_count):
#     mocker.patch('matplotlib.pyplot.figure')
#     mocker.patch('matplotlib.pyplot.show')
#     mocker.patch('statsmodels.tsa.seasonal.seasonal_decompose')

#     # Create a DataFrame based on the test parameters
#     dates = pd.date_range(start="2020-01-01", periods=expected_call_count, freq='M')
#     values = np.random.rand(len(dates))
#     df = pd.DataFrame({'Date': dates, 'Value': values})

#     # Instantiate and fit
#     plotter = TimeSeriesPlotter()
#     plotter.fit(data=df, date_col='Date', value_col='Value')

#     # Test plotDecomposition
#     plotter.plotDecomposition(freq=freq)

#     plt.show.assert_called_once()

# def test_plotCumulativeLine(mocker):
#     mocker.patch('matplotlib.pyplot.figure')
#     mocker.patch('matplotlib.pyplot.show')
#     mocker.patch('matplotlib.pyplot.plot')

#     # Setup DataFrame
#     dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='D')
#     values = np.random.rand(len(dates))
#     df = pd.DataFrame({'Date': dates, 'Value': values})

#     # Instantiate and fit
#     plotter = TimeSeriesPlotter()
#     plotter.fit(data=df, date_col='Date', value_col='Value')

#     # Test
#     plotter.plotCumulativeLine()

#     plt.show.assert_called_once()
# def test_plotDensity(mocker):
    
#     mocker.patch('matplotlib.pyplot.figure')
#     mocker.patch('matplotlib.pyplot.show')
#     mocker.patch('seaborn.kdeplot')

#     # Data setup
#     dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='D')
#     values = np.random.rand(len(dates))
#     df = pd.DataFrame({'Date': dates, 'Value': values})

#     # Instantiate and fit
#     plotter = TimeSeriesPlotter()
#     plotter.fit(data=df, date_col='Date', value_col='Value')

#     # Test
#     plotter.plotDensity()

#     plt.show.assert_called_once()

# def test_plotScatterWithTrendline(mocker):
#     mocker.patch('matplotlib.pyplot.figure')
#     mocker.patch('matplotlib.pyplot.show')
#     mocker.patch('seaborn.regplot')

#     # Data setup
#     dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='D')
#     values = np.random.rand(len(dates))
#     df = pd.DataFrame({'Date': dates, 'Value': values})

#     # Instantiate and fit
#     plotter = TimeSeriesPlotter()
#     plotter.fit(data=df, date_col='Date', value_col='Value')

#     # Test
#     plotter.plotScatterWithTrendline()

#     plt.show.assert_called_once()

# def test_plotBar(mocker):
#     mocker.patch('matplotlib.pyplot.show')
#     mocker.patch('pandas.plotting._core.PlotAccessor.bar')

#     # Setup DataFrame
#     dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
#     values = np.random.rand(len(dates))
#     df = pd.DataFrame({'Date': dates, 'Value': values})

#     # Instantiate and fit
#     plotter = TimeSeriesPlotter()
#     plotter.fit(data=df, date_col='Date', value_col='Value')

#     # Test
#     plotter.plotBar()

#     plt.show.assert_called_once()

# def test_plotStackedArea(mocker):
#     mocker.patch('matplotlib.pyplot.show')
#     mocker.patch('matplotlib.pyplot.stackplot')

#     # Prepare data
#     dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='D')
#     values = np.random.rand(len(dates))
#     df = pd.DataFrame({'Date': dates, 'Value': values})

#     # Instantiate and fit
#     plotter = TimeSeriesPlotter()
#     plotter.fit(data=df, date_col='Date', value_col='Value')

#     # Test
#     plotter.plotStackedArea()

#     # plt.stackplot.assert_called_once()
#     plt.show.assert_called_once()

# def test_plotViolin(mocker):
#     mocker.patch('matplotlib.pyplot.show')
#     mocker.patch('matplotlib.pyplot.xticks')
#     mocker.patch('seaborn.violinplot')

#     # Data setup
#     dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='D')
#     values = np.random.rand(len(dates))
#     df = pd.DataFrame({'Date': dates, 'Value': values})

#     # Instantiate and fit
#     plotter = TimeSeriesPlotter()
#     plotter.fit(data=df, date_col='Date', value_col='Value')

#     # Test
#     plotter.plotViolin()

#     # mock_violin.assert_called_once()
#     plt.show.assert_called_once()
#     plt.xticks.assert_called_once_with(rotation=45)

# def test_pieChart(mocker):
#     mocker.patch('matplotlib.pyplot.show')
#     mocker.patch('pandas.plotting._core.PlotAccessor.pie')

#     # Setup DataFrame
#     categories = ['A', 'B', 'C', 'D']
#     values = [10, 20, 30, 40]
#     df = pd.DataFrame({'Category': categories, 'Value': values})

#     # Instantiate and fit
#     plotter = TimeSeriesPlotter()
#     plotter.fit(data=df, date_col='Category', value_col='Value')

#     # Test
#     plotter.pieChart()

#     # pandas.plotting._core.PlotAccessor.pie.assert_called_once()
#     plt.show.assert_called_once()


# def test_plotLine(mocker):
#     # Create a mock for the subplot which returns a figure and an axes object
#     fig, ax = plt.subplots()  # Create real subplot to ensure correct tuple
#     mocker.patch('matplotlib.pyplot.subplots', return_value=(fig, ax))
    
#     mocker.patch('matplotlib.pyplot.show')
#     mock_lineplot = mocker.patch('seaborn.lineplot')

#     # Prepare data
#     dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq='D')
#     values = np.random.rand(len(dates))
#     df = pd.DataFrame({'Date': dates, 'Value': values})

#     # Instantiate and fit
#     plotter = TimeSeriesPlotter()
#     plotter.fit(data=df, date_col='Date', value_col='Value')

#     # Test
#     plotter.plotLine()

#     # Asserts to check if lineplot was called correctly
#     mock_lineplot.assert_called_once()
#     assert mock_lineplot.call_args[1]['ax'] == ax  # Ensure that ax is passed correctly
#     plt.show.assert_called_once()
#     plt.close() 

# def test_histogram(mocker):
#     mocker.patch('matplotlib.pyplot.subplots')
#     mocker.patch('matplotlib.pyplot.show')
#     mocker.patch('seaborn.histplot')

#     # Setup DataFrame
#     values = np.random.randn(100)
#     df = pd.DataFrame({'Value': values})

#     # Instantiate and fit
#     plotter = TimeSeriesPlotter()
#     plotter.fit(data=df, date_col=None, value_col='Value')

#     # Test
#     plotter.plotHistogram()

#     plt.show.assert_called_once()

# def test_histogram(mocker):
#     # Create a mock for the subplot which returns a figure and an axes object
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     mocker.patch('matplotlib.pyplot.subplots', return_value=(fig, ax))

#     mocker.patch('matplotlib.pyplot.show')
#     mocker.patch('seaborn.histplot')

#     # Setup DataFrame
#     values = np.random.randn(100)
#     df = pd.DataFrame({'Value': values})

#     # Instantiate and fit
#     plotter = TimeSeriesPlotter()
#     plotter.fit(data=df, date_col=None, value_col='Value')

#     # Test
#     plotter.plotHistogram()

#     plt.show.assert_called_once()
#     sns.histplot.assert_called_once()

#     # Asserts can be added here to test more specific behavior 
#     # such as the call parameters of histplot
#     call_args = sns.histplot.call_args
#     assert call_args[1]['bins'] == 30  # Default bins
#     assert call_args[1]['ax'] == ax  # Passed ax

# def test_plotBox(mocker):
#     mocker.patch('matplotlib.pyplot.subplots')
#     mocker.patch('matplotlib.pyplot.xticks')
#     mocker.patch('matplotlib.pyplot.show')
#     mock_boxplot = mocker.patch('seaborn.boxplot')

#     # Prepare data
#     dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq='D')
#     values = np.random.rand(len(dates))
#     df = pd.DataFrame({'Date': dates, 'Value': values})

#     # Instantiate and fit
#     plotter = TimeSeriesPlotter()
#     plotter.fit(data=df, date_col='Date', value_col='Value')

#     # Test
#     plotter.plotBox()

#     mock_boxplot.assert_called_once()
#     plt.xticks.assert_called_once_with(rotation=45)
#     plt.show.assert_called_once()

# def test_plotArea(mocker):
#     mocker.patch('matplotlib.pyplot.figure')
#     mocker.patch('matplotlib.pyplot.show')
#     mocker.patch('matplotlib.pyplot.fill_between')

#     # Setup DataFrame
#     dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq='D')
#     values = np.random.rand(len(dates))
#     df = pd.DataFrame({'Date': dates, 'Value': values})

#     # Instantiate and fit
#     plotter = TimeSeriesPlotter()
#     plotter.fit(data=df, date_col='Date', value_col='Value')

#     # Test
#     plotter.plotArea()

#     plt.fill_between.assert_called_once()
#     plt.show.assert_called_once()

# def test_heatmapCorrelation(mocker):
#     mocker.patch('matplotlib.pyplot.figure')
#     mocker.patch('matplotlib.pyplot.show')
#     mock_heatmap = mocker.patch('seaborn.heatmap')

#     # Prepare data
#     df = pd.DataFrame(np.random.rand(10, 10), columns=[f"col{i}" for i in range(10)])

#     # Instantiate and fit
#     plotter = TimeSeriesPlotter()
#     plotter.fit(data=df, date_col=None, value_col='col0')

#     # Test
#     plotter.heatmapCorrelation()

#     mock_heatmap.assert_called_once()
#     plt.show.assert_called_once()

# def test_plotBox(mocker):
#     fig, ax = plt.subplots()  # Create real subplot to ensure correct tuple
#     mocker.patch('matplotlib.pyplot.subplots', return_value=(fig, ax))
#     mocker.patch('matplotlib.pyplot.show')
#     mocker.patch('matplotlib.pyplot.xticks')
#     mock_boxplot = mocker.patch('seaborn.boxplot')

#     # Prepare data
#     dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq='D')
#     values = np.random.rand(len(dates))
#     df = pd.DataFrame({'Date': dates, 'Value': values})

#     # Instantiate and fit
#     plotter = TimeSeriesPlotter()
#     plotter.fit(data=df, date_col='Date', value_col='Value')

#     # Test
#     plotter.plotBox()

#     mock_boxplot.assert_called_once()
#     # plt.xticks.assert_called_once_with(rotation=45)
#     plt.show.assert_called_once()

# def test_plotArea(mocker):
#     fig = plt.figure()  # Create a real figure object
#     mocker.patch('matplotlib.pyplot.figure', return_value=fig)
#     mocker.patch('matplotlib.pyplot.show')
#     mocker.patch('matplotlib.pyplot.fill_between')

#     # Setup DataFrame
#     dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq='D')
#     values = np.random.rand(len(dates))
#     df = pd.DataFrame({'Date': dates, 'Value': values})

#     # Instantiate and fit
#     plotter = TimeSeriesPlotter()
#     plotter.fit(data=df, date_col='Date', value_col='Value')

#     # Test
#     plotter.plotArea()

#     #mock_fill_between.assert_called_once()
#     plt.show.assert_called_once()


# def test_heatmapCorrelation(mocker):
#     fig = plt.figure()  # Create a real figure object
#     mocker.patch('matplotlib.pyplot.figure', return_value=fig)
#     mocker.patch('matplotlib.pyplot.show')
#     mock_heatmap = mocker.patch('seaborn.heatmap')

#     # Prepare data
#     df = pd.DataFrame(np.random.rand(10, 10), columns=[f"col{i}" for i in range(10)])

#     # Instantiate and fit
#     plotter = TimeSeriesPlotter()
#     plotter.fit(data=df, date_col=None, value_col='col0')

#     # Test
#     plotter.heatmapCorrelation()

#     mock_heatmap.assert_called_once()
#     plt.show.assert_called_once()


if __name__== '__main__': 
    pytest.main( [__file__])