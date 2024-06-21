# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Time-Series Plots  
"""

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns

from ..api.formatter import format_iterable 
from ..api.property import BasePlot 
from ..exceptions import NotFittedError 
from ..tools.baseutils import smart_rotation  
from ..tools.coreutils import format_to_datetime
from ..tools.funcutils import ensure_pkg 
from ..tools.validator import is_time_series, build_data_if 

class TimeSeriesPlotter(BasePlot):
    """
    A versatile plotting class for time series data, built to extend the 
    BasePlot class with additional time series-specific functionalities.

    This class provides a robust framework for creating various types of plots 
    specifically tailored for time series analysis. It manages plot aesthetics 
    and  maintains consistency across different types of visualizations.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        An instance of Axes that the plots will be drawn on. If None, a new Axes 
        instance will be created whenever needed. Providing an Axes object allows 
        for plots to be added to existing figures, useful for subplots and more 
        complex layouts.
    figsize : tuple, default (10, 6)
        The dimensions for the figure as `(width, height)` in inches. This setting 
        dictates the initial size of the plot, influencing readability and space 
        for displaying data and annotations.
    fontsize : int, default 12
        The base font size for text elements in the plot. Adjusting this value 
        affects the size of labels, titles, tick marks, and other textual components, 
        which can enhance clarity and aesthetics depending on the visualization's 
        complexity and context.
    **kws : dict
        Additional keyword arguments passed to the BasePlot initializer. These can 
        include settings for style, context, palette, and other configurations that 
        are handled by the BasePlot class.
    """
    def __init__(self, ax=None, figsize=(10, 6), fontsize=12, **kws):
        super().__init__(fig_size=figsize, font_size=fontsize, **kws)
        self.ax = ax
        self.figsize = figsize
        self.fontsize = fontsize 

    def fit(self, data, date_col=None, value_col=None, **fit_params):
        """
        Fit the TimeSeriesPlotter with a time series dataset, preparing it for
        further plotting and analysis. The method adjusts the DataFrame to
        ensure that date-related data is recognized and properly formatted
        as datetime objects within pandas.
    
        Parameters
        ----------
        data : pandas.DataFrame
            The time series dataset to be processed.
        date_col : str, optional
            The name of the column in `data` that contains the date/time
            information. If `None`, the DataFrame's index is checked and
            converted to a date column, if it is not already a datetime type.
        value_col : str, optional
            The name of the column in `data` that contains the values to
            be analyzed. If `None`, the first column (excluding the date column)
            is selected by default.
        **fit_params : dict
            Additional keyword arguments for internal configuration or processing
            functions, such as `build_data_if`.
    
        Returns
        -------
        self : TimeSeriesPlotter
            The instance of the plotter, configured and ready for further
            operations like plotting.
    
        Raises
        ------
        ValueError
            If the index cannot be converted to datetime or if the specified
            `date_col` does not exist or cannot be converted to datetime.
    
        Examples
        --------
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
        >>> values = np.random.rand(len(dates))
        >>> df = pd.DataFrame({'Date': dates, 'Value': values})
        >>> plotter = TimeSeriesPlotter()
        >>> plotter.fit(df, 'Date', 'Value')
        >>> plotter.plotLine()
    
        # Example with DataFrame index as date
        >>> df = pd.DataFrame(values, index=pd.date_range("2020-01-01", periods=12, freq='M'))
        >>> plotter.fit(df)
        >>> plotter.plotHistogram()
    
        Notes
        -----
        - If `date_col` is `None` and the DataFrame's index is not a datetime type,
          an attempt will be made to convert it to datetime and use it as a new
          date column in the DataFrame. If the index has no name, it will be named 'date'.
        - If `date_col` is provided but the column is not in datetime format,
          an attempt will be made to convert this column to datetime using
          `format_to_datetime`.
        - This method does not handle time zones or other more complex datetime
          conversions that may be necessary for certain time series analyses.
          Users should preprocess their data accordingly in such cases.
        """

        columns = fit_params.pop("columns", None)
        data = build_data_if(data, columns=columns, to_frame=True, force=True,
                             input_name='ts', raise_warning="silence")
        if data.empty:
            raise ValueError(
                "The DataFrame is empty. Please provide a DataFrame with data."
                )
        # Handling when date_col is None and ensuring the DataFrame's index is datetime
        if date_col is None:
            if not pd.api.types.is_datetime64_any_dtype(data.index):
                try:
                    data.index = pd.to_datetime(data.index)
                    if data.index.name is None:
                        data.index.name = 'date'
                    date_col = data.index.name
                    data.reset_index(level=0, inplace=True)
                except Exception as e:
                    raise ValueError(
                        "'date_col' is None and the DataFrame index is not in"
                        " datetime format. Ensure that the DataFrame index is"
                        " a datetime index before calling 'fit', or provide a"
                        " 'date_col' name that exists in the DataFrame and can"
                        " be converted to datetime format. Error converting"
                        f" index to datetime: {e}")
            else:
                data.index.name = data.index.name or 'date'
                date_col = data.index.name 
                data.reset_index(level=0, inplace=True)
   
        # Convert date_col to datetime if not already done
        if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
            data = format_to_datetime(data, date_col)
    
        # Verify if data is time series
        is_time_series(data, time_col=date_col)
        
        # Set the class attributes
        self.data = data
        self.date_col = date_col
        self.value_col = value_col or data.columns[data.columns != date_col][0]
        
        return self
    
    def plotRollingMeanStd(
        self, 
        window=12, 
        mean_color='blue',
        std_color='red', 
        figsize=None, 
        title=None,
        xlabel=None, 
        ylabel=None, 
        rotation=0,
        **plot_kws
        ):
        """
        Generates plots for rolling mean and standard deviation, providing insights
        into trends and variability over time in the time series data.
    
        Parameters
        ----------
        window : int, default 12
            Size of the moving window. This defines the number of observations used
            for calculating the statistic.
        mean_color : str, default 'blue'
            Color for the rolling mean line. Allows customization of the plot's 
            appearance to improve readability or thematic styling.
        std_color : str, default 'red'
            Color for the rolling standard deviation line. Different colors can help
            distinguish between the mean and standard deviation lines.
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches. If not
            provided, defaults to the class attribute `figsize`.
        title : str, default 'Rolling Mean & Standard Deviation'
            The title of the plot. Provides context and description for the statistical
            visualization.
        xlabel : str, optional
            The label for the x-axis. If not provided, defaults to 'Date'.
        ylabel : str, optional
            The label for the y-axis. If not provided, defaults to 'Value'.
        rotation : int, default 0
            The rotation angle (in degrees) for x-axis labels, enhancing label
            readability.
        **plot_kws : dict
            Additional keyword arguments to pass to `plt.plot` for further 
            customization of the rolling plots.
    
        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.DataFrame({
        ...     'Date': pd.date_range(start='2021-01-01', periods=100),
        ...     'Value': np.random.randn(100).cumsum()
        ... })
        >>> plotter = TimeSeriesPlotter() 
        >>> plotter.fit(data=data, date_col='Date', value_col='Value')
        >>> plotter.plotRollingMeanStd(window=30, title='30-Day Rolling Mean & Std Dev')
    
        Notes
        -----
        - Rolling statistics are crucial in time series analysis to understand trends
          and measure volatility over a specified window of time.
        """
        self.inspect
        self._set_plot_style()
    
        # Create figure and axis if not provided
        fig, ax = self._get_figure_and_axis(figsize=figsize)
    
        # Calculate rolling mean and standard deviation
        rolmean = self.data[self.value_col].rolling(window=window).mean()
        rolstd = self.data[self.value_col].rolling(window=window).std()
    
        # Generate the rolling mean and standard deviation plots
        ax.plot(self.data[self.date_col], rolmean, label='Rolling Mean',
                color=mean_color, **plot_kws)
        ax.plot(self.data[self.date_col], rolstd, label='Rolling Std',
                color=std_color, **plot_kws)
    
        # Set the plot title and configure additional aesthetics
        ax.set_title(title or 'Rolling Mean & Standard Deviation', 
                     fontsize=self.fontsize + 2)
    
        # Configure grid settings and labels
        self._configure_grid(ax)
        self._configure_axes_and_labels(ax, xlabel or 'Date',
                                        ylabel or 'Value', rotation)
    
        plt.legend()
        plt.show()

    def plotAutoCorrelation(
        self, 
        figsize=None,
        title=None, 
        xlabel=None, 
        ylabel=None, 
        rotation=0,
        **plot_kws
        ):
        """
        Generates an autocorrelation plot of the time series data. Autocorrelation 
        plots are used to visualize how the data points in the time series are 
        related to their preceding points.
    
        Parameters
        ----------
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches. If not
            provided, defaults to the class attribute `figsize`.
        title : str, default 'Autocorrelation Plot'
            The title of the plot. Provides context and description for the 
            autocorrelation plot.
        xlabel : str, optional
            The label for the x-axis. If not provided, defaults to 'Lags'.
        ylabel : str, optional
            The label for the y-axis. If not provided, defaults to 'Autocorrelation'.
        rotation : int, default 0
            The rotation angle (in degrees) for x-axis labels, enhancing label
            readability.
        **plot_kws : dict
            Additional keyword arguments to pass to `plt.autocorrelation_plot` for 
            further customization of the autocorrelation plot.
    
        Examples
        --------
        >>> import pandas as pd 
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.DataFrame({
        ...     'Value': pd.Series(np.random.randn(100).cumsum())
        ... })
        >>> plotter = TimeSeriesPlotter()
        >>> plotter.fit(data=data, value_col='Value')
        >>> plotter.plotAutoCorrelation(title='Sample Autocorrelation Plot')
    
        Notes
        -----
        - Autocorrelation plots are particularly useful in identifying patterns in 
          time series data, such as seasonality or other forms of serial correlation.
        """
        self.inspect
        self._set_plot_style()
    
        # Create figure and axis if not provided
        fig, ax = self._get_figure_and_axis(figsize=figsize)
    
        # Generate the autocorrelation plot
        pd.plotting.autocorrelation_plot(self.data[self.value_col], ax=ax, **plot_kws)
    
        # Set the plot title and configure additional aesthetics
        ax.set_title(title or 'Autocorrelation Plot', fontsize=self.fontsize + 2)
    
        # Configure grid settings and labels
        self._configure_grid(ax)
        self._configure_axes_and_labels(
            ax, xlabel or 'Lags', ylabel or 'Autocorrelation', rotation)
    
        plt.show()

    @ensure_pkg ("statsmodels", extra = ( 
        "'plotPACF' expects statsmodels to be installed."
        )
    )
    def plotPACF(
        self, lags=15, 
        figsize=None, 
        title=None, 
        **plot_pacf_kws
        ):
        """
        Generates a partial autocorrelation plot (PACF) for the time series data. 
        This plot is instrumental in time series modeling, especially for determining 
        the order of autoregressive (AR) terms in ARIMA modeling.
    
        Parameters
        ----------
        lags : int, default 15
            Specifies the number of lags to be considered in the PACF plot. This helps 
            in understanding the lagged relationships of the series.
        figsize : tuple, optional
            The dimensions of the figure as `(width, height)` in inches. If not provided, 
            defaults to the class attribute `figsize` or a typical default of (10, 6).
        title : str, optional
            The title of the plot. If None, defaults to 'Partial Autocorrelation Plot'.
        **plot_pacf_kws : dict
            Additional keyword arguments passed to `statsmodels.graphics.tsaplots.plot_pacf`. 
            This allows for further customization of the plot, such as adjusting the confidence 
            intervals or the style of the plot.
    
        Examples
        --------
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.Series([0.25, 0.5, 0.1, -0.3, 0.0, 0.2, 0.3])
        >>> plotter = TimeSeriesPlotter(data=pd.DataFrame({'Value': data}))
        >>> plotter.plotPACF(lags=10, figsize=(12, 8), title='My PACF Plot')
    
        Notes
        -----
        - PACF plots are essential for identifying how many past points (lags) in 
          the series significantly predict future values.
        - This function leverages the statistical calculations from the `statsmodels` 
          library, which should be installed and available in the environment.
        - The method dynamically creates a figure and axes if not already provided via 
          class attributes, ensuring that the plot is properly sized and displayed.
        - Grid management is configured to visually enhance the plot, making it easier 
          to identify significant lags.
        """
        from statsmodels.graphics.tsaplots import plot_pacf
        
        self.inspect 
        # Create figure and axis if not provided
        fig, ax = self._get_figure_and_axis(figsize)
        
        # Generate the PACF plot
        plot_pacf(self.data[self.value_col], lags=lags, ax=ax, **plot_pacf_kws)
        
        # Setting the plot title
        ax.set_title(title or 'Partial Autocorrelation Plot', 
                     fontsize=self.fontsize + 2)
        
        # Configure grid settings
        self._configure_grid(ax)
        
        plt.show()


    @ensure_pkg ( "statsmodels", extra =(
        "'statsmodels' library is expected for 'plotDecomposition' to be feasible.")
        )
    def plotDecomposition(
        self,
        model='additive', 
        freq=12, 
        figsize=None,
        title=None, 
        **decompose_kws
        ):
        """
        Generates a decomposition plot of the time series data, breaking down
        the data into its trend, seasonal, and residual components. This analysis
        helps in understanding the underlying patterns such as seasonality and trends
        in the dataset.
    
        Parameters
        ----------
        model : str, default 'additive'
            Specifies the type of decomposition model to use. Options include:
            - 'additive': suitable when seasonal variations are roughly constant
              throughout the series.
            - 'multiplicative': suitable when seasonal variations change
              proportionally over time.
        freq : int, default 12
            Defines the frequency of the cycle in the time series (e.g., 12 for
            monthly data with an annual cycle).
        figsize : tuple, optional
            The dimensions of the figure as `(width, height)` in inches. If not
            provided, defaults to the class attribute `figsize`.
        title : str, optional
            The title of the plot. If not provided, defaults to 
            'Time Series Decomposition'.
    
        Examples
        --------
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.Series([1, 2, 3, 4, 5, 6] * 4,
                             index=pd.date_range(start="2020-01-01", periods=24, freq='M'))
        >>> plotter = TimeSeriesPlotter() 
        >>> plotter.fit(data=pd.DataFrame({'Value': data}), date_col='index')
        >>> plotter.plotDecomposition(model='additive', freq=12, 
                                      title='Monthly Data Decomposition')
    
        Notes
        -----
        - This method is particularly useful in identifying whether the underlying
          data is influenced more by seasonal or trend factors.
        - It is critical that the `freq` parameter correctly matches the periodicity
          of the data's cycle for accurate decomposition.
        - The decomposition is performed using the `seasonal_decompose` method from
          the `statsmodels` library, which must be installed in your environment.
        """
        self.inspect 
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Ensure data is appropriately indexed by date
        if not self.data.index.equals(pd.to_datetime(self.data[self.date_col])):
            self.data.set_index(pd.to_datetime(self.data[self.date_col]), inplace=True)
        
        result = seasonal_decompose(self.data[self.value_col], model=model, period=freq, 
                                    **decompose_kws)
        # Plot the decomposition results
        result.plot()
        plt.gcf().set_size_inches(figsize or self.figsize)
        plt.suptitle(title or 'Time Series Decomposition', fontsize=self.fontsize + 2)
        
        plt.show()
        
    def _set_plot_style(self):
        """
        Configures the default plot style for all visualizations created with 
        this plotting class. This method sets a consistent background grid and 
        adjusts the default size and font settings of the plots based on the 
        instance's configuration.
    
        This method ensures that all plots share a uniform look and feel, 
        which is crucial for producing professional, publication-quality figures.
    
        Notes
        -----
        - This method leverages seaborn's style setting and matplotlib's rcParams
          to ensure consistency. It sets the background to 'whitegrid', which is 
          suitable for most statistical plots by providing a clear delineation of 
          scale and context for the data visualized.
        - The figure size (`figure.figsize`) and font size (`font.size`) are set 
          according to the instance's attributes, allowing customization at the 
          instantiation of the plotter object. If not explicitly set during the
          instantiation, it defaults to [10, 6] for `figsize` and 12 for `fontsize`.
    
        Examples
        --------
        >>> plotter = TimeSeriesPlotter(figsize=(12, 8), fontsize=14)
        >>> plotter._set_plot_style()
        >>> # After setting the style, all subsequent plots will automatically
        >>> # use these aesthetic settings.
        """
        # Set the background grid style
        sns.set_style(self.sns_style or "whitegrid")
    
        # Apply figure size and font size settings from 
        # instance attributes or use defaults
        plt.rcParams['figure.figsize'] = self.figsize or [10, 6]
        plt.rcParams['font.size'] = self.fontsize or 12

    def plotCumulativeLine(
        self, 
        figsize=None,
        color='blue', 
        title=None,
        xlabel=None, 
        ylabel=None, 
        rotation=0, 
        **plot_kws
        ):
        """
        Generates a cumulative line plot of the time series data. This type of plot
        is useful for visualizing the accumulation of values over time, providing
        insight into the total growth or decline across the series.
    
        Parameters
        ----------
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches. If not
            provided, defaults to the class attribute `figsize`.
        color : str, default 'blue'
            The color of the line in the plot. Allows customization of the line color.
        title : str, default 'Cumulative Line Plot'
            The title of the plot. Provides context and description for the plot.
        xlabel : str, optional
            The label for the x-axis. If not provided, defaults to the class attribute
            `date_col` or a generic label if `date_col` is not set.
        ylabel : str, optional
            The label for the y-axis. If not provided, defaults to 'Cumulative Value'
            or the class attribute `value_col` if set.
        rotation : int, default 0
            The rotation angle (in degrees) for x-axis labels. Useful for improving
            readability of date labels or other categorical labels.
        **plot_kws : dict
            Additional keyword arguments to pass to the `ax.plot` function for
            further customization of the line plot.
    
        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.DataFrame({
        ...     'Date': pd.date_range(start='2020-01-01', periods=100),
        ...     'Value': np.random.rand(100).cumsum()
        ... })
        >>> plotter = TimeSeriesPlotter()
        >>> plotter.fit(data=data, date_col='Date', value_col='Value')
        >>> plotter.plotCumulativeLine(title='Total Growth Over Time', color='green')
    
        Notes
        -----
        This method visually emphasizes the progression or accumulation of values,
        making it particularly useful for time series data where tracking changes
        over time provides valuable insights into trends and patterns.
        """
        self.inspect
        self._set_plot_style()
    
        # Create figure and axis if not provided
        fig, ax = self._get_figure_and_axis(figsize=figsize)
    
        # Plot the cumulative sum of the value column
        ax.plot(self.data[self.date_col], self.data[self.value_col].cumsum(),
                color=color or 'Cumulative Line Plot', **plot_kws)
    
        # Set the plot title and configure grid and labels
        ax.set_title(title, fontsize=self.fontsize + 2)
        self._configure_grid(ax)
        self._configure_axes_and_labels(ax, xlabel, ylabel or 'Cumulative Value', rotation)
    
        plt.show()
        
    def plotDensity(
        self, 
        figsize=None, 
        color='green', 
        title=None, 
        xlabel=None,
        ylabel=None,
        rotation=0,
        **kde_kws
        ):
        """
        Generates a density plot (Kernel Density Estimate, KDE) of the time series data.
        This plot provides a smooth estimate of the data distribution, which is useful 
        for visualizing the underlying distribution of data points without the need 
        for predefined bins as in histograms.
    
        Parameters
        ----------
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches. If not
            provided, defaults to the class attribute `figsize`.
            
        color : str, default 'green'
            Specifies the color of the density line in the plot. This allows customization
            of the plot appearance to enhance readability or aesthetic preference.
        title : str, default 'Density Plot'
            The title of the plot. Provides context and description for the density plot.
        xlabel : str, optional
            The label for the x-axis. If not provided, defaults to a generic label 'Value'.
        ylabel : str, optional
            The label for the y-axis. If not provided, defaults to 'Density'.
        rotation : int, default 0
            The rotation angle (in degrees) for x-axis labels, enhancing label readability.
        **kde_kws : dict
            Additional keyword arguments to pass to `sns.kdeplot` for further customization
            of the density plot. This can include parameters such as `bw_adjust` or `fill`.
    
        Examples
        --------
        >>> import numpy as np 
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.DataFrame({
        ...     'Value': np.random.randn(100)
        ... })
        >>> plotter = TimeSeriesPlotter().fit(data=data, value_col='Value')
        >>> plotter.plotDensity(color='blue', title='Sample Density Plot')
    
        Notes
        -----
        - KDE plots are particularly useful in data analysis for identifying the shape
          of the distribution and potential outliers.
        - Ensure that the `value_col` is properly set in your TimeSeriesPlotter instance
          as this method relies on that column to generate the density plot.
        """
        self.inspect
        self._set_plot_style()
    
        # Create figure and axis if not provided
        fig, ax = self._get_figure_and_axis(figsize=figsize )
    
        # Generate the density plot
        sns.kdeplot(self.data[self.value_col], color=color, shade=True, ax=ax, **kde_kws)
    
        # Set the plot title and configure additional aesthetics
        ax.set_title(title or 'Density Plot', fontsize=self.fontsize + 2)
    
        # Configure grid settings and labels
        self._configure_grid(ax)
        self._configure_axes_and_labels(
            ax, xlabel or 'Value', ylabel or 'Density', rotation)
    
        plt.show()

    def plotScatterWithTrendline(
        self, 
        figsize=None, 
        color='red', 
        title=None, 
        xlabel=None, 
        ylabel=None, 
        rotation=0, 
        **reg_kws
        ):
        """
        Generates a scatter plot with a trendline of the time series data, using
        seaborn's regplot function. This plot is useful for visualizing the 
        relationship between two variables and assessing the linear trend 
        directly from the scatter plot.
    
        Parameters
        ----------
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches. If not
            provided, defaults to the class attribute `figsize`.
        color : str, default 'red'
            Specifies the color of the scatter points in the plot. This allows
            customization of the plot appearance to enhance readability or 
            aesthetic preference.
        title : str, default 'Scatter Plot with Trendline'
            The title of the plot. Provides context and description for the 
            scatter plot.
        xlabel : str, optional
            The label for the x-axis. If not provided, defaults to the class 
            attribute `date_col`.
        ylabel : str, optional
            The label for the y-axis. If not provided, defaults to the class 
            attribute `value_col`.
        rotation : int, default 0
            The rotation angle (in degrees) for x-axis labels, enhancing label
            readability.
        **reg_kws : dict
            Additional keyword arguments to pass to `sns.regplot` for further 
            customization of the scatter plot. This can include parameters 
            such as `fit_reg` to show or hide the regression line, and 
            `scatter_kws` to modify aspects of the scatter points.
    
        Examples
        --------
        >>> import numpy as np 
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.DataFrame({
        ...     'Date': pd.date_range(start='2020-01-01', periods=100),
        ...     'Value': np.random.rand(100).cumsum()
        ... })
        >>> plotter = TimeSeriesPlotter() 
        >>> plotter.fit(data=data, date_col='Date', value_col='Value')
        >>> plotter.plotScatterWithTrendline(figsize=(10, 6), title='Growth Trend Analysis')
    
        Notes
        -----
        - This method visually emphasizes the linear relationships between data points,
          making it particularly useful for initial explorations of potential linear 
          patterns or correlations in time series data.
        """
        self.inspect
        self._set_plot_style()
    
        # Create figure and axis if not provided
        fig, ax = self._get_figure_and_axis(figsize=figsize)
        
        # Check if the date column is of datetime type
        if pd.api.types.is_datetime64_any_dtype(self.data[self.date_col]):
            # Convert the datetime data to numeric data for regression analysis
            numeric_dates = mdates.date2num(self.data[self.date_col])
            x_values = numeric_dates
        else:
            x_values = self.data[self.date_col]
        
        # Generate the scatter plot with a regression trendline
        sns.regplot(x=x_values, y=self.value_col, data=self.data,
                    color=color, ax=ax, **reg_kws)
       
        # Set the plot title and configure additional aesthetics
        ax.set_title(title or 'Scatter Plot with Trendline', 
                     fontsize=self.fontsize + 2)
        
        if pd.api.types.is_datetime64_any_dtype(self.data[self.date_col]):
            # Configure the x-axis to show date labels instead of numeric values
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        # Configure grid settings and labels
        self._configure_grid(ax)
        self._configure_axes_and_labels(
            ax, xlabel or 'Date', ylabel or 'Value', rotation)
    
        plt.show()
        
    def plotBar(
        self, 
        figsize=None, 
        color='cyan', 
        title='Bar Plot', 
        xlabel=None, 
        ylabel=None, 
        rotation=0, 
        **plot_kws
        ):
        """
        Generates a bar plot of the time series data. This type of plot is 
        useful for visualizing the discrete values in the series across 
        different categories or times.
    
        Parameters
        ----------
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches. If not
            provided, defaults to the class attribute `figsize`.
        color : str, default 'cyan'
            Specifies the color of the bars in the plot. Allows customization of the plot's
            appearance to enhance readability or aesthetic preference.
        title : str, default 'Bar Plot'
            The title of the plot. Provides context and description for the bar plot.
        xlabel : str, optional
            The label for the x-axis. If not provided, defaults to the class attribute
            `date_col`.
        ylabel : str, optional
            The label for the y-axis. If not provided, defaults to 'Value'.
        rotation : int, default 0
            The rotation angle (in degrees) for x-axis labels, enhancing label 
            readability.
        **plot_kws : dict
            Additional keyword arguments to pass to `dataframe.plot` for further 
            customization of the plot. 
        Examples
        --------
        >>> import numpy as np 
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.DataFrame({
        ...     'Date': pd.date_range(start='2020-01-01', periods=10),
        ...     'Value': [1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
        ... })
        >>> plotter = TimeSeriesPlotter()
        >>> plotter.fit(data=data, date_col='Date', value_col='Value')
        >>> plotter.plotBar(title='Monthly Data Bar Plot', color='blue')
    
        Notes
        -----
        - This method is effective for showing the variation of data across different
          categories or time points, making it easier to spot trends and outliers.
        """
        self.inspect
        self._set_plot_style()
    
        # Create figure and axis if not provided
        fig, ax = self._get_figure_and_axis(figsize=figsize)
    
        # Generate the bar plot
        self.data.plot(kind='bar', x=self.date_col, y=self.value_col, color=color, 
                       ax=ax, **plot_kws)
    
        # Set the plot title and configure additional aesthetics
        ax.set_title(title, fontsize=self.fontsize + 2)
    
        # Configure grid settings and labels
        self._configure_grid(ax)
        self._configure_axes_and_labels(ax, xlabel or 'Date',
                                        ylabel or 'Value', rotation)
        plt.show()
        

    def _get_plotting_dates(self, ax):
        """
        Prepares the date column for plotting by checking if it is a datetime 
        type and converting it to numeric format if necessary. It also configures
        the x-axis of the plot to display appropriate date labels if the dates
        are datetime objects.
    
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The Axes object where the plot will be drawn.
    
        Returns
        -------
        x_values : array_like
            The x-values to be used for plotting. These will either be numeric 
            representations of the datetime data or the original data if it 
            is not of datetime type.
    
        Examples
        --------
        >>> fig, ax = plt.subplots()
        >>> x_values = self.get_plotting_dates(ax)
        >>> ax.plot(x_values, self.data[self.value_col], color='red')  # Example use in a plot
    
        Notes
        -----
        - This method is particularly useful when plotting time series data 
          where the x-axis is expected to represent dates. It ensures the data 
          is in the correct format for regression or other numerical analyses 
          and configures the plot to display dates in a human-readable format.
        """
        # Check if the date column is of datetime type and prepare x_values accordingly
        if pd.api.types.is_datetime64_any_dtype(self.data[self.date_col]):
            # Convert the datetime data to numeric data for regression analysis
            numeric_dates = mdates.date2num(self.data[self.date_col])
            x_values = numeric_dates
    
            # Configure the x-axis to show date labels instead of numeric values
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        else:
            x_values = self.data[self.date_col]
    
        return x_values

    def plotStackedArea(
        self, 
        figsize=None, 
        color=None,  
        title=None, 
        xlabel=None,
        ylabel=None,
        rotation=0, 
        **stack_kws
        ):
        """
        Generates a stacked area plot of the time series data. This type of plot 
        is useful for showing the contribution of various components to the whole 
        over time.
    
        Parameters
        ----------
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches. If not
            provided, defaults to the class attribute `figsize`.
        color : list or tuple, optional
            A list or tuple of colors for the areas under the plot, enhancing the
            visualization's readability or aesthetic preference. If not provided,
            defaults to a set of matplotlib's default colors.
        title : str, default 'Stacked Area Plot'
            The title of the plot. Provides context and description for the area plot.
        xlabel : str, optional
            The label for the x-axis. If not provided, defaults to the class attribute
            `date_col`.
        ylabel : str, optional
            The label for the y-axis. If not provided, defaults to 'Value'.
        rotation : int, default 0
            The rotation angle (in degrees) for x-axis labels, enhancing label readability.
    
        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.DataFrame({
        ...     'Date': pd.date_range(start='2021-01-01', periods=100),
        ...     'Value': np.random.rand(100).cumsum()
        ... })
        >>> plotter = TimeSeriesPlotter () 
        >>> plotter.fit(data=data, date_col='Date', value_col='Value')
        >>> plotter.plotStackedArea(title='Cumulative Growth Over Time')
    
        Notes
        -----
        - Stacked area plots are effective for visualizing the cumulative contribution
          of multiple data series over time, emphasizing the total growth or composition.
        """
        self.inspect
        self._set_plot_style()
    
        # Create figure and axis if not provided
        fig, ax = self._get_figure_and_axis(figsize=figsize)
    
        # Generate the stacked area plot
        ax.stackplot(self.data[self.date_col], self.data[self.value_col],
                     colors=color, **stack_kws)
    
        # Set the plot title and configure additional aesthetics
        ax.set_title(title or 'Stacked Area Plot', fontsize=self.fontsize + 2)
    
        # Configure grid settings and labels
        self._configure_grid(ax)
        self._configure_axes_and_labels(
            ax, xlabel or 'Date', ylabel or 'Value', rotation)
    
        plt.show()
            
    def plotViolin(
        self, 
        figsize=None, 
        color='purple', 
        title='Violin Plot', 
        xlabel=None,
        ylabel=None,
        rotation=45,
        **violin_kws
        ):
        """
        Generates a violin plot of the time series data. This plot type is 
        particularly useful for visualizing the distribution of data across 
        different categories and comparing the frequency distribution between
        groups.
    
        Parameters
        ----------
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches. If not
            provided, defaults to the class attribute `figsize`.
        color : str, default 'purple'
            Specifies the color of the violin in the plot. This allows 
            customization of the plot appearance to enhance readability or 
            aesthetic preference.
        title : str, default 'Violin Plot'
            The title of the plot. Provides context and description for the 
            violin plot.
        xlabel : str, optional
            The label for the x-axis. If not provided, defaults to the class 
            attribute `date_col`.
        ylabel : str, optional
            The label for the y-axis. If not provided, defaults to 'Value'.
        rotation : int, default 45
            The rotation angle (in degrees) for x-axis labels, enhancing label
            readability.
        **violin_kws : dict
            Additional keyword arguments to pass to `sns.violinplot` for further 
            customization  of the violin plot.
    
        Examples
        --------
        >>> import numpy as np 
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.DataFrame({
        ...     'Date': pd.date_range(start='2021-01-01', periods=10),
        ...     'Value': np.random.rand(10)
        ... })
        >>> plotter = TimeSeriesPlotter()
        >>> plotter.fit(data=data, date_col='Date', value_col='Value')
        >>> plotter.plotViolin(title='Distribution of Values')
    
        Notes
        -----
        - Violin plots are effective for showing the full distribution of the data,
          including peaks, median, and outliers.
        """
        self.inspect
        self._set_plot_style()
    
        # Create figure and axis if not provided
        fig, ax = self._get_figure_and_axis(figsize=figsize)
    
        # Generate the violin plot
        sns.violinplot(x=self.data[self.date_col], 
                       y=self.data[self.value_col], 
                       color=color,
                       ax=ax, 
                       **violin_kws)
    
        # Set the plot title and configure additional aesthetics
        ax.set_title(title, fontsize=self.fontsize + 2)
    
        # Configure grid settings and labels
        self._configure_grid(ax)
        self._configure_axes_and_labels(
            ax, xlabel or 'Date', ylabel or 'Value', rotation)
    
        plt.show()

    def pieChart(
        self, 
        figsize=None, 
        title='Pie Chart', 
        **plot_kws
        ):
        """
        Generates a pie chart of the time series data. This type of chart is 
        particularly useful for showing the proportional contributions of 
        different categories within the data.
    
        Parameters
        ----------
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches. If not
            provided, defaults to the class attribute `figsize`.
        title : str, default 'Pie Chart'
            The title of the chart. Provides context and description for the 
            pie chart.
        **plot_kws : dict
            Additional keyword arguments to pass to `DataFrame.plot.pie` for 
            further customization of the pie chart. This can include parameters
            like `startangle`, `shadow`, and `explode`.
    
        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.DataFrame({
        ...     'Category': ['A', 'B', 'A', 'C', 'B', 'A'],
        ...     'Value': [1, 2, 3, 4, 2, 1]
        ... })
        >>> plotter = TimeSeriesPlotter() 
        >>> plotter.fit(data=data, value_col='Category')
        >>> plotter.pieChart(title='Category Distribution')
    
        Notes
        -----
        - Pie charts are best used with categorical data to visualize the 
          percentage share of each category in the whole dataset, helping in 
          quick comparison and insight extraction about the data distribution.
        """
        self.inspect
        self._set_plot_style()
    
        # Create figure and axis if not provided
        fig, ax = self._get_figure_and_axis(figsize=figsize)
    
        # Generate the pie chart
        self.data[self.value_col].value_counts().plot(
            kind='pie', 
            autopct='%1.1f%%', 
            ax=ax,
            **plot_kws
        )
    
        # Set the plot title and remove the y-axis label for aesthetics
        ax.set_title(title, fontsize=self.fontsize + 2)
        ax.set_ylabel('')  # Remove the y-axis label for clarity in pie charts
    
        plt.show()

    def plotLine(
        self, 
        figsize=None, 
        title=None,
        xlabel=None, 
        ylabel=None, 
        color=None, 
        **lineplot_kws
        ):
        """
        Generates a line plot of the time series data using seaborn's lineplot
        function. This method automatically adjusts for overlapping x-axis labels
        and can be customized extensively via additional keyword arguments.
    
        Parameters
        ----------
        figsize : tuple, optional
            The size of the figure to create. If None, defaults to matplotlib's
            figure size.
        title : str, optional
            The title of the plot. If None, defaults to 'Time Series Line Plot'.
        xlabel : str, optional
            The label for the x-axis. If None, defaults to 'Date'.
        ylabel : str, optional
            The label for the y-axis. If None, defaults to 'Value'.
        color : str, optional
            The color of the line in the plot. If None, defaults to 'blue'.
        **lineplot_kws : dict
            Additional keyword arguments to be passed to seaborn.lineplot.
    
        Raises
        ------
        ValueError
            If the DataFrame is empty or does not contain the specified `date_col`
            or `value_col`.
    
        Examples
        --------
        >>> import numpy as np 
        >>> import pandas as pd 
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> dates = pd.date_range(start="2020-01-01", periods=100, freq='D')
        >>> values = np.random.rand(100)
        >>> df = pd.DataFrame({'Date': dates, 'Value': values})
        >>> plotter = TimeSeriesPlotter().fit(df, 'Date', 'Value')
        >>> plotter.plotLine(title='Random Time Series', color='green')
    
        Notes
        -----
        This method is part of a class that assumes `data`, `date_col`, and `value_col`
        have been predefined as attributes of the instance. The `smart_rotation` function
        is called to handle overlapping x-axis labels automatically, enhancing the 
        readability of the plot, especially when dealing with dense date labels.
        """
        self.inspect 
        if self.data.empty:
            raise ValueError("The DataFrame is empty. Please provide a "
                             "DataFrame with data.")
    
        if self.date_col not in self.data.columns or self.value_col not in self.data.columns:
            raise ValueError(f"DataFrame must contain specified columns"
                             f" '{self.date_col}' and '{self.value_col}'")
    
        if self.ax is None: 
            fig, ax = plt.subplots(figsize=figsize or self.figsize )
        else:
            ax = self.ax
            
        sns.lineplot(data=self.data, x=self.date_col, y=self.value_col,
                     ax=ax, color=color or 'blue', **lineplot_kws)
        ax.set_title(title or 'Time Series Line Plot')
        # Apply smart rotation to adjust tick labels dynamically
        self._configure_axes_and_labels(ax, xlabel, ylabel )

    def plotHistogram(
        self, 
        figsize=None, 
        title=None,
        xlabel=None, 
        ylabel=None, 
        bins=30, 
        **hist_kws
        ):
        """
        Generates a histogram of the time series data using seaborn's histplot
        function. This method allows customization of figure size, labels, title,
        and the number of bins, as well as additional seaborn histplot keyword
        arguments.
    
        Parameters
        ----------
        figsize : tuple, optional
            The size of the figure to create. If None, defaults to matplotlib's
            figure size.
        title : str, optional
            The title of the plot. If None, defaults to 'Time Series Histogram'.
        xlabel : str, optional
            The label for the x-axis. If None, defaults to 'Value'.
        ylabel : str, optional
            The label for the y-axis. If None, defaults to 'Frequency'.
        bins : int, optional
            Number of bins in the histogram. Default is 30.
        **hist_kws : dict
            Additional keyword arguments to be passed to seaborn.histplot.
    
        Raises
        ------
        ValueError
            If the DataFrame is empty or does not contain the specified `value_col`.
    
        Examples
        --------
        >>> import numpy as np 
        >>> import pandas as pd 
        >>> from gofast.plot.ts import TimeSeriesPlotter 
        >>> values = np.random.randn(100)
        >>> df = pd.DataFrame({'Value': values})
        >>> plotter = TimeSeriesPlotter().fit(df, value_col='Value')
        >>> plotter.plotHistogram(title='Distribution of Values', bins=20, color='red')
    
        Notes
        -----
        This method is part of a class that assumes `data` and `value_col` have been
        predefined as attributes of the instance. It uses seaborn to create the
        histogram and matplotlib to customize the plot aesthetics.
        """
        self.inspect 
        if self.data.empty:
            raise ValueError("The DataFrame is empty. Please provide a DataFrame with data.")
    
        if self.value_col not in self.data.columns:
            raise ValueError(f"DataFrame must contain the specified column '{self.value_col}'")
    
        if self.ax is None: 
            fig, ax = plt.subplots(figsize=figsize or self.figsize )
        else:
            ax = self.ax
            
        sns.histplot(self.data[self.value_col], bins=bins, kde=True, ax=ax, **hist_kws)
        ax.set_title(title or 'Time Series Histogram', fontsize = self.fontsize +2 )
        self._configure_axes_and_labels(ax, xlabel, ylabel )

    def plotBox(
        self, 
        figsize=None, 
        title=None,
        xlabel=None, 
        ylabel=None, 
        rotation=None, 
        **boxplot_kws
        ):
        """
        Generates a box plot of the time series data using seaborn's boxplot
        function. This method allows customization of figure size, labels, title,
        and rotation of x-axis labels, as well as additional seaborn boxplot keyword
        arguments.
    
        Parameters
        ----------
        figsize : tuple, optional
            The size of the figure to create. If None, defaults to matplotlib's
            figure size.
        title : str, optional
            The title of the plot. If None, defaults to 'Time Series Box Plot'.
        xlabel : str, optional
            The label for the x-axis. If None, defaults to the column name
            specified by `date_col`.
        ylabel : str, optional
            The label for the y-axis. If None, defaults to the column name 
            specified by `value_col`.
        rotation : int, optional
            Rotation angle of x-axis labels. Default is 45 degrees.
        **boxplot_kws : dict
            Additional keyword arguments to be passed to seaborn.boxplot.
    
        Raises
        ------
        ValueError
            If the DataFrame is empty or does not contain the specified `date_col`
            or `value_col`.
    
        Examples
        --------
        >>> import numpy as np
        >>> from gofast.plot.ts import TimeSeriesPlotter 
        >>> dates = pd.date_range(start="2020-01-01", periods=10, freq='M')
        >>> values = np.random.rand(10)
        >>> df = pd.DataFrame({'Date': dates, 'Value': values})
        >>> plotter = TimeSeriesPlotter().fit(df, date_col='Date', value_col='Value')
        >>> plotter.plotBox(title='Monthly Value Distribution', xlabel='Month',
                            ylabel='Value')
    
        Notes
        -----
        This method is part of a class that assumes `data`, `date_col`, and `value_col`
        have been predefined as attributes of the instance. It uses seaborn to create
        the box plot and matplotlib to customize the plot aesthetics.
        """
        self.inspect 
        if self.data.empty:
            raise ValueError("The DataFrame is empty. Please provide a DataFrame with data.")
    
        if self.date_col not in self.data.columns or self.value_col not in self.data.columns:
            raise ValueError(f"DataFrame must contain specified columns"
                             f" '{self.date_col}' and '{self.value_col}'")
    
        if self.ax is None: 
            fig, ax = plt.subplots(figsize=figsize or self.figsize )
        else:
            ax = self.ax
 
        sns.boxplot(data=self.data, x=self.date_col, y=self.value_col, ax=ax, 
                    **boxplot_kws )
    
        # Set the title and labels
        ax.set_title(title or 'Time Series Box Plot', fontsize = self.fontsize  +2 )
        ax.set_xlabel(xlabel or self.date_col, fontsize = self.fontsize )
        ax.set_ylabel(ylabel or self.value_col, fontsize = self.fontsize )
    
        # Improve date formatting on x-axis
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        # Optionally apply smart rotation if date labels overlap
        # Adjust the rotation of the x-axis labels if specified
        if rotation is None: 
            smart_rotation(ax ) 
        else:  
            plt.xticks(rotation=rotation)
    
        # Improve layout
        plt.gcf().autofmt_xdate()
    
        plt.show()


    def plotArea(
        self, 
        figsize=None, 
        title=None, 
        xlabel=None, 
        ylabel=None,
        color=None, 
        alpha=None
        ):

        """
        Generates an area plot of the time series data. This plot can be useful
        for displaying the volume beneath a line chart, emphasizing the magnitude
        of change over time.

        Parameters
        ----------
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches.
            Defaults to the value of the class attribute `figsize` if not
            provided.
        title : str, optional
            The title of the plot. Defaults to 'Time Series Area Plot' if 
            not specified.
        xlabel : str, optional
            The label text for the x-axis. Defaults to the class attribute
            `date_col` if not provided.
        ylabel : str, optional
            The label text for the y-axis. Defaults to the class attribute
            `value_col` if not provided.
        color : str, optional
            The fill color for the area under the line. Defaults to 'skyblue'
            if not specified.
        alpha : float, optional
            The transparency level of the fill color. Value should be between
            0 (transparent) and 1 (opaque). Defaults to 0.4 if not provided.

        Raises
        ------
        ValueError
            If `self.data` is empty or if `self.date_col` or `self.value_col`
            are not columns in `self.data`.

        Examples
        --------
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> plotter = TimeSeriesPlotter()
        >>> plotter.fit(data=df, date_col='Date', value_col='Value')
        >>> plotter.plotArea(figsize=(12, 6), title='My Area Plot',
                             xlabel='Time', ylabel='Observations',
                             color='lightgreen', alpha=0.5)

        Notes
        -----
        - The method uses `plt.fill_between` from Matplotlib to generate the
          area plot. 
        - The x-axis labels will be automatically rotated if they are
          determined to overlap, enhancing readability.
        - This method should be used after the `fit` method, which prepares
          `self.data`, `self.date_col`, and `self.value_col` for plotting.
        """
        self.inspect 
        if self.data.empty:
            raise ValueError("The DataFrame is empty. Please provide a DataFrame with data.")
    
        if self.date_col not in self.data.columns or self.value_col not in self.data.columns:
            raise ValueError("DataFrame must contain specified columns"
                             f" '{self.date_col}' and '{self.value_col}'")
    
        # Use self.ax if it has been defined, else create a new axis
        if self.ax is None:
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
        else:
            ax = self.ax
    
        # Plot the area chart
        ax.fill_between(self.data[self.date_col], self.data[self.value_col],
                        color=color or 'skyblue', alpha=alpha or 0.4)
    
        # Set the title and labels with a default font size from self.fontsize
        ax.set_title(title or 'Time Series Area Plot', fontsize=self.fontsize + 2)
        ax.set_xlabel(xlabel or self.date_col, fontsize=self.fontsize)
        ax.set_ylabel(ylabel or self.value_col, fontsize=self.fontsize)
        
        # configure grid 
        self._configure_grid(ax )
        
        # Adjust the rotation of the x-axis labels if specified using 
        # smart_rotation function
        smart_rotation(ax)
        # Improve layout
        plt.gcf().autofmt_xdate()
        plt.show()

    def _configure_axes_and_labels(
            self, ax, xlabel=None, ylabel=None, rotation=None):
        """
        Configures the axes labels, their font size, and label rotation for a
        given Axes object. It applies smart rotation to x-axis labels to prevent
        overlap and improve readability, unless a specific rotation is specified.
    
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The Axes object for which to set labels and properties.
        xlabel : str, optional
            The label for the x-axis. If None, defaults to the class attribute
            `date_col`. The font size is set to the class attribute `fontsize`.
        ylabel : str, optional
            The label for the y-axis. If None, defaults to the class attribute
            `value_col`. The font size is set to the class attribute `fontsize`.
        rotation : int, optional
            The rotation angle (in degrees) for the x-axis tick labels.
            If None, applies smart rotation based on label overlap.
    
        Notes
        -----
        This method should be called within other plotting methods after the
        plot has been created but before it is displayed or saved. It relies
        on the `smart_rotation` function to automatically adjust the x-axis
        label rotation if needed.
        """
        # Set axis labels and their font size
        ax.set_xlabel(xlabel or self.date_col, fontsize=self.fontsize)
        ax.set_ylabel(ylabel or self.value_col, fontsize=self.fontsize)
    
        # Adjust x-axis label rotation
        if rotation is None:
            smart_rotation(ax)
        else:
            plt.xticks(rotation=rotation)
    
        # Improve layout and show plot
        plt.gcf().autofmt_xdate()
        plt.show()
        
    def heatmapCorrelation(
        self, 
        method='pearson', 
        min_periods=1, 
        figsize=None, 
        title=None, 
        cmap=None,
        annot=True, 
        fmt=".2f"
        ):
        """
        Generates a heatmap to visualize the correlation matrix of the dataset's
        numerical features. This method provides a color-coded representation
        that highlights the strength of correlations among variables.
    
        Parameters
        ----------
        method : {'pearson', 'kendall', 'spearman'}, default 'pearson'
            Specifies the method of correlation:
            - 'pearson' : standard correlation coefficient
            - 'kendall' : Kendall Tau correlation coefficient
            - 'spearman' : Spearman rank correlation
        min_periods : int, default 1
            Minimum number of observations required per pair of columns
            to have a valid result. If fewer than `min_periods` non-NA values
            are present the result will be NA.
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches.
            If not provided, defaults to the class attribute `figsize`.
        title : str, optional
            The title of the heatmap. If not specified, defaults to
            'Heatmap of Correlations'.
        cmap : str, optional
            The colormap scheme to use for the heatmap. If None, uses the
            'coolwarm' colormap.
        annot : bool, optional
            Determines whether to annotate each cell with the numeric value.
            Defaults to True.
        fmt : str, optional
            The string formatting code to use when adding annotations.
            Defaults to ".2f", which represents floating-point numbers
            with two decimal places.
    
        Raises
        ------
        ValueError
            If the DataFrame is empty or if `self.data` does not contain
            enough non-NA values to calculate a correlation.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> # Creating a sample DataFrame with random numerical data for the heatmap
        >>> data = {
        ...     'A': np.random.rand(10),
        ...     'B': np.random.rand(10),
        ...     'C': np.random.rand(10),
        ...     'D': np.random.rand(10)
        ... }
        >>> df = pd.DataFrame(data)
        >>> plotter = TimeSeriesPlotter().fit(df)
        >>> plotter.heatmapCorrelation(
        ...     figsize=(8, 8),
        ...     title='Variable Correlation',
        ...     cmap='viridis',
        ...     annot=True,
        ...     fmt=".2f"
        ... )

     
        Notes
        -----
        - The 'method' and 'min_periods' parameters are passed directly to the
          pandas 'corr' function. They determine the type of correlation coefficient
          and the handling of missing data respectively.
        - The heatmap provides a visual representation of the pairwise correlations
          between the columns in `self.data`. High positive correlations are
          typically indicated by darker shades in the heatmap, and high negative
          correlations by lighter shades, with the actual color depending on the
          chosen 'cmap'.
        """
        self.inspect 
        if self.data.empty:
            raise ValueError("The DataFrame is empty. Please provide"
                             " a DataFrame with numerical data.")
     
        # Create figure and axis if not provided
        if self.ax is None:
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
        else:
            ax = self.ax
     
        # Calculate correlation matrix and plot heatmap
        corr_matrix = self.data.corr(method=method, min_periods =min_periods )
        sns.heatmap(corr_matrix, ax=ax, annot=annot, fmt=fmt, cmap=cmap or 'coolwarm')
     
        # Set the title with a default font size from self.fontsize
        ax.set_title(title or 'Heatmap of Correlations', fontsize=self.fontsize + 2)
        
        # Display the plot
        plt.show()
        
    def _get_figure_and_axis(self, figsize=None):
        """
        Retrieves or creates a figure and axis object for plotting. If an axis
        has been predefined in the class instance, it will be used; otherwise,
        a new figure and axis will be created with the specified figure size.
    
        Parameters
        ----------
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches.
            If not provided, defaults to the class attribute `figsize`.
    
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object for the plot. Only returned if a new figure is created.
        ax : matplotlib.axes.Axes
            The axis object for the plot.
    
        Examples
        --------
        >>> # Assuming this method is part of a class with self.figsize and self.ax
        >>> fig, ax = self._get_figure_and_axis(figsize=(8, 6))
        >>> ax.plot(data)  # Continue with plotting commands
        >>> plt.show()
        
        Notes
        -----
        The `self.ax` attribute is used to allow for subplot reuse. When creating
        new subplots, this function will respect the `figsize` parameter or
        default to `self.figsize` if not provided.
        """
        if self.ax is None:
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
            return fig, ax
        else:
            return None, self.ax
           
    def plotLag(
        self, lag=1, 
        figsize=None, 
        title=None, 
        c='orange', 
        alpha=0.5, 
        **lag_kws
        ):
        """
        Generates a lag plot for the time series data to analyze the relationship 
        between an observation and its lag. Useful for checking randomness or 
        serial correlation in time series datasets.
    
        Parameters
        ----------
        lag : int, default 1
            Specifies the lag of the scatter plot, the number of time units to 
            offset.
        figsize : tuple, optional
            The size of the figure as `(width, height)` in inches.
            If not provided, defaults to the class attribute `figsize`.
        title : str, optional
            The title of the plot. If None, a default title including the lag 
            number is generated (e.g., 'Lag 1 Plot').
        c : str, default 'orange'
            The color of the scatter points in the plot.
        alpha : float, default 0.5
            The transparency level of the scatter points in the plot.
        **lag_kws : dict
            Additional keyword arguments to pass to the `pd.plotting.lag_plot` 
            function.
    
        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.Series([1, 2, 3, 4, 5, 6])
        >>> plotter = TimeSeriesPlotter().fit(data=data)
        >>> plotter.plotLag(lag=1, title='Daily Returns Lag Plot',
        ...                c='blue', alpha=0.6)
    
        Notes
        -----
        This method is intended for time series data analysis, helping identify 
        if there are patterns consistent over time. It plots data against the same 
        data offset by the lag amount, useful for identifying seasonal effects 
        or autocorrelation.
        """
        self.inspect 
    
        # Use self.ax if it has been defined, else create a new axis
        fig, ax = self._get_figure_and_axis(figsize = figsize  )
    
        # Generate the lag plot
        pd.plotting.lag_plot(self.data[self.value_col], 
                             lag=lag, ax=ax, c=c, alpha=alpha, **lag_kws)
    
        # Setting the title if not provided
        ax.set_title(title or f'Lag {lag} Plot', fontsize=self.fontsize + 2)
        # configure grid 
        self._configure_grid(ax )
        
        plt.show()
        
    def plotScatter(
        self, 
        color='blue', 
        figsize=None, 
        title=None, 
        rotation=0,
        xlabel=None, 
        ylabel=None, 
        **scatter_kws
        ):
        """
        Generates a scatter plot of the time series data. This method provides a
        way to visually examine the relationship between two variables.
    
        Parameters
        ----------
        color : str, default 'blue'
            Color of the scatter points in the plot.
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches.
            If not provided, defaults to the class attribute `figsize`.
        title : str, optional
            The title of the plot. If None, defaults to 'Scatter Plot'.
        rotation : int, optional
            The rotation angle (in degrees) for the x-axis tick labels.
        xlabel : str, optional
            The label for the x-axis. If None, defaults to the class attribute
            `date_col`.
        ylabel : str, optional
            The label for the y-axis. If None, defaults to the class attribute
            `value_col`.
        **scatter_kws : dict
            Additional keyword arguments to pass to the `ax.scatter` function.
    
        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.DataFrame({
        ...     'Date': pd.date_range(start="2020-01-01", periods=100, freq='D'),
        ...     'Value': np.random.rand(100)
        ... })
        >>> plotter = TimeSeriesPlotter().fit(data, 'Date', 'Value')
        >>> plotter.plotScatter(color='red', figsize=(10, 6),
        ...                     title='Daily Values Scatter Plot',
        ...                     xlabel='Date', ylabel='Value')
    
        Notes
        -----
        This method is useful for spotting outliers, trends, or clusters of
        data points. It is particularly effective in time series analysis to
        reveal patterns or data irregularities over time.
        """
        self.inspect 
    
        # Create figure and axis if not provided
        fig, ax = self._get_figure_and_axis(figsize=figsize)
    
        # Plot scatter using additional keyword arguments
        ax.scatter(self.data[self.date_col], self.data[self.value_col], 
                   color=color, **scatter_kws)
    
        # Setting the title and configuring axes labels
        ax.set_title(title or 'Scatter Plot', fontsize=self.fontsize + 2)
        
        # configure grid 
        self._configure_grid(ax )
        
        # Configure axes, labels, and rotation
        self._configure_axes_and_labels(ax, xlabel, ylabel, rotation=rotation)
        
    def plotCumulativeDistribution(
        self, color='green', 
        figsize=None, 
        title=None,
        bins=30, 
        alpha=0.6, 
        density=True, 
        rotation=0,
        xlabel=None, 
        ylabel=None, 
        **hist_kws
        ):
        """
        Generates a cumulative distribution plot of the time series data, highlighting
        how values are distributed across the dataset. This visualization is essential
        for understanding the overall distribution and identifying the presence of
        outliers, skewness, or other anomalies in the data.
    
        Parameters
        ----------
        color : str, default 'green'
            The color of the plot lines. This parameter allows customization of the
            plot's color scheme to enhance readability or thematic presentation.
        figsize : tuple, optional
            Dimensions of the figure as `(width, height)` in inches. If not specified,
            defaults to the class attribute `figsize`.
        title : str, optional
            The title of the plot. If not provided, 'Cumulative Distribution' is used.
        bins : int, default 30
            The number of histogram bins to use. More bins provide a more detailed
            view of the distribution but can lead to overfitting in visualization.
        alpha : float, default 0.6
            Transparency of the histogram bars. Helps in visual overlap and depth
            perception in dense plots.
        density : bool, default True
            If True, the histogram is normalized to form a probability density, where
            the area under the histogram will sum to one. This is useful for comparing
            distributions of datasets of different sizes.
        rotation : int, default 0
            The rotation angle of x-axis labels. Useful for improving label readability
            in cases of closely spaced or long text labels.
        xlabel : str, optional
            Label for the x-axis. If not provided, defaults to the class attribute
            `date_col` if applicable.
        ylabel : str, optional
            Label for the y-axis. Defaults to 'Cumulative Probability' if not provided.
        **hist_kws : dict
            Additional keyword arguments to pass to `plt.hist`. Allows for further
            customization of the histogram plot, such as edge color, line style, etc.
    
        Examples
        --------
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.Series(np.random.randn(1000))
        >>> plotter = TimeSeriesPlotter().fit(data=pd.DataFrame({'Value': data}),
                                              value_col='Value')
        >>> plotter.plotCumulativeDistribution(
                color='blue', figsize=(12, 8),
                title='Cumulative Distribution of Returns',
                bins=40, alpha=0.5, density=True
            )
    
        Notes
        -----
        Cumulative distribution plots are particularly useful in statistical analysis for
        determining the probability bounds of the dataset. The plot can be used to quickly
        ascertain the percentile ranking of data points, aiding in decision-making processes
        related to risk management or quality control.
        """
        self.inspect 
    
        # Create figure and axis using helper function 
        fig, ax = self._get_figure_and_axis(figsize=figsize)
    
        # Plot the cumulative distribution histogram
        ax.hist(self.data[self.value_col], bins=bins, density=density,
                cumulative=True, color=color, alpha=alpha, **hist_kws)
    
        # Setting plot titles and labels
        ax.set_title(title or 'Cumulative Plot', fontsize=self.fontsize + 2)
        
        # configure grid 
        self._configure_grid(ax )
        
        # Configure axes, labels, and rotation
        self._configure_axes_and_labels(
            ax, xlabel, ylabel or 'Cumulative Probability', rotation=rotation
        )
        
    def plotStackedBar(
        self, 
        secondary_col, 
        title=None, 
        figsize=None,
        xlabel=None, 
        ylabel=None, 
        rotation=0
        ):
        """
        Generates a stacked bar plot of the time series data. This type of plot
        is useful for comparing the proportion of two data series at each date
        point, emphasizing their cumulative totals.
    
        Parameters
        ----------
        secondary_col : str
            The name of the secondary column to stack. This column and the primary
            column (`self.value_col`) will be displayed cumulatively in the stacked bars.
        title : str, optional
            The title of the plot. If None, defaults to 'Stacked Bar Plot'.
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches.
            If not provided, defaults to the class attribute `figsize` or (10, 6).
        xlabel : str, optional
            The label for the x-axis. If None, defaults to the class attribute `date_col`.
        ylabel : str, optional
            The label for the y-axis. If None, defaults to 'Total'.
        rotation : int, optional
            The rotation angle (in degrees) for the x-axis tick labels. Useful for improving
            label readability, particularly when labels are numerous or lengthy.
    
        Examples
        --------
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.DataFrame({
        ...     'Date': pd.date_range(start="2021-01-01", periods=4, freq='M'),
        ...     'Sales': [200, 240, 310, 400],
        ...     'Costs': [150, 190, 250, 300]
        ... })
        >>> plotter = TimeSeriesPlotter().fit(data=data, date_col='Date',
        ...                                   value_col='Sales')
        >>> plotter.plotStackedBar(
        ...     secondary_col='Costs',
        ...     title='Monthly Sales and Costs',
        ...     figsize=(8, 6),
        ...     xlabel='Month',
        ...     ylabel='Dollars',
        ...     rotation=45
        ... )
    
        Notes
        -----
        - This method is particularly effective in displaying how two components
          contribute to a total over time.
        - Ensure that the `secondary_col` exists in `self.data` before calling this method.
        - The method uses the `plot` function from pandas with the `kind='bar'` and `stacked=True`
          arguments to generate the plot.
        """
        self.inspect 
    
        # Create figure and axis if not provided
        fig, ax = self._get_figure_and_axis(figsize=figsize)
        
        # Plot the data
        self.data.groupby(self.date_col)[[self.value_col, secondary_col]].sum().plot(
            kind='bar', stacked=True, ax=ax
        )
        
        # Set the title and configure axes and labels
        ax.set_title(title or 'Stacked Bar Plot', fontsize=self.fontsize + 2)
    
        # configure grid 
        self._configure_grid(ax )
        # Configure axes, labels, and rotation
        self._configure_axes_and_labels(ax, xlabel, ylabel, rotation=rotation)
    
        plt.show()
        
    def plotHexbin(
        self, 
        gridsize=30, 
        title='Hexbin Plot', 
        figsize=None,
        xlabel=None, 
        ylabel=None, 
        rotation=0, 
        cmap='Blues',
        adjust_date_ticks=True, 
        **hexbin_kws
        ):
        """
        Generates a hexbin plot of the time series data, which is useful for 
        visualizing the density of points in two dimensions and is an 
        alternative to a scatter plot that can manage better overplotting in 
        large datasets.
    
        Parameters
        ----------
        gridsize : int, default 30
            Size of the hexagons in the grid. A larger gridsize means smaller
            hexagons.
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches. If not
            provided, defaults to the class attribute `figsize`.
        title : str, default 'Hexbin Plot'
            The title of the plot. Provides context and description for the 
            hexbin plot.
        xlabel : str, optional
            The label for the x-axis. If not provided, defaults to the class 
            attribute `date_col`.
        ylabel : str, optional
            The label for the y-axis. If not provided, defaults to 'Value'.
        rotation : int, default 0
            The rotation angle (in degrees) for x-axis labels, enhancing label 
            readability.
        cmap : str, default 'Blues'
            Colormap to be used for the hexbins, enhancing the visual appeal 
            of the plot.
        adjust_date_ticks : bool, default True
            If True, adjusts the x-axis to display formatted date labels when 
            the `date_col` contains datetime type data. This setting enables 
            the x-axis to present human-readabledate formats instead of numeric
            ordinal values, enhancing the interpretability of the
            plot. This adjustment is particularly useful when the datetime 
            scale needs clear representation, such as in long time series data.
        **hexbin_kws : dict
            Additional keyword arguments to pass to `plt.hexbin` for further 
            customization of the hexbin plot.
    
        Examples
        --------
        >>> import numpy as np 
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.DataFrame({
        ...     'Date': pd.date_range(start='2021-01-01', periods=100),
        ...     'Value': np.random.randn(100).cumsum()
        ... })
        >>> plotter = TimeSeriesPlotter() 
        >>> plotter.fit(data=data, date_col='Date', value_col='Value')
        >>> plotter.plotHexbin(title='Density of Data Points')
    
        Notes
        -----
        - Hexbin plots are particularly effective when dealing with large 
          datasets that would result in overplotting if a scatter plot were
          used. They help visualize the intensity of data points in different 
          regions of the plot.
        """
        self.inspect
        self._set_plot_style()
    
        # Create figure and axis if not provided
        fig, ax = self._get_figure_and_axis(figsize=figsize)
    
        # Convert dates to ordinal numbers if they are datetime objects
        if pd.api.types.is_datetime64_any_dtype(self.data[self.date_col]):
            x_values = self.data[self.date_col].apply(lambda x: x.toordinal())
        else:
            x_values = self.data[self.date_col]
    
        # Generate the hexbin plot
        ax.hexbin(x_values, self.data[self.value_col], gridsize=gridsize, 
                  cmap=cmap, **hexbin_kws)
    
        # Set the plot title and configure additional aesthetics
        ax.set_title(title, fontsize=self.fontsize + 2)
        
        # if adjust_date_ticks: 
        #     if pd.api.types.is_datetime64_any_dtype(self.data[self.date_col]):
        #         # Configure the x-axis to show date labels instead of toordinal values
        #         # Automatically adjust date ticks
        #         ax.xaxis.set_major_locator(mdates.AutoDateLocator())  
        #         ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        #             mdates.AutoDateLocator()))
        #         # Optionally, you could use a more specific formatter if needed:
        #         ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        if adjust_date_ticks and pd.api.types.is_datetime64_any_dtype(
                self.data[self.date_col]):
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
                mdates.AutoDateLocator()))

        # Configure grid settings and labels
        self._configure_grid(ax)
        self._configure_axes_and_labels(ax, xlabel or 'Date',
                                        ylabel or 'Value', rotation=rotation)
    
        plt.show()


    def plotKDE(
        self, 
        shade=True, 
        figsize=None,
        title='KDE Plot', 
        xlabel=None, 
        ylabel=None, 
        rotation=0,
        **kde_kws
        ):
        """
        Generates a Kernel Density Estimate (KDE) plot of the time series data. 
        KDE plots are useful for visualizing the distribution of data points in a
        continuous manner.
    
        Parameters
        ----------
        shade : bool, default True
            If True, fill the area under the KDE curve, providing a visual 
            representation of density mass.
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches. If not
            provided, defaults to the class attribute `figsize` or (10, 6) if that
            is not set.
        title : str, default 'KDE Plot'
            The title of the plot. Provides context and description for the 
            KDE plot.
        xlabel : str, optional
            The label for the x-axis. If not provided, defaults to 'Value'.
        ylabel : str, optional
            The label for the y-axis. If not provided, defaults to 'Density'.
        rotation : int, default 0
            The rotation angle (in degrees) for x-axis labels, enhancing label
            readability.
        **kde_kws : dict
            Additional keyword arguments to pass to `sns.kdeplot` for further 
            customization of the KDE plot. This can include parameters such as
            `bw_adjust` or `cut`.
    
        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.DataFrame({
        ...     'Value': np.random.randn(100)
        ... })
        >>> plotter = TimeSeriesPlotter() 
        >>> plotter.fit(data=data, value_col='Value')
        >>> plotter.plotKDE(title='Sample KDE Plot', shade=False)
    
        Notes
        -----
        - KDE plots are particularly effective for identifying the shape of the 
          distribution, highlighting peaks and tails. They are a smooth alternative 
          to histograms for data distribution visualization.
        """
        self.inspect
        self._set_plot_style()
    
        # Create figure and axis if not provided
        fig, ax = self._get_figure_and_axis(figsize=figsize)
    
        # Generate the KDE plot
        sns.kdeplot(self.data[self.value_col], shade=shade, ax=ax, **kde_kws)
    
        # Set the plot title and configure additional aesthetics
        ax.set_title(title, fontsize=self.fontsize + 2)
    
        # Configure grid settings and labels
        self._configure_grid(ax)
        self._configure_axes_and_labels(ax, xlabel or 'Value', 
                                        ylabel or 'Density', rotation)
    
        plt.show()
 
    def plotStep(
        self, 
        color='green', 
        linestyle='-', 
        linewidth=2,
        figsize=None,
        title='Step Plot', 
        xlabel=None, 
        ylabel=None, 
        rotation=0,
        **step_kws
        ):
        """
        Generates a step plot of the time series data. Step plots are useful for 
        visualizing changes in data values at discrete intervals, making it clear 
        where changes occur.
    
        Parameters
        ----------
        color : str, default 'green'
            Specifies the color of the step line in the plot.
        linestyle : str, default '-'
            Style of the line used in the plot.
        linewidth : int, default 2
            Thickness of the step line.
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches. If not
            provided, defaults to the class attribute `figsize`.
        title : str, default 'Step Plot'
            The title of the plot. Provides context and description for the step plot.
        xlabel : str, optional
            The label for the x-axis. If not provided, defaults to 'Date'.
        ylabel : str, optional
            The label for the y-axis. If not provided, defaults to 'Value'.
        rotation : int, default 0
            The rotation angle (in degrees) for x-axis labels, enhancing label 
            readability.
        **step_kws : dict
            Additional keyword arguments to pass to `plt.step` for further 
            customization of the step plot.
    
        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.DataFrame({
        ...     'Date': pd.date_range(start='2021-01-01', periods=100),
        ...     'Value': np.random.randn(100).cumsum()
        ... })
        >>> plotter = TimeSeriesPlotter() 
        >>> plotter.fit(data=data, date_col='Date', value_col='Value')
        >>> plotter.plotStep(title='Cumulative Changes Over Time')
    
        Notes
        -----
        - Step plots are particularly effective for time series data where 
          you want to emphasize how values change from one time point to another,
          showing discrete jumps.
        """
        self.inspect
        self._set_plot_style()
    
        # Create figure and axis if not provided
        fig, ax = self._get_figure_and_axis(figsize=figsize)
    
        # Generate the step plot
        ax.step(self.data[self.date_col], self.data[self.value_col],
                color=color, linestyle=linestyle or self.ls,
                linewidth=linewidth or self.lw, **step_kws)
    
        # Set the plot title and configure additional aesthetics
        ax.set_title(title, fontsize=self.fontsize + 2)
    
        # Configure grid settings and labels
        self._configure_grid(ax)
        self._configure_axes_and_labels(
            ax, xlabel or 'Date', ylabel or 'Value', rotation)
    
        plt.show()
        
    def plotErrorBar(
        self, 
        yerr=None, 
        color='blue', 
        ecolor='red', 
        elinewidth=2, 
        capsize=5, 
        figsize=None,
        title='Error Bar Plot', 
        xlabel=None, 
        ylabel=None, 
        rotation=0,
        **errorbar_kws
        ):
        """
        Generates an error bar plot of the time series data, which is useful for
        visualizing the variability of the data.
    
        Parameters
        ----------
        yerr : float or array-like, optional
            Represents the error bar sizes. If None, error bars will not be shown.
        color : str, default 'blue'
            Color of the main elements of the plot.
        ecolor : str, default 'red'
            Color of the error bars.
        elinewidth : int, default 2
            Width of the error bars.
        capsize : int, default 5
            Size of the end cap of the error bars, enhancing the visibility of 
            the error extents.
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches. If not
            provided, defaults to the class attribute `figsize`.
        title : str, default 'Error Bar Plot'
            The title of the plot. Provides context and description for the 
            error bar plot.
        xlabel : str, optional
            The label for the x-axis. If not provided, defaults to 'Date'.
        ylabel : str, optional
            The label for the y-axis. If not provided, defaults to 'Value'.
        rotation : int, default 0
            The rotation angle (in degrees) for x-axis labels, enhancing label
            readability.
        **errorbar_kws : dict
            Additional keyword arguments to pass to `plt.errorbar` for further 
            customization of the error bar plot.
    
        Examples
        --------
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.DataFrame({
        ...     'Date': pd.date_range(start='2021-01-01', periods=100),
        ...     'Value': np.random.randn(100).cumsum()
        ... })
        >>> yerr = np.random.rand(100) * 0.5  # Random errors
        >>> plotter = TimeSeriesPlotter() 
        >>> plotter.fit(data=data, date_col='Date', value_col='Value')
        >>> plotter.plotErrorBar(yerr=yerr, title='Measurement Uncertainty')
    
        Notes
        -----
        - Error bar plots are crucial for visualizing the uncertainty or 
          variability around data points, which can be essential for scientific
          measurements, polls, and other data where precision is variable.
        """
        self.inspect
        self._set_plot_style()
    
        # Create figure and axis if not provided
        fig, ax = self._get_figure_and_axis(figsize=figsize)
    
        # Generate the error bar plot
        ax.errorbar(self.data[self.date_col], self.data[self.value_col], 
                    yerr=yerr, fmt='o', color=color, ecolor=ecolor, 
                    elinewidth=elinewidth, capsize=capsize, **errorbar_kws)
    
        # Set the plot title and configure additional aesthetics
        ax.set_title(title, fontsize=self.fontsize + 2)
    
        # Configure grid settings and labels
        self._configure_grid(ax)
        self._configure_axes_and_labels(
            ax, xlabel or 'Date', ylabel or 'Value', rotation)
    
        plt.show()

    def plotStackedLine(
        self, 
        secondary_col, 
        figsize=None,
        title=None, 
        xlabel=None, 
        ylabel=None, 
        rotation=0,
        **line_kws
        ):
        """
        Generates a stacked line plot of the time series data, overlaying the values
        of two specified columns.
    
        Parameters
        ----------
        secondary_col : str
            The name of the secondary column to stack with the primary value column. This
            column's data will be plotted on the same graph as the primary value column,
            providing a comparative view.
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches. If not
            provided, defaults to the class attribute `figsize`.
        title : str, default 'Stacked Line Plot'
            The title of the plot. Provides context and description for the stacked line plot.
        xlabel : str, optional
            The label for the x-axis. If not provided, defaults to 'Date'.
        ylabel : str, optional
            The label for the y-axis. If not provided, defaults to 'Values'.
        rotation : int, default 0
            The rotation angle (in degrees) for x-axis labels, enhancing label readability.
        **line_kws : dict
            Additional keyword arguments to pass to `plt.plot` for further customization
            of the line plots.
    
        Examples
        --------
        >>> import numpy as np 
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.DataFrame({
        ...     'Date': pd.date_range(start='2021-01-01', periods=100),
        ...     'Value': np.random.randn(100).cumsum(),
        ...     'Secondary': np.random.randn(100).cumsum()
        ... })
        >>> plotter = TimeSeriesPlotter() 
        >>> plotter.fit(data=data, date_col='Date', value_col='Value')
        >>> plotter.plotStackedLine(secondary_col='Secondary', title='Comparison of Growth')
    
        Notes
        -----
        - Stacked line plots are beneficial for comparing multiple data series visually,
          illustrating how each category contributes to the total over time.
        """
        self.inspect
        self._set_plot_style()
    
        # Create figure and axis if not provided
        fig, ax = self._get_figure_and_axis(figsize=figsize)
    
        # Generate the primary line plot
        ax.plot(self.data[self.date_col], self.data[self.value_col],
                label='Primary', **line_kws)
    
        # Generate the secondary line plot
        line_kws.setdefault('linestyle', '--')  # Default secondary line style if not specified
        ax.plot(self.data[self.date_col], self.data[secondary_col], 
                label='Secondary', **line_kws)
    
        # Set the plot title and configure additional aesthetics
        ax.set_title(title or 'Stacked Line Plot', fontsize=self.fontsize + 2)
    
        # Configure grid settings and labels
        self._configure_grid(ax)
        self._configure_axes_and_labels(ax, xlabel or 'Date',
                                        ylabel or 'Values', rotation)
    
        ax.legend()
        plt.show()

    def plotBubble(
        self, 
        bubble_size_col, 
        figsize=None,
        title='Bubble Plot', 
        xlabel=None, 
        ylabel=None, 
        alpha=0.5,
        scale_factor=1000,
        rotation=0,
        **scatter_kws
        ):
        """
        Generates a bubble plot of the time series data, where each point's 
        size is proportional to its value in another specified column.
        
        This type of plot is useful for visualizing three dimensions of 
        data: x-values, y-values, and a size metric.
    
        Parameters
        ----------
        bubble_size_col : str
            The name of the column determining the size of each bubble. 
            This column's  values will scale the size of the bubbles proportionally.
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches. If not
            provided, defaults to the class attribute `figsize`.
        title : str, default 'Bubble Plot'
            The title of the plot. Provides context and description for the
            bubble plot.
        xlabel : str, optional
            The label for the x-axis. If not provided, defaults to 'Date'.
        ylabel : str, optional
            The label for the y-axis. If not provided, defaults to 'Value'.
        alpha : float, default 0.5
            The transparency of the bubbles, enhancing visual clarity 
            especially in plots with dense bubble clusters.
        scale_factor : int, default 1000
            A factor to scale the bubble sizes, making them visually significant.
        rotation : int, default 0
            The rotation angle (in degrees) for x-axis labels, enhancing label 
            readability.
        **scatter_kws : dict
            Additional keyword arguments to pass to `plt.scatter` for further 
            customization of the bubble plot.
    
        Examples
        --------
        >>> import numpy as np 
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.DataFrame({
        ...     'Date': pd.date_range(start='2021-01-01', periods=100),
        ...     'Value': np.random.rand(100),
        ...     'Size': np.random.rand(100) * 100
        ... })
        >>> plotter = TimeSeriesPlotter() 
        >>> plotter.fit(data=data, date_col='Date', value_col='Value')
        >>> plotter.plotBubble(bubble_size_col='Size', title='Market Volume and Price')
    
        Notes
        -----
        - Bubble plots are particularly effective for adding dimensionality to 
          data, revealing patterns or correlations involving magnitudes, and 
          assessing data distribution and density visually.
        """
        self.inspect 
        self._set_plot_style()
    
        # Create figure and axis if not provided
        fig, ax = self._get_figure_and_axis(figsize=figsize)
    
        # Generate the bubble plot
        ax.scatter(self.data[self.date_col], self.data[self.value_col],
                   s=self.data[bubble_size_col] * scale_factor, alpha=alpha,
                   **scatter_kws)
    
        # Set the plot title and configure additional aesthetics
        ax.set_title(title or 'Bubble Plot', fontsize=self.fontsize + 2)
    
        # Configure grid settings and labels
        self._configure_grid(ax)
        self._configure_axes_and_labels(
            ax, xlabel or 'Date', ylabel or 'Value', rotation)
    
        plt.show()

    @ensure_pkg("plotly", extra= ( 
        "`plotSunburst` expects 'plotly' library to be installed."
        )
    )
    def plotSunburst(
        self, 
        path_col, 
        values_col, 
        color_col=None, 
        figsize=(600, 600),
        title='Sunburst Plot'
        ):
        """
        Generates a sunburst plot of the time series data, which is useful for 
        visualizing hierarchical relationships alongside their magnitudes. This 
        type of plot is helpful for understanding proportions within different 
        levels of hierarchy in the data.
    
        Parameters
        ----------
        path_col : list
            The list of column names representing the hierarchical structure 
            from top to bottom. Each level in the list represents a deeper 
            level in the hierarchy.
        values_col : str
            The name of the column representing the values for each segment, 
            used to size the segments of the sunburst plot. Larger values 
            result in larger segments.
        color_col : str, optional
            The name of the column used to color the segments. This can 
            enhance the visual appeal and make the plot more informative 
            by encoding additional data dimensions.
        figsize : tuple, default (600, 600)
            The dimensions for the figure in pixels. This size should be 
            large enough to accommodate the detail and complexity of the 
            hierarchy.
        title : str, default 'Sunburst Plot'
            The title of the plot. Provides context and description for the 
            sunburst plot.
    
        Examples
        --------
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.DataFrame({
        ...     'Level1': ['Category A', 'Category A', 'Category B'],
        ...     'Level2': ['Subcategory 1', 'Subcategory 2', 'Subcategory 1'],
        ...     'Values': [10, 15, 5]
        ... })
        >>> plotter = TimeSeriesPlotter(data=data)
        >>> plotter.plotSunburst(path_col=['Level1', 'Level2'], values_col='Values', 
        ...                      title='Sample Sunburst Plot')

        >>> data = pd.DataFrame({
        ...     'Category': ['Fruit', 'Fruit', 'Vegetable', 'Vegetable'],
        ...     'Type': ['Apple', 'Orange', 'Carrot', 'Bean'],
        ...     'Count': [50, 30, 24, 22]
        ... })
        >>> plotter = TimeSeriesPlotter()
        >>> plotter.fit(data=data)
        >>> plotter.plotSunburst(path_col=['Category', 'Type'], values_col='Count')
        
        Notes
        -----
        - Sunburst plots are effective for displaying hierarchical data in a 
          compact visual format, making it easy to compare parts of the hierarchy 
          relative to each other.
        """
        import plotly.express as px
        self.inspect
        # Generate the sunburst plot
        fig = px.sunburst(
            self.data, 
            path=path_col, 
            values=values_col, 
            color=color_col,
            width=figsize[0] or self.figsize[0], 
            height=figsize[1] or self.figsize[1],
            title=title
        )
        # Adjust margins to fit the title
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))  
    
        fig.show()
            
    @ensure_pkg("squarify", extra= ( 
        "`plotTreemap` expects 'squarify' package to be installed."
        )
    )
    def plotTreemap(
        self, 
        sizes_col, 
        label_col=None, 
        color_col=None, 
        figsize=None,
        title='Treemap Plot', 
        **sqfy_kws
        ):
        """
        Generates a treemap of the time series data using the squarify library. 
        
        This visualization is particularly useful for displaying hierarchical 
        data as nested rectangles, where each rectangle's size is proportional
        to the data value it represents.
    
        Parameters
        ----------
        sizes_col : str
            The name of the column representing the sizes of each rectangle in 
            the treemap. Each size value dictates the area of the rectangles 
            in the treemap, allowing for quick visual comparisons of magnitude.
        label_col : str, optional
            The name of the column used to label the rectangles. Providing 
            this parameter helps identify each rectangle by displaying the 
            corresponding label. If not provided, rectangles will not be 
            labeled.
        color_col : str, optional
            The name of the column used to color the rectangles. This parameter
            can be used to apply a specific color scheme based on data values
            or categories, enhancing visual differentiation. If not provided,
            a default color scheme is applied.
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches. If 
            not provided, defaults to (10, 10), creating a balanced aspect 
            ratio suitable for most displays.
        title : str, default 'Treemap Plot'
            The title of the plot. Providing a title can contextualize the
            treemap, making it easier to understand the data being represented.
        **sqfy_kws : dict
            Additional keyword arguments to pass to the `squarify.plot` 
            function. These can include styling options like `alpha` for 
            transparency, enhancing the ability to customize the plot further
            according to specific visualization needs.
    
        Examples
        --------
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.DataFrame({
        ...     'Categories': ['Category A', 'Category B', 'Category C'],
        ...     'Sizes': [50, 30, 20],
        ...     'Colors': ['red', 'green', 'blue']
        ... })
        >>> plotter = TimeSeriesPlotter(data=data)
        >>> plotter.plotTreemap(sizes_col='Sizes', label_col='Categories',
        ...                     color_col='Colors', 
        ...                     title='Sample Treemap')
    
        Notes
        -----
        - Treemaps are excellent for providing an overview of data, allowing 
          viewers to instantly perceive the relative sizes of data points. 
          They are widely used in financial analysis, portfolio management, 
          and space allocation analysis.
        """
        self.inspect
        import squarify
        plt.figure(figsize=figsize or self.figsize)
        # Prepare data
        sizes = self.data[sizes_col]
        labels = self.data[label_col] if label_col else None
        color = self.data[color_col] if color_col else None
    
        # Generate the treemap
        squarify.plot(sizes=sizes, label=labels, alpha=0.6, color=color, 
                      **sqfy_kws)
        plt.axis('off')
        plt.title(title)
        plt.show()

    def radarChart(
        self, 
        categories, 
        figsize=None,
        title=None, 
        color='b', 
        alpha=0.1,
        line_style='solid',
        **plot_kws
        ):
        """
        Generates a radar chart of the time series data, suitable for showing 
        multidimensional observations with an equal number of variables.
    
        Parameters
        ----------
        categories : list
            List of categories corresponding to each axis in the radar chart.
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches. If not
            provided, defaults to (6, 6).
        title : str, default 'Radar Chart'
            The title of the plot, providing context and description.
        color : str, default 'blue'
            Color of the chart line and fill.
        alpha : float, default 0.1
            Transparency level of the fill color.
        line_style : str, default 'solid'
            Style of the chart line ('solid', 'dotted', etc.).
        **plot_kws : dict
            Additional keyword arguments to pass to `ax.plot` and `ax.fill` 
            for further customization of the radar chart.
    
        Examples
        --------
        >>> import numpy as np 
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> categories = ['Speed', 'Agility', 'Strength', 'Endurance', 'Accuracy']
        >>> values = [8, 5, 7, 6, 9]
        >>> data = pd.DataFrame([values], columns=categories)
        >>> plotter = TimeSeriesPlotter().fit(data=data)
        >>> plotter.radarChart(categories=categories, title='Athlete Skills')
    
        Notes
        -----
        - Radar charts are particularly effective for comparing different items or 
          observations with several quantitative variables.
        """
        self.inspect
        self._set_plot_style()
    
        # Calculate the angle for each category
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop
    
        # Create figure and polar subplot
        fig, ax = plt.subplots(figsize=figsize or self.figsize,
                               subplot_kw=dict(polar=True))
        
        # Draw one axe per variable and add labels
        plt.xticks(angles[:-1], categories)
    
        # Draw data points and connect them
        values = self.data[categories].iloc[0].tolist()
        values += values[:1]  # Complete the loop
        ax.plot(angles, values, linewidth=2, linestyle=line_style or self.ls,
                **plot_kws)
        ax.fill(angles, values, color=color, alpha=alpha, **plot_kws)
    
        # Add a title
        plt.title(title or 'Radar Chart', size=15, color=color, y=1.1)
    
        plt.show()

    def plotWaterfall(
        self, 
        figsize=None,
        title=None, 
        xlabel=None, 
        ylabel=None, 
        rotation=0,
        color_map=None,
        **bar_kws
        ):
        """
        Generates a waterfall plot of the time series data, useful for visualizing 
        sequential positive and negative contributions towards a cumulative total.
    
        Parameters
        ----------
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches. If not
            provided, defaults to the class attribute `figsize`.
        title : str, default 'Waterfall Plot'
            The title of the plot. Provides context and description for the 
            waterfall plot.
        xlabel : str, optional
            The label for the x-axis. If not provided, defaults to 'Date'.
        ylabel : str, optional
            The label for the y-axis. If not provided, defaults to 'Change 
            in Value'.
        rotation : int, default 0
            The rotation angle (in degrees) for x-axis labels, enhancing label
            readability.
        color_map : dict, optional
            A dictionary to map conditions to colors, e.g., 
            {True: 'green', False: 'red'}.
        **bar_kws : dict
            Additional keyword arguments to pass to `plt.bar` for further 
            customization of the waterfall plot.
    
        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.DataFrame({
        ...     'Date': pd.date_range(start='2021-01-01', periods=5),
        ...     'Value': [100, -50, 150, -100, 200]
        ... })
        >>> plotter = TimeSeriesPlotter() 
        >>> plotter.fit(data=data, date_col='Date', value_col='Value')
        >>> plotter.plotWaterfall(title='Monthly Cash Flow')
    
        Notes
        -----
        - Waterfall plots are particularly effective for financial modeling 
          and analysis, helping to illustrate how initial value is influenced 
          by a series of intermediate positive and negative values.
        """
        self.inspect
        self._set_plot_style()
    
        # Create figure and axis if not provided
        fig, ax = self._get_figure_and_axis(figsize=figsize)
    
        # Determine color based on the value direction
        color_map = color_map or {True: 'g', False: 'r'}
        colors = self.data[self.value_col].apply(lambda x: color_map[x >= 0])
    
        # Generate the waterfall plot
        ax.bar(self.data[self.date_col], self.data[self.value_col],
               width=1, color=colors, **bar_kws)
    
        # Set the plot title and configure additional aesthetics
        ax.set_title(title or 'Waterfall Plot', fontsize=self.fontsize + 2)
    
        # Configure grid settings and labels
        self._configure_grid(ax)
        self._configure_axes_and_labels(
            ax, xlabel or 'Date', ylabel or 'Change in Value', rotation)
    
        plt.show()

    @property
    def inspect(self):
        """
        Property to ensure that the TimeSeriesPlotter instance is properly 
        configured before proceeding with any plotting operations.

        This method checks whether the necessary attributes, specifically 
        'value_col', have been set. The check is crucial for confirming that 
        the instance has been fitted with data correctly via the 'fit' method.

        Returns
        -------
        int
            Returns 1 if the instance is correctly configured, enabling further 
            method executions.

        Raises
        ------
        NotFittedError
            If the instance has not been fitted correctly, indicating that the
            'fit' method has not been called with necessary data specifications.

        Notes
        -----
        - This is an internal method used as a preliminary check in other methods
          of this class to ensure that data-dependent methods do not run without
          the necessary preliminary data setup.
        - The use of this method helps prevent the common error of attempting to
          perform operations on an unfitted model.

        Examples
        --------
        Here's how this property works internally when called within another method:

        >>> plotter = TimeSeriesPlotter()
        >>> plotter.plotLine()  # Assume plotLine calls 'self.inspect' internally
        Traceback (most recent call last):
            ...
        gofast.exceptions.NotFittedError: TimeSeriesPlotter instance is not 
        fitted yet. Call 'fit' with appropriate arguments before using this
        method.
        """
        msg = ("{expobj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method.")

        if self.value_col is None: 
            raise NotFittedError(msg.format(expobj=self))
        
        return 1

    def _configure_grid(self, ax=None):
        """
        Configures the grid for a given matplotlib Axes object based on the instance's
        grid display preferences. Allows detailed customization of the grid's appearance
        if the grid is enabled.
    
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional 
            The Axes object for which the grid settings are to be configured.
    
        Attributes
        ----------
        show_grid : bool
            A class attribute that determines whether to display the grid.
        gswitch : bool
            Determines whether the grid is turned on or off.
        gls : str
            Grid line style, e.g., '-', '--', '-.', ':'
        glw : float
            Grid line width.
        gaxis : {'both', 'x', 'y'}
            Specifies which grid lines to display.
        galpha : float
            Transparency of grid lines, where 0 is fully transparent and 1 is fully opaque.
        gc : str
            Grid color.
    
        Examples
        --------
        >>> ax = plt.gca()
        >>> self.show_grid = True
        >>> self.gswitch = True
        >>> self.gls = '--'
        >>> self.glw = 0.5
        >>> self.gaxis = 'both'
        >>> self.galpha = 0.7
        >>> self.gc = 'gray'
        >>> self._configure_grid(ax)
    
        Notes
        -----
        - This method should be called after all plotting is done on the ax object
          to ensure the grid settings do not interfere with other plot elements.
        - The grid settings are applied only if `show_grid` is True. Otherwise, the grid
          is removed if previously displayed.
        """
        if not self.show_grid:
            if ax:
                ax.grid(False)
            else:
                plt.grid(False)
            return
    
        # Define common grid settings
        grid_settings = {
            "which": self.gwhich,
            "axis": self.gaxis,  # Controls which grid lines to apply settings on
            "linestyle": self.gls,  # Line style, e.g., '-', '--', '-.', ':'
            "linewidth": self.glw,  # Line width
            "alpha": self.galpha,  # Transparency of grid lines
            "color": self.gc  # Color of the grid lines
        }
    
        # Apply grid settings based on whether an Axes object is provided
        target = ax if ax else plt
        target.grid(True, **grid_settings)

    def __repr__(self):
        """
        Provides a string representation of the TimeSeriesPlotter instance that 
        includes crucial information about its configuration, specifically the 
        columns used for date and values and a preview of the data.
    
        This method enhances the usability of logging and debugging by displaying
        essential attributes of the TimeSeriesPlotter instance.
    
        Returns
        -------
        str
            A formatted string that represents the configuration of the 
            TimeSeriesPlotter instance.
        """
        # Ensure essential attributes are defined, else use placeholder '<N/A>'
        date_col = getattr(self, 'date_col', '<N/A>')
        value_col = getattr(self, 'value_col', '<N/A>')
        data_repr = getattr(self, 'data', '<No data loaded>')
    
        # Format the data using a custom formatting function 
        # if it's not '<No data loaded>'
        data_str = ( 'Data not loaded' if data_repr == '<No data loaded>' 
                    else format_iterable(data_repr)
                    )
    
        # Create a formatted string that includes class name and attributes
        return (
            "<{class_name}: date_col={date_col!r}, value_col={value_col!r},"
            " data={data_str}>").format(
            class_name=self.__class__.__name__,
            date_col=date_col,
            value_col=value_col,
            data_str=data_str
        )
    
TimeSeriesPlotter.__doc__+="""\
    
A comprehensive plotting tool designed for time series data analysis. 
`TimeSeriesPlotter` provides a wide range of plotting methods to visualize 
different aspects of time series data, from basic line plots to complex 
hierarchical sunburst plots.

Attributes
----------
data : pandas.DataFrame
    The DataFrame provided by the user, containing the data to be plotted.
date_col : str
    The name of the column in `data` that represents the time dimension.
    This column is used to order data chronologically in time-based plots.
value_col : str
    The name of the column in `data` that contains the numerical values
    for plotting. This column is used as the primary variable in various
    statistical visualizations.

Methods
-------
plotLine() : Generates a line plot of the time series data.
plotHistogram() : Creates a histogram of the values in the time series.
plotBox() : Produces a box plot to summarize the distribution of the time series.
plotArea() : Displays an area plot for the time series data.
heatmapCorrelation() : Shows correlations between columns in the data.
plotLag() : Visualizes autocorrelation with a lag plot.
plotRollingMeanStd() : Plots the rolling mean and standard deviation.
plotAutocorrelation() : Creates an autocorrelation plot of the time series.
plotPACF() : Generates a partial autocorrelation plot.
plotDecomposition() : Decomposes the time series into its components.
plotScatter() : Generates a scatter plot of the time series data.
plotViolin() : Creates a violin plot to show the value distribution.
plotCumulativeDistribution() : Produces a cumulative distribution plot.
plotStackedBar() : Visualizes data with a stacked bar plot.
pieChart() : Displays a pie chart of the value distribution.
plotHexbin() : Creates a hexbin plot to represent data density.
plotKDE() : Generates a kernel density estimate plot.
plotStep() : Visualizes data with a step plot.
plotErrorBar() : Displays an error bar plot of the time series.
plotStackedLine() : Shows a stacked line plot of the time series.
plotWaterfall() : Generates a waterfall plot to illustrate sequential changes.
plotBubble() : Creates a bubble plot with variable-sized markers.
plotSunburst() : Visualizes hierarchical data with a sunburst plot.
radarChart() : Compares multivariate data with a radar chart.
...

Examples
--------
>>> import pandas as pd
>>> from gofast.plot.ts import TimeSeriesPlotter
>>> data = pd.DataFrame({
...     'Date': pd.date_range(start='2021-01-01', periods=100),
...     'Value': range(100)
... })
>>> plotter = TimeSeriesPlotter(figsize=(12, 8), fontsize=14)
>>> plotter.fit(data=data, date_col='Date', value_col='Value')
>>> plotter.plotLine(title='Sample Time Series Line Plot')

Notes
-----
- The TimeSeriesPlotter is designed to seamlessly integrate with Matplotlib's 
  plotting ecosystem, offering both simplicity in routine plotting tasks and 
  flexibility for more advanced customization needs. It utilizes an optional 
  Axes object to allow for integration into complex figure layouts, making it 
  ideal for creating detailed reports and interactive visualizations in 
  scientific computing environments.
"""


if __name__ == "__main__":
    
    from gofast.plot.ts import TimeSeriesPlotter 
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values})

    plotter = TimeSeriesPlotter().fit(df, 'Date', 'Value')
    plotter.plotStep()
    plotter.plotErrorBar(yerr=0.1)
    

    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values,
                       'SecondaryValue': np.random.rand(len(dates))})

    plotter = TimeSeriesPlotter().fit(df, 'Date', 'Value')
    plotter.plotScatter()

    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values})

    plotter = TimeSeriesPlotter().fit(df, 'Date', 'Value')
    plotter.plotRollingMeanStd()
    plotter.plotAutoCorrelation()
    plotter.plotPACF()
    plotter.plotDecomposition()
    
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values})

    plotter = TimeSeriesPlotter().fit(df, 'Date', 'Value')
    plotter.plotArea()
    plotter.heatmapCorrelation()
    plotter.plotLag()
    
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values})

    plotter = TimeSeriesPlotter().fit(df, 'Date', 'Value')
    plotter.plotArea()
    plotter.heatmapCorrelation()
    plotter.plotLag()


    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values})

    plotter = TimeSeriesPlotter().fit(df, 'Date', 'Value')
    plotter.plotLine()
    plotter.plotHistogram()
    plotter.plotBox()

    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values})

    plotter = TimeSeriesPlotter().fit(df, 'Date', 'Value')
 
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values, 
                       'SecondaryValue': np.random.rand(len(dates))})

    plotter = TimeSeriesPlotter().fit(df, 'Date', 'Value')
    plotter.plotArea()
    plotter.heatmapCorrelation()
    plotter = TimeSeriesPlotter().fit(df, 'Date', 'Value')
    plotter.plotStackedLine(secondary_col='SecondaryValue')
    plotter.plotWaterfall()
    plotter.plotBubble(bubble_size_col='BubbleSize')
    categories = ['Category1', 'Category2', 'Category3', 'Category4', 'Category5']
    plotter.radarChart(categories=categories)

