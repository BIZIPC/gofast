# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
import click
from pkg_resources import iter_entry_points
import gofast as gf  

import pandas as pd
from gofast.dataops.inspection import inspect_data
from gofast.dataops.quality import (
    audit_data, handle_categorical_features, convert_date_features, 
    scale_data, handle_outliers_in, handle_missing_data, assess_outlier_impact,
    merge_frames_on_index, check_missing_data, analyze_data_corr, correlation_ops,
    drop_correlated_features, handle_skew, validate_skew_method, quality_control
)

class PluginGroup(click.Group):
    def __init__(self, *args, **kwds):
        self.extra_commands = {
            e.name: e.load() for e in iter_entry_points('gf.commands')
        }
        super().__init__(*args, **kwds)

    def list_commands(self, ctx):
        return sorted(super().list_commands(ctx) + list(self.extra_commands))

    def get_command(self, ctx, name):
        return self.extra_commands.get(name) or super().get_command(ctx, name)

@click.group(cls=PluginGroup, context_settings={'help_option_names': ('-h', '--help')})
def cli():
    """The gofast command line interface."""
    pass

@cli.command()
@click.option('-v', '--version', is_flag=True, help='Show gofast version')
@click.option('--show', is_flag=True, help='Show gofast version and dependencies')
def version(version, show):
    """Display the installed version of gofast."""
    if show:
        click.echo(f"gofast {gf.show_versions()}")
    else:
        click.echo(f"gofast {gf.__version__}")

@cli.command()
@click.argument('data', type=click.Path(exists=True))
@click.option('--correlation-threshold', default=0.8, 
              help='Threshold for flagging high correlation between numeric features.')
@click.option('--categorical-threshold', default=0.75, 
              help='Threshold for detecting imbalance in categorical variables.')
@click.option('--include-stats-table', is_flag=True, 
              help='Include the table of the calculated statistic in the report.')
@click.option('--return-report', is_flag=True, 
              help='Return a report object containing the comprehensive analysis of the data inspection.')
def inspect(data, correlation_threshold, categorical_threshold, include_stats_table,
                    return_report):
    """Inspect data using the dataops module."""
    df = pd.read_csv(data)
    inspect_data(
        df,
        correlation_threshold=correlation_threshold,
        categorical_threshold=categorical_threshold,
        include_stats_table=include_stats_table,
        return_report=return_report
    )
    
@cli.command()
@click.argument('data', type=click.Path(exists=True))
@click.option('--dropna-threshold', default=0.5, help='Threshold for dropping columns or rows with missing values.')
@click.option('--categorical-threshold', default=10, help='Maximum unique values for a column to be considered categorical.')
@click.option('--handle-outliers', is_flag=True, help='Handle outliers in numerical columns.')
@click.option('--handle-missing', is_flag=True, help='Handle missing data in the DataFrame.')
@click.option('--handle-scaling', is_flag=True, help='Scale numerical columns.')
@click.option('--handle-date-features', is_flag=True, help='Handle date features.')
@click.option('--handle-categorical', is_flag=True, help='Handle categorical features.')
@click.option('--replace-with', default='median', help='Replacement method for outliers.')
@click.option('--lower-quantile', default=0.01, help='Lower quantile for clipping outliers.')
@click.option('--upper-quantile', default=0.99, help='Upper quantile for clipping outliers.')
@click.option('--fill-value', default=None, help='Value to fill missing data.')
@click.option('--scale-method', default='minmax', help='Method for scaling numerical data.')
@click.option('--missing-method', default='drop_cols', help='Method for handling missing data.')
@click.option('--outliers-method', default='clip', help='Method for handling outliers.')
@click.option('--date-features', multiple=True, default=None, help='Columns to be treated as date features.')
@click.option('--day-of-week', is_flag=True, help='Add day of the week for date features.')
@click.option('--quarter', is_flag=True, help='Add quarter of the year for date features.')
@click.option('--format-date', default=None, help='Date format for date features.')
@click.option('--return-report', is_flag=True, help='Return a detailed report.')
@click.option('--view', is_flag=True, help='Enable visualization of the data\'s state before and after preprocessing.')
@click.option('--cmap', default='viridis', help='Colormap for visualizations.')
@click.option('--fig-size', default=(12, 5), type=(int, int), help='Figure size for visualizations.')
def audit(**kwargs):
    """Audit and preprocess a DataFrame."""
    data = pd.read_csv(kwargs.pop('data'))
    result = audit_data(data, **kwargs)
    click.echo(result)

@cli.command()
@click.argument('data', type=click.Path(exists=True))
@click.option('--categorical-threshold', default=10, help='Maximum unique values for a column to be considered categorical.')
@click.option('--return-report', is_flag=True, help='Return a report summarizing the categorical feature handling.')
@click.option('--view', is_flag=True, help='Display heatmap of data distribution before and after handling.')
@click.option('--cmap', default='viridis', help='Colormap for the heatmap visualization.')
@click.option('--fig-size', default=(12, 5), type=(int, int), help='Figure size for the heatmap.')
def categorical_features_handler(**kwargs):
    """Handle categorical features in a DataFrame."""
    data = pd.read_csv(kwargs.pop('data'))
    result = handle_categorical_features(data, **kwargs)
    click.echo(result)

@cli.command()
@click.argument('data', type=click.Path(exists=True))
@click.option('--date-features', multiple=True, required=True, help='Columns to be converted to datetime.')
@click.option('--day-of-week', is_flag=True, help='Add day of the week for date features.')
@click.option('--quarter', is_flag=True, help='Add quarter of the year for date features.')
@click.option('--format', default=None, help='Date format for date features.')
@click.option('--return-report', is_flag=True, help='Return a report summarizing the date feature transformations.')
@click.option('--view', is_flag=True, help='Display heatmap of data distribution before and after conversion.')
@click.option('--cmap', default='viridis', help='Colormap for the heatmap visualization.')
@click.option('--fig-size', default=(12, 5), type=(int, int), help='Figure size for the heatmap.')
def date_features_converter(**kwargs):
    """Convert specified columns to datetime and extract relevant features."""
    data = pd.read_csv(kwargs.pop('data'))
    result = convert_date_features(data, **kwargs)
    click.echo(result)

@cli.command()
@click.argument('data', type=click.Path(exists=True))
@click.option('--method', default='norm', help='Method for scaling numerical data.')
@click.option('--return-report', is_flag=True, help='Return a report summarizing the scaling process.')
@click.option('--use-sklearn', is_flag=True, help='Use scikit-learn for scaling.')
@click.option('--view', is_flag=True, help='Display heatmap of data distribution before and after scaling.')
@click.option('--cmap', default='viridis', help='Colormap for the heatmap visualization.')
@click.option('--fig-size', default=(12, 5), type=(int, int), help='Figure size for the heatmap.')
def data_scaler(**kwargs):
    """Scale numerical columns in the DataFrame."""
    data = pd.read_csv(kwargs.pop('data'))
    result = scale_data(data, **kwargs)
    click.echo(result)

@cli.command()
@click.argument('data', type=click.Path(exists=True))
@click.option('--method', default='clip', help='Method for handling outliers.')
@click.option('--replace-with', default='median', help='Replacement method for outliers.')
@click.option('--lower-quantile', default=0.01, help='Lower quantile for clipping outliers.')
@click.option('--upper-quantile', default=0.99, help='Upper quantile for clipping outliers.')
@click.option('--return-report', is_flag=True, help='Return a report summarizing the outlier handling process.')
@click.option('--view', is_flag=True, help='Display comparative plot showing data distribution before and after handling outliers.')
@click.option('--cmap', default='viridis', help='Colormap for the heatmap visualization.')
@click.option('--fig-size', default=(12, 5), type=(int, int), help='Figure size for the heatmap.')
def outliers_handler(**kwargs):
    """Handle outliers in numerical columns."""
    data = pd.read_csv(kwargs.pop('data'))
    result = handle_outliers_in(data, **kwargs)
    click.echo(result)

@cli.command()
@click.argument('data', type=click.Path(exists=True))
@click.option('--method', default=None, help='Method to handle missing data.')
@click.option('--fill-value', default=None, help='Value to use for filling missing data.')
@click.option('--dropna-threshold', default=0.5, help='Threshold for dropping rows/columns with missing data.')
@click.option('--return-report', is_flag=True, help='Return a report summarizing the missing data handling process.')
@click.option('--view', is_flag=True, help='Display heatmap of missing data before and after handling.')
@click.option('--cmap', default='viridis', help='Colormap for the heatmap visualization.')
@click.option('--fig-size', default=(12, 5), type=(int, int), help='Figure size for the heatmap.')
def missing_handler(**kwargs):
    """Handle missing data in the DataFrame."""
    data = pd.read_csv(kwargs.pop('data'))
    result = handle_missing_data(data, **kwargs)
    click.echo(result)

@cli.command()
@click.argument('data', type=click.Path(exists=True))
@click.option('--outlier-threshold', default=3, help='Z-score threshold for considering a data point an outlier.')
@click.option('--handle-na', default='ignore', help='How to handle NaN values.')
@click.option('--view', is_flag=True, help='Generate plots to visualize outliers and their impact.')
@click.option('--fig-size', default=(14, 6), type=(int, int), help='Size of the figure for the plots.')
def assess_outliers(**kwargs):
    """Assess the impact of outliers on dataset statistics."""
    data = pd.read_csv(kwargs.pop('data'))
    result = assess_outlier_impact(data, **kwargs)
    click.echo(result)

@cli.command()
@click.argument('data1', type=click.Path(exists=True))
@click.argument('data2', type=click.Path(exists=True))
@click.option('--index-col', required=True, help='The name of the column to set as the index before merging.')
@click.option('--join-type', default='outer', help='The type of join to perform.')
@click.option('--axis', default=1, help='The axis to concatenate along.')
@click.option('--ignore-index', is_flag=True, help='Whether to ignore the index in the result.')
@click.option('--sort', is_flag=True, help='Sort the result DataFrame.')
def merge_frames(**kwargs):
    """Merge multiple DataFrames based on a specified column set as the index."""
    data1 = pd.read_csv(kwargs.pop('data1'))
    data2 = pd.read_csv(kwargs.pop('data2'))
    result = merge_frames_on_index(data1, data2, **kwargs)
    click.echo(result)

@cli.command()
@click.argument('data', type=click.Path(exists=True))
@click.option('--view', is_flag=True, help='Display a pie chart visualization of the missing data distribution.')
@click.option('--explode', default=None, help='Explode data for the pie chart visualization.')
@click.option('--shadow', is_flag=True, help='Draw a shadow beneath the pie chart.')
@click.option('--startangle', default=90, help='Starting angle of the pie chart.')
@click.option('--cmap', default='viridis', help='Colormap for the pie chart visualization.')
@click.option('--autopct', default='%1.1f%%', help='String format for the percentage of each slice in the pie chart.')
@click.option('--verbose', is_flag=True, help='Print messages about automatic adjustments.')
def check_missing(**kwargs):
    """Check for missing data in a DataFrame."""
    data = pd.read_csv(kwargs.pop('data'))
    result = check_missing_data(data, **kwargs)
    click.echo(result)

@cli.command()
@click.argument('data', type=click.Path(exists=True))
@click.option('--columns', multiple=True, default=None, help='Specific columns to consider for correlation calculation.')
@click.option('--method', default='pearson', help='Method to use for computing the correlation.')
@click.option('--min-periods', default=1, help='Minimum number of observations required per pair of columns to have a valid result.')
@click.option('--min-corr', default=0.5, help='Minimum threshold for correlations to be noted.')
@click.option('--high-corr', default=0.8, help='Threshold above which correlations are considered high.')
@click.option('--interpret', is_flag=True, help='Use symbolic representation for interpretation instead of numeric values.')
@click.option('--hide-diag', is_flag=True, help='Hide diagonal values in the correlation matrix visualization.')
@click.option('--no-corr-placeholder', default='...', help='Text to display for correlation values below min_corr.')
@click.option('--autofit', is_flag=True, help='Adjust column widths and number of visible rows based on content.')
@click.option('--view', is_flag=True, help='Display a heatmap of the correlation matrix.')
@click.option('--cmap', default='viridis', help='Colormap for the heatmap visualization.')
@click.option('--fig-size', default=(8, 8), type=(int, int), help='Dimensions of the figure that displays the heatmap.')
def corr_analyzer(**kwargs):
    """Compute the correlation matrix for specified columns in a DataFrame."""
    data = pd.read_csv(kwargs.pop('data'))
    result = analyze_data_corr(data, **kwargs)
    click.echo(result)

@cli.command()
@click.argument('data', type=click.Path(exists=True))
@click.option('--correlation-type', default='all', help='Type of correlations to consider in the analysis.')
@click.option('--min-corr', default=0.5, help='Minimum correlation value to consider for moderate correlations.')
@click.option('--high-corr', default=0.8, help='Threshold above which correlations are considered strong.')
@click.option('--method', default='pearson', help='Method of correlation to use.')
@click.option('--min-periods', default=1, help='Minimum number of observations required per pair of columns to have a valid result.')
@click.option('--display-corrtable', is_flag=True, help='Print the correlation matrix to the console.')
def corrops(**kwargs):
    """Perform correlation analysis on a DataFrame and classify correlations."""
    data = pd.read_csv(kwargs.pop('data'))
    result = correlation_ops(data, **kwargs)
    click.echo(result)

@cli.command()
@click.argument('data', type=click.Path(exists=True))
@click.option('--method', default='pearson', help='Method of correlation to use.')
@click.option('--threshold', default=0.8, help='Threshold above which one of the features in a pair will be removed.')
@click.option('--display-corrtable', is_flag=True, help='Print the correlation matrix to the console.')
def drop_corr_features(**kwargs):
    """Analyze and remove highly correlated features from a DataFrame."""
    data = pd.read_csv(kwargs.pop('data'))
    result = drop_correlated_features(data, **kwargs)
    click.echo(result)

@cli.command()
@click.argument('data', type=click.Path(exists=True))
@click.option('--method', default='log', help='Method for transformation to correct skewness.')
@click.option('--view', is_flag=True, help='Visualize the distribution of original and transformed data.')
@click.option('--fig-size', default=(12, 8), type=(int, int), help='Dimensions of the figure that displays the plots.')
def skew_handler(**kwargs):
    """Apply a specified transformation to numeric columns to correct for skewness."""
    data = pd.read_csv(kwargs.pop('data'))
    result = handle_skew(data, **kwargs)
    click.echo(result)

@cli.command()
@click.argument('data', type=click.Path(exists=True))
@click.option('--missing-threshold', default=0.05, help='Threshold to determine excessive missing data in a column.')
@click.option('--outlier-method', default='IQR', help='Method to detect outliers.')
@click.option('--value-ranges', default=None, help='Acceptable (min, max) range for values.')
@click.option('--unique-value-columns', default=None, help='Columns expected to have unique values throughout.')
@click.option('--string-patterns', default=None, help='Patterns that values in specified columns should match.')
@click.option('--include-data-types', is_flag=True, help='Include data types of each column in the results.')
@click.option('--verbose', is_flag=True, help='Enable printing of messages about the operations being performed.')
@click.option('--polish', is_flag=True, help='Clean the DataFrame based on the checks performed.')
@click.option('--columns', default=None, help='Specific subset of columns to perform checks on.')
@click.option('--kwargs', default=None, help='Additional keyword arguments for extensions or underlying functions.')
def qc(**kwargs):
    """Perform comprehensive data quality checks on a DataFrame."""
    data = pd.read_csv(kwargs.pop('data'))
    result = quality_control(data, **kwargs)
    click.echo(result)


if __name__ == '__main__':
    cli()

# XXXTODO : write consistent CLI 

# references 
# https://click.palletsprojects.com/en/8.1.x/
# https://setuptools.pypa.io/en/latest/userguide/entry_point.html
