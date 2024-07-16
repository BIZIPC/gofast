# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
The GeoIntelligentSystem class is a  tool designed for advanced analysis 
and manipulation of geographical data. This class forms part of the 
'gofast' library's experimental suite, offering cutting-edge functionalities 
for geospatial intelligence, including data loading, transformation, impact 
evaluation, action recommendation, and dynamic visualization.

As an experimental feature, the GeoIntelligentSystem class is triggered via
the 'experimental' subpackage of 'gofast'. To utilize this class, you must 
first enable the experimental features. This ensures you are aware of the 
experimental nature of the functionalities, which may be subject to change 
or require further testing for production use.

To get started with the GeoIntelligentSystem class, follow these steps:

1. Enable the experimental subpackage:
  
   >>> from gofast.experimental import enable_geo_intel_system  # noqa
   >>> from gofast.geo.system import GeoIntelligentSystem
   
2. Initialize the GeoIntelligentSystem with optional parameters
   
   >>> geo_sys = GeoIntelligentSystem(
       source='path/to/your/data.geojson', format='GeoJSON')

3. Load data, perform transformations, evaluate impacts, recommend actions, 
   and visualize results

   >>> geo_sys.fit()  
   >>> transformed_data = geo_sys.transformCoordinates(data, targetCRS='EPSG:4326')
   >>> impact_scores = geo_sys.evaluateImpact(scenarios, criteria_weights, impact_data)
   >>> recommendations = geo_sys.recommendActions(data, objectives)
   >>> geo_sys.visualizeData(data, visualization_parameters={
       'type': 'heatmap', 'column': 'population_density'})

Note:
- The GeoIntelligentSystem class and its methods are part of an experimental 
  API that is still under development. As such, they may undergo significant 
  changes in future releases of the 'gofast' library.
- Ensure you consult the latest documentation and release notes of 'gofast' 
  to stay updated on the status and capabilities of the GeoIntelligentSystem 
  class and other experimental features.

"""
import threading
import time
try :import geopandas as gpd
except : pass 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from ..exceptions import NotFittedError
from ..tools.funcutils import ensure_pkg

__all__= ["GeoIntelligentSystem"]

class GeoIntelligentSystem:
    """
    A GeoIntelligentSystem class designed for analyzing and interacting with 
    geographical data. 
    
    It supports loading data from various sources, transforming coordinates, 
    evaluating impacts of scenarios, recommending actions based on geographical 
    analysis, and visualizing data.

    The system integrates with external APIs for real-time data streaming and
    utilizes popular geospatial libraries for data manipulation and visualization, 
    such as GeoPandas and Folium.

    Attributes
    ----------
    source : str, optional
        The path or URL to the static data source. This is used to load 
        geographical data into the system.
    format : str, optional
        The format of the static data, e.g., 'GeoJSON', 'KML'. Specifies 
        how the data should be parsed.
    stream_source : str, optional
        The URL or path to the data source for streaming real-time geographical data.
        
    verbose: int, default=0
       Display information to user and control level of verbosity. 

    Methods
    -------
    visualizeData(gdf, plot_type='map', **kwargs):
        Renders geographical data on maps or charts, supporting various 
        visualization types.

    interactiveQuery(gdf, query_action='info', **kwargs):
        Allows users to interactively query geographical information on a map.

    evaluateImpact(scenarios, criteria_weights, impact_data, scoring_func=None,
                   **kwargs):
        Evaluates the potential impact of scenarios based on specified criteria
        and an optional custom scoring function.

    recommendActions(data, objectives, optimization_method='maximize', 
                     constraints=None, scoring_func=None, **kwargs):
        Provides recommendations for actions based on the analysis of 
        geographical data and predefined objectives.

    Examples
    --------
    >>> import geopandas as gpd 
    >>> from gofast.experimental import enable_geo_intel_system
    >>> from gofast.geo.system import GeoIntelligentSystem 
    Initializing the GeoIntelligent System with a static data source and format:

    >>> geo_sys = GeoIntelligentSystem(source='path/to/data.geojson',
                                       format='GeoJSON')

    Visualizing geographical data with a heatmap:

    >>> gdf = gpd.read_file('path/to/data.geojson')
    >>> geo_sys.visualizeData(gdf, plot_type='heatmap', 
                              column='population_density', cmap='viridis')

    Querying and interacting with geographical data on an interactive map:

    >>> geo_sys.interactiveQuery(gdf, query_action='info',
                                 location=[45.5236, -122.6750], zoom_start=13)

    Evaluating the impact of different scenarios on environmental and 
    economic criteria:

    >>> scenarios = ['Project A', 'Project B']
    >>> criteria_weights = {'environmental': 0.7, 'economic': 0.3}
    >>> # Creating a DataFrame to simulate impact data for different scenarios 
    >>> # and criteria scores
    >>> impact_data = pd.DataFrame({
        'Scenario': ['Project A', 'Project B', 'Project C'],
        # Environmental impact scores (higher is better)
        'Environmental Impact': [85, 75, 90],  
        # Economic benefit scores (higher is better)
        'Economic Benefit': [70, 85, 65],  
        'Cost': [50, 40, 60]  # Cost scores (lower is better)
    })

    >>> scores = geo_sys.evaluateImpact(scenarios, criteria_weights, impact_data)

    Recommending actions based on geographical analysis with constraints:

    >>> objectives = ['accessibility', 'cost_efficiency']
    >>> constraints = {'budget_limit': (None, 10000), 'minimum_area': (500, None)}
    >>> # Creating a DataFrame to simulate potential actions and their metrics
    >>> actions_data = pd.DataFrame({
    'Action': ['Build Park', 'Develop Residential Area', 'Renovate Downtown'],
    'Accessibility': [95, 70, 85],  # Accessibility scores (higher is better)
    'Cost Efficiency': [80, 65, 75],  # Cost Efficiency scores (higher is better)
    'Sustainability Score': [90, 60, 80],  # Sustainability scores (higher is better)
    'Budget Required': [10000, 25000, 15000]  # Budget required for each action
      })
    >>> recommendations = geo_sys.recommendActions(data, objectives,
                                                   constraints=constraints)
    """
    def __init__(self, source=None, format=None, stream_source=None, verbose=0):
        """
        Initialize the GeoIntelligentSystem with optional data source and format.

        Parameters
        ----------
        source : str, optional
            The path or URL to the static data source.
        format : str, optional
            The format of the static data (e.g., 'GeoJSON', 'KML').
        stream_source : str, optional
            The URL or path to the data source for streaming real-time data.
        verbose: int, default=0 
           Print informations to user for warnings. 
        """
        self.source = source
        self.format = format
        self.stream_source = stream_source
        self.verbose=verbose 

    def fit(self, data=None, **kwargs):
        """
        Fits the GeoIntelligentSystem model to the geographical data. This method
        can load data from a specified source if data is not directly provided.

        Parameters
        ----------
        data : GeoDataFrame or str, optional
            Direct input of geographical data as a GeoDataFrame, or the path 
            to the data file. If not provided, the method will attempt to 
            load data using the internal `_load_data` method with the 
            `source` attribute.

        Other Parameters
        ----------------
        **kwargs : dict
            Additional keyword arguments to customize the data loading process,
            such as data format if loading from a file.

        Raises
        ------
        ValueError
            If both `data` and `source` are None, indicating no data source 
            was specified.

        Returns
        -------
        self : GeoIntelligentSystem
            The instance itself, to allow for method chaining.
            
        Examples
        --------
        Directly loading data from a GeoDataFrame:
    
        >>> from geopandas import GeoDataFrame
        >>> from gofast.experimental import enable_geo_intel_system
        >>> from gofast.geo.system import GeoIntelligentSystem 
        >>> geo_sys = GeoIntelligentSystem()
        >>> gdf = GeoDataFrame({...})  # Assuming gdf is a predefined GeoDataFrame
        >>> geo_sys.fit(data=gdf)
    
        Loading data from a file by specifying the file path and format:
    
        >>> geo_sys = GeoIntelligentSystem()
        >>> geo_sys.fit(data='path/to/data.geojson', format='GeoJSON')
    
        Initializing with a data source and format, then fitting without 
        directly providing data:
    
        >>> geo_sys = GeoIntelligentSystem(source='path/to/data.geojson
                                           format='GeoJSON')
        >>> geo_sys.fit()
    
        Streaming data from a specified source:
    
        >>> geo_sys = GeoIntelligentSystem(stream_source='url/to/streaming/data')
        >>> geo_sys.fit()  # This will start streaming in addition to loading any static data if source is specified
        """
        if data is not None:
            if isinstance(data, str):  # Assuming `data` is a path to a file
                self._load_data(data, kwargs.get('format', self.format))
            else:  # Assuming `data` is a GeoDataFrame
                self.data = data
        elif self.source is not None:
            self._load_data(self.source, self.format)
        else:
            raise ValueError("No data source specified. Please provide data"
                             " or a source path.")

        if self.stream_source is not None:
            self._stream_data(self.stream_source)

        return self

    def _load_data(self, source, format):
        """
        Internally loads geographical data from a specified source and format.

        Parameters
        ----------
        source : str
            The path or URL to the data source.
        format : str
            The format of the data (e.g., 'GeoJSON', 'KML').
        """
        if format.lower() == 'geojson':
            self.data = gpd.read_file(source)
        elif format.lower() == 'kml':
            self.data = gpd.read_file(source, driver='KML')
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _stream_data(self, source):
        """
        Internally simulates streaming real-time geographical data from a source.

        Parameters
        ----------
        source : str
            The URL or path to the data source for streaming real-time data.
        """
        def fetch_data():
            while True:
                print(f"Fetching new data from {source}")
                time.sleep(10)  # Simulate delay

        threading.Thread(target=fetch_data, daemon=True).start()
    
    def transformCoordinates(self, data=None, targetCRS=None, inplace=False):
        """
        Transforms the coordinate reference system (CRS) of the geographical 
        data to a target CRS.
    
        This method allows for the transformation of the CRS of the input data,
        making it compatible with other geo datasets or for specific analysis 
        needs. It uses GeoPandas for the transformation,
        which must be installed and correctly configured in the environment.
    
        Parameters
        ----------
        data : GeoDataFrame, optional
            The geographical data to transform. If not provided, it will attempt
            to transform the instance's stored data. If no data is stored, 
            it raises an error.
        targetCRS : str or dict, required
            The target coordinate reference system to transform the data into. 
            This can be anything  accepted by GeoPandas' `to_crs` method, such 
            as an EPSG code string ('EPSG:4326') or a PROJ string.
        inplace : bool, default False
            If True, the transformation is applied in-place on the provided or 
            stored data. If False,a new GeoDataFrame with the transformed data 
            is returned.
    
        Returns
        -------
        GeoDataFrame or None
            The transformed geographical data as a new GeoDataFrame if `inplace=False`.
            If `inplace=True`, nothing is returned as the transformation is applied in-place.
    
        Raises
        ------
        ValueError
            If `data` is None and no data is stored in the instance.
            If `targetCRS` is not provided.
    
        Examples
        --------
        Transforming an external GeoDataFrame and returning a new GeoDataFrame:
    
        >>> from geopandas import GeoDataFrame
        >>> from gofast.experimental import enable_geo_intel_system
        >>> from gofast.geo.system import GeoIntelligentSystem 
        >>> geo_sys = GeoIntelligentSystem()
        >>> gdf = GeoDataFrame({...})  # Assuming gdf is a predefined GeoDataFrame
        >>> transformed_gdf = geo_sys.transformCoordinates(data=gdf, targetCRS='EPSG:4326')
    
        Transforming the instance's stored data in-place:
    
        >>> geo_sys = GeoIntelligentSystem(source='path/to/data.geojson', format='GeoJSON')
        >>> geo_sys.fit()  # Load data into the instance
        >>> geo_sys.transformCoordinates(targetCRS='EPSG:4326', inplace=True)
        """
        if targetCRS is None:
            raise ValueError("targetCRS must be specified.")
        
        if data is None:
            if self.data is None:
                raise ValueError("No data provided and no data stored in the instance.")
            data = self.data
        
        if not inplace:
            return data.to_crs(targetCRS)
        else:
            data.to_crs(targetCRS, inplace=True)
            self.data = data
            
    def calculateArea(self, geometry):
        """
        Calculates the area of a given geographical feature. 
        
        This method is useful for land use analysis, environmental monitoring, 
        and other applications where the size of a geographical feature is 
        of interest.
    
        Parameters
        ----------
        geometry : GeoSeries or GeoDataFrame
            A GeoPandas GeoSeries or GeoDataFrame containing the geometries 
            whose areas are to be calculated.
    
        Returns
        -------
        float or GeoSeries
            The area of the geographical feature(s). If `geometry` is a GeoSeries,
            the return is a float representing the total area. If `geometry` 
            is a GeoDataFrame, the return is a GeoSeries with the area of 
            each feature.
    
        Examples
        --------
        >>> from gofast.experimental import enable_geo_intel_system
        >>> from gofast.geo.system import GeoIntelligentSystem 
        >>> from geopandas import GeoDataFrame, GeoSeries
        >>> geo_sys = GeoIntelligentSystem()
        >>> geometry = GeoSeries([...])  # Assuming a GeoSeries of geometries
        >>> area = geo_sys.calculateArea(geometry)
        >>> print(area)
        """
        return geometry.area

    def clusterLocations(self, data, algorithm='kmeans', 
                         **algorithmParameters):
        """
        Applies clustering algorithms to group geographical locations based on
        proximity and other criteria.
        
        Supports various algorithms like K-means, DBSCAN, etc., through scikit-learn.
    
        Parameters
        ----------
        data : GeoDataFrame
            The geographical data to be clustered.
        algorithm : str, default 'kmeans'
            The clustering algorithm to use. Supported values include 'kmeans',
            'dbscan', etc.
        **algorithmParameters : dict
            Additional parameters for the clustering algorithm.
    
        Returns
        -------
        array
            An array of cluster labels for each feature in the data.
    
        Examples
        --------
        >>> from gofast.experimental import enable_geo_intel_system
        >>> from gofast.geo.system import GeoIntelligentSystem 
        >>> from geopandas import GeoDataFrame
        >>> geo_sys = GeoIntelligentSystem()
        >>> data = GeoDataFrame([...])  # Assuming a GeoDataFrame of locations
        >>> cluster_labels = geo_sys.clusterLocations(data, algorithm='dbscan',
                                                      eps=0.3, min_samples=10)
        >>> print(cluster_labels)
        """
        from sklearn.cluster import KMeans, DBSCAN
    
        # Extracting coordinates from GeoDataFrame for clustering
        coordinates = data[['geometry']].apply(lambda x: (x.x, x.y))
    
        if algorithm.lower() == 'kmeans':
            model = KMeans(**algorithmParameters)
        elif algorithm.lower() == 'dbscan':
            model = DBSCAN(**algorithmParameters)
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
    
        # Fit model and predict clusters
        labels = model.fit_predict(list(coordinates))
        return labels

    def findNearest(self, feature, features_list, n_neighbors=1):
        """
        Identifies the nearest geographical feature(s) from a list to the 
        specified feature(s).
        
        This method supports applications like nearest facility location, 
        emergency response planning, and more.

        Parameters
        ----------
        feature : GeoSeries or GeoDataFrame
            A single geographical feature or multiple features to which the 
            nearest neighbor(s) are found.
        features_list : GeoDataFrame
            A collection of geographical features from which the nearest to 
            the `feature` is identified.
        n_neighbors : int, default 1
            The number of nearest neighbors to find. Default is 1, which means 
            the single nearest neighbor.

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the nearest geographical feature(s) 
            to the specified `feature`.
            If `n_neighbors` > 1, the result includes the specified number 
            of nearest features for each item in `feature`.

        Examples
        --------
        >>> from gofast.experimental import enable_geo_intel_system
        >>> from gofast.geo.system import GeoIntelligentSystem 
        >>> from geopandas import GeoDataFrame, GeoSeries
        >>> from shapely.geometry import Point
        >>> geo_sys = GeoIntelligentSystem()
        >>> feature = GeoSeries([Point(1, 1)])  # A single feature
        >>> # Potential nearest features
        >>> features_list = GeoDataFrame({'geometry': [Point(2, 2), Point(3, 3)]})  
        >>> nearest_feature = geo_sys.findNearest(feature, features_list)
        >>> print(nearest_feature)
        """
        # import geopandas as gpd
        # from shapely.ops import nearest_points
        # Ensure input is in GeoDataFrame format
        if isinstance(feature, gpd.GeoSeries):
            feature = gpd.GeoDataFrame(feature)

        # Calculate the nearest feature(s)
        nearest_indices = []
        for feat in feature.geometry:
            # Calculate all distances from the feature to each in the features_list
            distances = features_list.geometry.distance(feat)
            # Find the index(es) of the nearest neighbor(s)
            nearest_indices.extend(distances.nsmallest(n_neighbors).index.tolist())
        
        # Get unique indices in case of overlapping nearest for multiple input features
        unique_indices = list(set(nearest_indices))
        
        # Return the nearest feature(s) as a GeoDataFrame
        return features_list.iloc[unique_indices]

    @ensure_pkg ("networkx")
    def calculateShortestPath(self, graph, start_point, end_point, criteria='distance'):
        """
        Computes the shortest path between two points on a graph considering 
        various criteria.
    
        Parameters
        ----------
        graph : networkx.Graph
            The graph representing the area with nodes and edges where 
            pathfinding is performed.
        start_point : tuple or int
            The starting node identifier. In a geospatial context, this might
            be a tuple of (latitude, longitude),
            but in the graph, it must correspond to a node ID.
        end_point : tuple or int
            The ending node identifier, similar to `startPoint`.
        criteria : str, default 'distance'
            The criteria used to determine the shortest path, such as 
            'distance' or a custom attribute on the edges.
    
        Returns
        -------
        path : list
            A list of node IDs representing the shortest path from start to 
            end point.
    
        Examples
        --------
        >>> from gofast.experimental import enable_geo_intel_system
        >>> from gofast.geo.system import GeoIntelligentSystem 
        >>> geo_sys = GeoIntelligentSystem()
        >>> # Assume G is populated with nodes and edges
        >>> start_node = 1  # Node ID in the graph
        >>> end_node = 100  # Node ID in the graph
        >>> path = geo_sys.calculateShortestPath(G, start_node, end_node,
                                                 criteria='distance')
        >>> print(path)
        """
        import networkx as nx
        path = nx.shortest_path(graph, source=start_point, target=end_point,
                                weight=criteria)
        return path
    
    @ensure_pkg("networkx")
    @ensure_pkg ("osmnx")
    def generateRoute(self, waypoints, constraints=None):
        """
        Generates a route based on a set of waypoints and constraints, 
        using a routing service or API.
    
        Parameters
        ----------
        waypoints : list
            A list of tuples representing waypoints for the route
            (latitude, longitude).
        constraints : dict, optional
            A dictionary of constraints to apply to the route generation.
    
        Returns
        -------
        route : list
            A list of tuples (latitude, longitude) representing the 
            generated route considering the constraints.
    
        Examples
        --------
        >>> from gofast.experimental import enable_geo_intel_system
        >>> from gofast.geo.system import GeoIntelligentSystem 
        >>> waypoints = [(40.7128, -74.0060), (41.8781, -87.6298),
                         (34.0522, -118.2437)]  # NYC -> Chicago -> LA
        >>> constraints = {'avoid_tolls': True}
        >>> # Instantiate and use your GeoIntelligentSystem
        >>> geo_sys = GeoIntelligentSystem()
        >>> route = generateRoute(waypoints, constraints)
        >>> print(route)
        >>> # Define your waypoints as (latitude, longitude) tuples
        >>> waypoints = [(40.748817, -73.985428), (40.691831, -74.179687), 
                         (40.689247, -74.044502)]  # Example waypoints
        route = geo_sys.generateRoute(waypoints)
        print(route)
        """

        import osmnx as ox
        import networkx as nx
        # Ensure osmnx is configured to use the desired travel mode, e.g., 'drive'
        ox.config(use_cache=True, log_console=True)
        graph = ox.graph_from_point(waypoints[0], distance=5000, network_type='drive')

        # Calculate the shortest path between each pair of waypoints
        route_nodes = []
        for start, end in zip(waypoints[:-1], waypoints[1:]):
            start_node = ox.get_nearest_node(graph, start)
            end_node = ox.get_nearest_node(graph, end)
            shortest_path = nx.shortest_path(graph, start_node, end_node, weight='length')
            route_nodes.extend(shortest_path[:-1])  # Exclude the last node to avoid duplication

        # Add the last node of the last segment
        last_node = ox.get_nearest_node(graph, waypoints[-1])
        route_nodes.append(last_node)

        # Convert node IDs back to coordinates
        route_coords = [[graph.nodes[node]['y'], graph.nodes[node]['x']]
                        for node in route_nodes]

        return route_coords
    
    def predictTrends(
        self, data=None, 
        target=None, features=None, 
        test_size=0.2, 
        random_state=None,
        view=False, 
        **algo_params
        ):
        """
        Predicts future geographical trends or patterns using machine learning 
        models, with support for using instance data.

        Parameters
        ----------
        data : pandas.DataFrame, optional
            The data to use for model training and prediction. If None, uses 
            `self.data`.
        target : str or array-like, optional
            The name of the target variable column (if `data` is a DataFrame) 
            or the target variable array itself.
        features : list of str or array-like, optional
            Column names to use as features (if `data` is a DataFrame) or the
            feature matrix itself.
        test_size : float, optional
            The proportion of the dataset to include in the test split, 
            defaults to 0.2.
        random_state : int, optional
            Controls the shuffling applied to the data before applying the split, 
            defaults to None.
 
        view : bool, optional
            If True, plots the predicted vs actual values for the target 
            variable on the test set.
        **algo_params : key, value mappings
            Other parameters to pass to the machine learning model's constructor
            (e.g., `n_estimators=100` for a RandomForest).

        Returns
        -------
        model : estimator object
            The trained machine learning model.
        metrics : dict
            A dictionary containing performance metrics of the model, such as 
            RMSE on the test set.

        Examples
        --------
        >>> from gofast.experimental import enable_geo_intel_system
        >>> from gofast.geo.system import GeoIntelligentSystem 
        >>> geo_sys = GeoIntelligentSystem()
        >>> geo_sys.data = pd.DataFrame({...})
        >>> model, metrics = geo_sys.predictTrends(
            target='y', features=['x1', 'x2'], n_estimators=100, max_depth=5)
        """

        # Validate and prepare data
        X, y = self._validate_and_prepare_data(data, target, features)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

        # Initialize the model with provided algorithm parameters
        model = RandomForestRegressor(**algo_params)

        # Fit the model
        model.fit(X_train, y_train)

        # Make predictions and calculate metrics
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        metrics = {'RMSE': rmse}
        
        if view:
            # Plotting the Actual vs Predicted values
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, predictions, alpha=0.6, color='blue', 
                        label='Predicted')
            plt.plot(y_test, y_test, color='red', linewidth=2, label='Actual')
            plt.title('Predicted vs Actual Values')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.legend()
            plt.grid(True)
            plt.show()

        return model, metrics

    def _validate_and_prepare_data(self, data, target, features):
        """
        Validates the input data, target, and features, and prepares 
        them for model training or prediction.

        Parameters
        ----------
        data : pandas.DataFrame or None
            The data to be validated and prepared. If None, `self.data` is used.
        target : str or array-like
            The target variable. If `data` is a DataFrame, `target` should be 
            the column name as a string.
        features : list of str or array-like
            The features for the model. If `data` is a DataFrame, `features` 
            should be the column names.

        Returns
        -------
        X, y : array-like, array-like
            The features and target variable prepared for model training or prediction.

        Raises
        ------
        ValueError
            If no data is provided and no instance data is available.
        TypeError
            If the types of `features` or `target` are invalid.
        """
        if data is None:
            self.inspect 
            if self.data is not None:
                data = self.data
            else:
                raise ValueError("No data provided and no instance data available.")

        if isinstance(data, pd.DataFrame) and isinstance(
                target, str) and isinstance(features, list):
            X = data[features]
            y = data[target]
        elif isinstance(features, np.ndarray) and isinstance(target, np.ndarray):
            X = features
            y = target
        else:
            raise TypeError("Invalid types for features or target. Must be"
                            " list of str for DataFrame or arrays.")

        return X, y
    
    def forecastEnvironmentalChanges(
        self, data=None, 
        target=None, 
        features=None, 
        test_size=0.2, 
        random_state=None, 
        view=False, 
        **model_params
        ):
        """
        Forecasts environmental changes based on historical data and 
        simulation models, with optional visualization.

        Parameters
        ----------
        data : pandas.DataFrame, optional
            The data to use for model training and prediction. If None, uses 
            `self.data`.
        target : str or array-like
            The name of the target variable column (if `data` is a DataFrame)
            or the target variable array itself.
        features : list of str or array-like
            Column names to use as features (if `data` is a DataFrame) or 
            the feature matrix itself.
        test_size : float, optional
            The proportion of the dataset to include in the test split, defaults to 0.2.
        random_state : int, optional
            Controls the shuffling applied to the data before applying the split.
        view : bool, optional
            If True, plots the predicted vs actual values for the target variable on the test set.
        **model_params : key, value mappings
            Other parameters to pass to the forecasting model's constructor.

        Returns
        -------
        model : estimator object
            The trained forecasting model.
        metrics : dict
            A dictionary containing performance metrics of the model, such as
            RMSE on the test set.

        Examples
        --------
        >>> from gofast.experimental import enable_geo_intel_system
        >>> from gofast.geo.system import GeoIntelligentSystem 
        >>> geo_sys = GeoIntelligentSystem()
        >>> geo_sys.data = pd.DataFrame({...})
        >>> model, metrics = geo_sys.forecastEnvironmentalChanges(
            target='deforestation_rate', features=['year', 'protected_area',
                                                   'local_policies'],
            n_estimators=100, max_depth=5, view=True)
        """
        # Validate and prepare data
        X, y = self._validate_and_prepare_data(data, target, features)
        
        # Proceed with the method's original logic for train-test split, 
        # model training, and optional visualization
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        model = RandomForestRegressor(**model_params)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        metrics = {'RMSE': rmse}

        if view:
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, predictions, alpha=0.6, color='blue', label='Predicted')
            plt.plot(y_test, y_test, color='red', linewidth=2, label='Actual')
            plt.title('Forecasted vs Actual Environmental Changes')
            plt.xlabel('Actual Values')
            plt.ylabel('Forecasted Values')
            plt.legend()
            plt.grid(True)
            plt.show()

        return model, metrics
    
    @ensure_pkg("folium")
    def visualizeData(self, gdf, plot_type='map', **kwargs):
        """
        Renders geographical data on maps or charts, supporting various 
        visualization types including static plots and interactive maps.

        Parameters
        ----------
        gdf : GeoDataFrame
            The geographical data to visualize.
        plot_type : str, default 'map'
            The type of visualization. Options include 'map' for interactive 
            maps, 'plot' for static plots, and 'heatmap' for heatmaps.
        **kwargs : dict
            Additional parameters for customization depending on the plot_type. 
            This can include 'column' for specifying data column,
            'cmap' for colormap in static plots, or 'tiles' for folium map tiles.

        Examples
        --------
        >>> from gofast.experimental import enable_geo_intel_system
        >>> from gofast.geo.system import GeoIntelligentSystem 
        >>> geo_sys = GeoIntelligentSystem()
        >>> gdf = gpd.read_file('path/to/geospatial/data.geojson')
        >>> geo_sys.visualizeData(gdf, plot_type='heatmap',
                                  column='population_density', cmap='viridis')
        """
        import folium
        from folium.plugins import HeatMap
        if plot_type == 'plot':
            column = kwargs.get('column', None)
            cmap = kwargs.get('cmap', 'viridis')
            if column:
                gdf.plot(column=column, cmap=cmap, legend=True)
            else:
                gdf.plot()
            plt.show()
        
        elif plot_type == 'map':
            m = folium.Map(location=kwargs.get(
                'location', [gdf.geometry.centroid.y.mean(),
                             gdf.geometry.centroid.x.mean()]),
                           zoom_start=kwargs.get('zoom_start', 12),
                           tiles=kwargs.get('tiles', 'OpenStreetMap'))
            folium.GeoJson(gdf).add_to(m)
            return m
        
        elif plot_type == 'heatmap':
            m = folium.Map(location=kwargs.get(
                'location', [gdf.geometry.centroid.y.mean(), 
                             gdf.geometry.centroid.x.mean()]),
                           zoom_start=kwargs.get('zoom_start', 12))
            HeatMap(data=gdf[['geometry']].apply(
                lambda x: [x.geometry.y, x.geometry.x], axis=1).tolist(),
                    radius=kwargs.get('radius', 10)).add_to(m)
            return m
        
        else:
            raise ValueError(f"Unsupported plot_type: {plot_type}")

    @ensure_pkg("folium")
    def interactiveQuery(self, gdf, query_action='info', **kwargs):
        """
        Allows users to interactively query geographical information on a map,
        supporting actions like displaying info or filtering data.

        Parameters
        ----------
        gdf : GeoDataFrame
            The geographical data for interactive querying.
        query_action : str, default 'info'
            The action to perform. Currently supports 'info' for displaying 
            feature information.
        **kwargs : dict
            Additional parameters for customization, such as 'location' and 
            'zoom_start' for initial map view.

        Examples
        --------
        >>> from gofast.experimental import enable_geo_intel_system
        >>> from gofast.geo.system import GeoIntelligentSystem 
        >>> geo_sys = GeoIntelligentSystem()
        >>> gdf = gpd.read_file('path/to/geospatial/data.geojson')
        >>> map = geo_sys.interactiveQuery(gdf, query_action='info', 
                                           location=[45.5236, -122.6750], 
                                           zoom_start=13)
        >>> map  # This will display the interactive map in a Jupyter 
        >>> # notebook environment.
        """
        import folium
        if query_action == 'info':
            m = folium.Map(location=kwargs.get(
                'location', [gdf.geometry.centroid.y.mean(), 
                             gdf.geometry.centroid.x.mean()]),
                           zoom_start=kwargs.get('zoom_start', 12))
            
            folium.GeoJson(
                gdf,
                tooltip=folium.GeoJsonTooltip(
                    fields=kwargs.get('tooltip_fields', ['name'])),
            ).add_to(m)
            return m
        
        else:
            raise ValueError(f"Unsupported query_action: {query_action}")

    def evaluateImpact(
        self, scenarios, 
        criteria_weights, 
        impact_data, 
        scoring_func=None, 
        **kwargs
        ):
        """
        Evaluates the potential impact of scenarios based on specified 
        criteria, weights, and an optional custom scoring function.

        Parameters
        ----------
        scenarios : list of str
            A list of scenario identifiers or names to evaluate.
        criteria_weights : dict
            A dictionary mapping criteria names to their weights, indicating 
            the importance of each criterion.
        impact_data : pandas.DataFrame
            A DataFrame containing impact scores for each scenario across
            different criteria.
        scoring_func : callable, optional
            A custom function to calculate the overall impact score for each 
            scenario. It must accept two arguments:
            a row from the `impact_data` DataFrame and the `criteria_weights` 
            dictionary, and return a numerical score.
        **kwargs : dict
            Additional parameters for further customization of the scoring process.

        Returns
        -------
        pandas.Series
            A series containing the overall impact score for each scenario, 
            sorted in descending order of impact.

        Examples
        --------
        >>> from gofast.experimental import enable_geo_intel_system
        >>> from gofast.geo.system import GeoIntelligentSystem 
        >>> geo_sys = GeoIntelligentSystem()
        >>> scenarios = ['New Park', 'Residential Development']
        >>> criteria_weights = {'environmental': 0.7, 'cost': 0.3}
        >>> impact_data = pd.DataFrame({...})  # DataFrame with scenarios as rows and criteria as columns
        >>> custom_scoring = lambda row, weights: (
            row['environmental'] * weights['environmental']) - (
                row['cost'] * weights['cost'])
        >>> scores = geo_sys.evaluateImpact(scenarios, criteria_weights,
                                            impact_data, scoring_func=custom_scoring)
        >>> print(scores)
        """
        # Normalize the weights
        total_weight = sum(criteria_weights.values())
        normalized_weights = {k: v / total_weight for k, v in criteria_weights.items()}

        if scoring_func:
            # Use the custom scoring function if provided
            weighted_scores = impact_data.apply(
                scoring_func, args=(normalized_weights,), axis=1)
        else:
            # Default to a simple weighted sum if no custom scoring function is provided
            weighted_scores = impact_data.apply(
                lambda x: sum(x[c] * normalized_weights[c] for c in criteria_weights), 
                axis=1)

        return weighted_scores.sort_values(ascending=False)

    def recommendActions(
        self, 
        objectives, data=None, 
        optimization_method='maximize', 
        constraints=None, 
        scoring_func=None,
        **kwargs):
        """
        Provides recommendations for actions based on multi-criteria decision 
        analysis (MCDA) methods, predefined objectives, and optional constraints.

        Parameters
        ----------
        data : pandas.DataFrame
            The geographical data to analyze, with each row representing 
            a potential action and columns for relevant metrics.
        objectives : list of str
            The objectives to consider for recommendations, which should 
            correspond to column names in `data`.
        optimization_method : str, default 'maximize'
            The optimization method to use, either 'maximize' or 'minimize', 
            applied if no scoring_func is provided.
        constraints : dict, optional
            Constraints to apply to `data` before analysis. Specified as 
            {column_name: (min_val, max_val)}.
        scoring_func : callable, optional
            A custom scoring function for complex MCDA, accepting `data` 
            and `objectives` and returning a scored `data` DataFrame.
        **kwargs : dict
            Additional parameters for the scoring function or further 
            customization of the recommendation process.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the recommended actions, sorted according 
            to the scores calculated by the MCDA method.

        Examples
        --------
        >>> from gofast.experimental import enable_geo_intel_system
        >>> from gofast.geo.system import GeoIntelligentSystem 
        >>> geo_sys = GeoIntelligentSystem()
        >>> data = pd.DataFrame({...})
        >>> objectives = ['profit', 'sustainability_score']
        >>> constraints = {'budget': (None, 50000), 'area': (1000, None)}
        >>> custom_scoring = lambda data, objectives: my_custom_mcda_scoring(
            data, objectives, **kwargs)
        >>> recommendations = geo_sys.recommendActions(
            data, objectives, constraints=constraints, scoring_func=custom_scoring)
        >>> print(recommendations)
        """
        if data is None: 
            self.inspect 
            data = self.data.copy() 
            
        # Apply constraints
        if constraints:
            for column, (min_val, max_val) in constraints.items():
                if min_val is not None:
                    data = data[data[column] >= min_val]
                if max_val is not None:
                    data = data[data[column] <= max_val]

        # Apply custom scoring function for MCDA
        if scoring_func:
            data = scoring_func(data, objectives, **kwargs)
        else:
            # Fallback to a simple scoring method if no scoring_func is provided
            ascending = optimization_method == 'minimize'
            data['overall_score'] = data[objectives].mean(axis=1)
            data = data.sort_values(by='overall_score', ascending=ascending)

        # Ensure the recommendations are returned without the internal
        # 'overall_score' if it was used
        return data.drop(columns=['overall_score'], errors='ignore')

    def saveResults(
        self, data=None, 
        format='csv', 
        destination=None, 
        include_index=False, 
        **kwargs
        ):
        """
        Saves processed and analyzed results in various formats and destinations,
        with enhanced versatility.

        Parameters
        ----------
        data : pandas.DataFrame, optional
            The data to be saved. If None, tries to use `self.data`.
        format : str, optional
            The format in which to save the data. Defaults to 'csv'. 
            Supported formats include 'csv', 'json'.
        destination : str, optional
            The file path where the data should be saved. Defaults to the 
            current folder with a generic filename.
        include_index : bool, optional
            Whether to include the DataFrame index in the saved file.
            Defaults to False.
        **kwargs : dict
            Additional keyword arguments to pass to the pandas saving 
            function (e.g., `sep` for CSV files).

        Examples
        -------- 
        >>> from gofast.experimental import enable_geo_intel_system
        >>> from gofast.geo.system import GeoIntelligentSystem 
        >>> geo_sys = GeoIntelligentSystem()
        >>> data = pd.DataFrame({...})
        >>> geo_sys.saveResults(data, format='csv')
        >>> # Saves to 'results.csv' in the current directory if no 
        destination is specified
        """
        if data is None:
            if self.data is not None:
                data = self.data
            else:
                raise ValueError("No data provided and no instance data available.")
        
        if destination is None:
            destination = f"results.{format}"
        
        if format.lower() == 'csv':
            data.to_csv(destination, index=include_index, **kwargs)
        elif format.lower() == 'json':
            data.to_json(destination, orient='records', **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}. Supported "
                             "formats are 'csv' and 'json'.")

        print(f"Data saved to {destination} in {format.upper()} format.")

    @property 
    def inspect(self): 
        """ Inspect data and trigger plot after checking the data entry. 
        Raises `NotFittedError` if `ExPlot` is not fitted yet."""
        
        msg = ( "{expobj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method."
               )
        if not hasattr ( self, 'data'):  
            raise NotFittedError(msg.format(expobj=self))
        return 1