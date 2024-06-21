# -*- coding: utf-8 -*-
# test_geo_system.py
import pytest
from gofast.experimental import enable_geo_intel_system # noqa 
from gofast.geo.system import GeoIntelligentSystem
from gofast.tools.funcutils import ensure_pkg, install_package 
import pandas as pd
try:import geopandas as gpd
except: 
    install_package("geopandas")

@pytest.fixture
def setup_geo_system():
    # Initialize GeoIntelligentSystem with mock parameters or paths
    return GeoIntelligentSystem(source='mock/path/to/data.geojson', format='GeoJSON')

def test_initialization(setup_geo_system):
    assert setup_geo_system.source == 'mock/path/to/data.geojson'
    assert setup_geo_system.format == 'GeoJSON'
    # Add more assertions based on initialization parameters and defaults

def test_load_data(setup_geo_system, tmp_path):
    # Create a temporary GeoJSON file to simulate loading data
    geojson_content = """
    {
      "type": "FeatureCollection",
      "features": [
        {
          "type": "Feature",
          "geometry": {
            "type": "Point",
            "coordinates": [125.6, 10.1]
          },
          "properties": {
            "name": "Test Point"
          }
        }
      ]
    }
    """
    temp_geojson = tmp_path / "temp_data.geojson"
    temp_geojson.write_text(geojson_content)

    # Assuming loadData is implemented to update self.data
    setup_geo_system.fit(str(temp_geojson))
    assert setup_geo_system.data is not None
    assert len(setup_geo_system.data) > 0
    assert 'name' in setup_geo_system.data.columns
    
@ensure_pkg("folium", auto_install= True, verbose =True )
def test_visualize_data(setup_geo_system):
    # Assuming the GeoIntelligentSystem class has been set up with a 
    # valid GeoDataFrame `self.data`
    # and `visualizeData` returns a folium Map object for interactive map visualizations
    import folium
    # Setup: Create a mock GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'name': ['Test Point'],
        'geometry': gpd.points_from_xy([125.6], [10.1])
    })

    # Test the visualization method for an interactive map
    map_result = setup_geo_system.visualizeData(gdf, plot_type='map',
                                                location=[10.1, 125.6], zoom_start=12)
    assert isinstance(map_result, folium.Map)

def test_evaluate_impact(setup_geo_system):
    scenarios = ['Project A', 'Project B']
    criteria_weights = {'environmental': 0.7, 'cost': 0.3}
    impact_data = pd.DataFrame({
        'Scenario': scenarios,
        'environmental': [85, 75],
        'cost': [50, 40]
    })
    scores = setup_geo_system.evaluateImpact(scenarios, criteria_weights, impact_data)
    assert not scores.empty

def test_recommend_actions(setup_geo_system):
    data = pd.DataFrame({
        'Action': ['Action A', 'Action B'],
        'profit': [100, 150],
        'sustainability_score': [80, 90]
    })
    objectives = ['profit', 'sustainability_score']
    recommendations = setup_geo_system.recommendActions(
       objectives,  data, optimization_method='maximize')
    assert not recommendations.empty

@ensure_pkg("folium", auto_install= True, verbose =True )
def test_interactive_query(setup_geo_system):
    import folium
    # Setup: Create a mock GeoDataFrame for the interactive query test
    gdf = gpd.GeoDataFrame({
        'name': ['Interactive Point'],
        'geometry': gpd.points_from_xy([125.6], [10.1])
    })

    # Execute: Call the interactiveQuery method, assuming it returns a Folium map object
    interactive_map = setup_geo_system.interactiveQuery(
        gdf, query_action='info', location=[10.1, 125.6], zoom_start=12)

    # Verify: The map object is correctly instantiated and contains 
    # at least one layer (the base map counts as one)
    assert isinstance(interactive_map, folium.Map)
    assert len(interactive_map._children) > 1  # Check if more than the base layer is present, indicating added interaction layers

def test_stream_data(setup_geo_system):
    # Mock the streaming data source to return a known dataset
    # This requires the streaming data method to be adaptable for testing with a mock source
    mock_stream = [
        {'id': 1, 'data': 'mock data point 1'},
        {'id': 2, 'data': 'mock data point 2'},
    ]

    # Assuming _stream_data is designed to handle streaming data and update
    # an attribute like self.streamed_data
    setup_geo_system._stream_data(source=mock_stream)

    # Verify: The system correctly handles and stores the streamed data
    # This assumes that _stream_data or a related method updates `self._streamed_data`
    assert hasattr(setup_geo_system, 'streamed_data')
    assert len(setup_geo_system.streamed_data) == len(mock_stream)
    # Further verification can include checking the content of the streamed data

if __name__=='__main__': 
    
    pytest.main( [ __file__])