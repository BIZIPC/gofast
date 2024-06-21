# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:21:00 2024
@author: LKouadio
"""
import pytest
import pandas as pd
from gofast.query import DBAnalysis  

# Sample data for testing
test_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
test_table_name = 'test_table'
# Sample data for testing
test_data_1 = pd.DataFrame({'ID': [1, 2, 3], 'Value': [10, 20, 30]})
test_data_2 = pd.DataFrame({'ID': [1, 2, 4], 'OtherValue': [100, 200, 400]})
test_table_name_1 = 'test_table_1'
test_table_name_2 = 'test_table_2'

# @pytest.fixture
# def db_analysis():
#     db = DBAnalysis(':memory:')
#     db.fit(test_data, test_table_name)
#     return db
@pytest.fixture
def db_analysis():
    db = DBAnalysis(':memory:')
    db.fit(test_data, test_table_name)
    # db.fit(test_data_1, test_table_name_1)
    # db.fit(test_data_2, test_table_name_2)
    return db
# @pytest.fixture
# def db_analysis2():
#     db2 = DBAnalysis(':memory:')
#     db2.fit(test_data_1, test_table_name_1)
#     return db2
# @pytest.fixture
# def db_analysis3():
#     db3 = DBAnalysis(':memory:')
#     db3.fit(test_data_2, test_table_name_2)
#     return db3

def test_fit(db_analysis):
    # Test if the data is correctly fitted into the database
    query_result = db_analysis.query(f"SELECT * FROM {test_table_name}")
    assert not query_result.empty
    assert query_result.shape == test_data.shape
    assert all(query_result.columns == test_data.columns)

def test_query(db_analysis):
    # Test if the queryData method retrieves correct data
    result = db_analysis.query(f"SELECT A FROM {test_table_name}")
    expected = test_data[['A']]
    assert result.equals(expected)
    
# Example for a more complex method
def test_aggregate(db_analysis):
    # Test the aggregateData method for a COUNT operation
    result = db_analysis.aggregate(f"SELECT COUNT(*) as count FROM {test_table_name}")
    expected_count = len(test_data)
    assert result.at[0, 'count'] == expected_count

# def test_joinTables(db_analysis):
#     join_query = f"SELECT * FROM {test_table_name_1} INNER JOIN {test_table_name_2} ON {test_table_name_1}.ID = {test_table_name_2}.ID"
#     result = db_analysis.joinTables(join_query)
#     assert len(result) > 0
#     assert 'OtherValue' in result.columns

# def test_subqueriesAndTempTables(db_analysis):
#     queries = [
#         f"CREATE TEMPORARY TABLE TempTable AS SELECT * FROM {test_table_name_1}",
#         "SELECT * FROM TempTable"
#     ]
#     result = db_analysis.subqueriesAndTempTables(queries)
#     assert not result.empty
#     assert result.equals(test_data_1)

# def test_manipulate(db_analysis):
#     insert_query = f"INSERT INTO {test_table_name_1} (ID, Value) VALUES (4, 40)"
#     db_analysis.manipulate(insert_query)
#     result = db_analysis.query(f"SELECT * FROM {test_table_name_1} WHERE ID = 4")
#     assert not result.empty
#     assert result.iloc[0]['Value'] == 40

# def test_transform(db_analysis):
#     transform_query = f"UPDATE {test_table_name_1} SET Value = Value * 2 WHERE ID = 1"
#     db_analysis.transform(transform_query)
#     result = db_analysis.query(f"SELECT Value FROM {test_table_name_1} WHERE ID = 1")
#     assert result.iloc[0]['Value'] == 20  # Expecting the value to be doubled

# def test_windowFunctions(db_analysis):
#     # Example assumes the database supports window functions
#     window_query = f"SELECT ID, SUM(Value) OVER (ORDER BY ID) as RunningTotal FROM {test_table_name_1}"
#     result = db_analysis.windowFunctions(window_query)
#     assert not result.empty
#     assert 'RunningTotal' in result.columns

# def test_storedProcedures(db_analysis):
#     # This test is highly dependent on your DBMS and its setup
#     # Example: Call a stored procedure that exists in your database
#     # proc_name = "my_stored_procedure"
#     # params = [param1, param2]
#     # result = db_analysis.storedProcedures(proc_name, params)
#     # assert relevant conditions based on your procedure's functionality
#     pass
# def test_ensureDataIntegrity(db_analysis):
#     integrity_query = f"ALTER TABLE {test_table_name_1} ADD UNIQUE (ID)"
#     db_analysis.ensureDataIntegrity(integrity_query)
#     # To assert, you might try to violate the new constraint and expect failure

# def test_scalabilityPerformance(db_analysis):
#     # Example of a performance analysis query; adjust according to your DBMS
#     perf_query = f"EXPLAIN QUERY PLAN SELECT * FROM {test_table_name_1}"
#     result = db_analysis.scalabilityPerformance(perf_query)
#     assert not result.empty  # Assuming EXPLAIN returns data

# def test_compatibilityIntegration(db_analysis):
#     # Compatibility or integration queries will be specific to your environment
#     # Example: Querying database metadata
#     compat_query = "SELECT name FROM sqlite_master WHERE type='table'"
#     result = db_analysis.compatibilityIntegration(compat_query)
#     assert test_table_name_1 in result['name'].values

if __name__=='__main__': 
    
    pytest.main ( [__file__])