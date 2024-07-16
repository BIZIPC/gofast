# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
:mod:`gofast.query` module provides tools for database analysis, featuring 
the `DBAnalysis` class to facilitate querying and analyzing database 
structures and contents efficiently.
"""

import os 
import re
import pandas as pd

from .api.types import Optional, DataFrame
from .exceptions import NotFittedError 
from .tools._dependency import import_optional_dependency
from .tools.coreutils import normalize_string
 
__all__= ['DBAnalysis']

class DBAnalysis:
    """
    A class for performing various data analysis tasks using SQL.
    
    This class provides methods for executing and managing different types 
    of SQL queries,including data querying, aggregation, joining, subqueries, 
    and more. It is designed to handle database connections and execute SQL 
    commands in a structured and reusable way.
    
    Attributes
    ----------
    db_path : str, optional
        The file path or URI for the database. If not provided, an in-memory 
        database is used.
    verbose : int
        Verbosity level of the class operations. Higher values indicate more 
        detailed messages.
    engine_ : sqlalchemy.engine.base.Engine
        The SQLAlchemy engine object used for connecting to the database.
    connection_ : sqlalchemy.engine.base.Connection
        The connection object to the database. Used for executing SQL queries.
    cursor_ : sqlalchemy.engine.base.CursorResult
        The cursor object for the database connection. Used for executing and 
        fetching query results.
    
    Parameters
    ----------
    db_path : str, optional
        The database path or URI. Defaults to an in-memory database if not 
        provided.
    verbose : int, optional
        Verbosity level for operation messages. Defaults to 0 (no verbose output).
    
    Methods
    -------
    fit(data: Optional[pd.DataFrame] = None, table_name: str = 'default_table')
        Initializes the database connection and stores a provided DataFrame.
    
    query(query: str, return_type: str = 'dataframe')
        Executes a given SQL query and returns the results in the specified 
        format.
    
    aggregate(query: str, return_type: str = 'dataframe')
        Executes a SQL aggregation query and returns the results in the 
        specified format.
    
    joinTables(query: str, return_type: str = 'dataframe')
        Executes a SQL join query and returns the results in the 
        specified format.
    
    subqueriesAndTempTables(queries: list, return_type: str = 'dataframe')
        Executes a series of SQL queries for subqueries or temporary tables.
    
    manipulate(query: str, auto_commit: bool = True, raise_error: bool = True)
        Executes a SQL data manipulation query, with options for 
        transaction control.
    
    transform(query: str, auto_commit: bool = True, raise_error: bool = True)
        Executes a SQL data transformation query, with options for 
        transaction control.
    
    windowFunctions(query: str, return_type: str = 'dataframe')
        Executes a SQL query containing window functions.
    
    storedProcedures(procedure_name: str, params: list, return_type: str = 'dataframe')
        Executes a stored SQL procedure with given parameters.
    
    ensureDataIntegrity(query: str, auto_commit: bool = True)
        Executes a SQL query intended to ensure data integrity.
    
    scalabilityPerformance(query: str, return_type: str = 'dataframe')
        Executes a SQL query intended for scalability and performance analysis.
    
    compatibilityIntegration(query: str, return_type: str = 'dataframe')
        Executes a SQL query related to database compatibility and integration.
    
    commit()
        Commits the current transaction, used when auto_commit is set to False.

    Examples
    --------
    >>> from gofast.query import DBAnalysis 
    >>> db_analysis = DBAnalysis('my_database.db', verbose=1)
    >>> data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> db_analysis.fit(data, 'my_table')
    >>> result = db_analysis.query('SELECT * FROM my_table')
    >>> print(result)
    """
    def __init__(self, db_path: Optional[str] = None, verbose: int=0 ):
        self.db_path = db_path 
        self.verbose=verbose 

    def fit(self,
            data: Optional[DataFrame] = None,
            table_name: str = 'default_table'):
        """
        Initializes the database and stores the provided DataFrame. 
        
        If no DataFrame is provided, prompts the user to provide either a 
        DataFrame or a path to an existing database.

        Parameters
        ----------
        data : pandas.DataFrame, optional
            The DataFrame to be stored in the SQL database. If None, the 
            method checks for an existing database or prompts the user to 
            provide one.
        table_name : str, optional
            The name of the table where the DataFrame will be stored. Defaults to
            'default_table'.

        Raises
        ------
        ValueError
            If no DataFrame is provided and no existing database is available.

        Examples
        --------
        >>> import pandas as pd
        >>> from gofast.query import DBAnalysis 
        >>> db_analysis = DBAnalysis('my_database.db')
        >>> data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> db_analysis.fit(data, 'my_table')

        Notes
        -----
        - If no DataFrame is provided and a database path is not set, a 
          ValueError is raised
          prompting the user to provide either a DataFrame or a database path.
        - If a DataFrame is provided but no database path is set, a temporary 
          in-memory
          database is created, and the user is informed about this.
        - If a database path is provided, the method checks if the specified 
          database exists. If not, it creates a new one.
        """
        self.db_path = self.db_path or ':memory:'
        if data is None:
            if self.db_path == ':memory:' or not os.path.exists(self.db_path):
                raise ValueError("No data provided. Please provide a DataFrame"
                                 " or a database path.")
        
        if self.db_path == ':memory:' and self.verbose > 1:
            print("No database path provided. Using a temporary in-memory database.")
        
        self._setup_connection()
        
        if hasattr(self.connection_, 'execute'):
            # Using sqlalchemy to store data
            data.to_sql(table_name, self.engine_, if_exists='replace', index=False)
        else:
            # Using sqlite3 to store data
            data.to_sql(table_name, self.connection_, if_exists='replace', 
                        index=False, method='multi')
    
        return self

    def _setup_connection(self):
        """Setup the database connection based on the availability of 
        sqlalchemy."""
        try:
            import_optional_dependency("sqlalchemy")
            from sqlalchemy import create_engine
            self.engine_ = create_engine(f'sqlite:///{self.db_path}')
            self.connection_ = self.engine_.connect()
        except ImportError:
            import sqlite3
            self.connection_ = sqlite3.connect(self.db_path)

    @property
    def cursor_(self):
        """Get a cursor object to execute SQL commands."""
        if hasattr(self.connection_, 'execute'):
            return self.connection_
        else:
            return self.connection_.cursor()

    def __del__(self):
        if self.connection_:
            self.connection_.close()
    
    def execute_query(self, query):
        if hasattr(self.connection_, 'execute'):
            # Assuming sqlalchemy connection
            result = self.connection_.execute(query)
            return result  # This is the ResultProxy in sqlalchemy
        else:
            # Assuming sqlite3 connection
            cursor = self.connection_.cursor()
            cursor.execute(query)
            return cursor  # This is the cursor that can fetchall in sqlite3

    def query(self, query: str, return_type: str = 'dataframe') -> DataFrame:
        """
        Executes a SQL query and returns the results. The results can be returned either
        as a DataFrame or as a raw cursor result, based on the specified return type.

        Parameters
        ----------
        query : str
            The SQL query to be executed.
        return_type : {'dataframe', 'raw'}, optional
            The format in which to return the query results. If 'dataframe', the results
            are returned as a pandas DataFrame. If 'raw', the results are returned as
            they are from the database cursor. Defaults to 'dataframe'.

        Returns
        -------
        pandas.DataFrame or list
            Query results in the specified format. Either a DataFrame or a raw list,
            depending on the 'return_type' parameter.

        Raises
        ------
        ValueError
            If 'return_type' is not one of the expected options ('dataframe' or 'raw').

        Examples
        --------
        >>> from gofast.query import DBAnalysis 
        >>> db_analysis = DBAnalysis('my_database.db').fit(table_name='my_table')
        >>> result = db_analysis.query('SELECT * FROM my_table', 'dataframe')
        >>> print(result)

        Notes
        -----
        - Ensure that the database connection is established before calling this method.
        - The method assumes that the SQL query is correctly formatted and valid.
        - If 'return_type' is 'dataframe', ensure pandas is installed.
        """
        self.inspect 
        self._validate_return_type(return_type )

        result=self.execute_query(query)
        return self._format_result(result, return_type)

    def aggregate(
            self, query: str, return_type: str = 'dataframe'
            ) -> DataFrame:
        """
        Executes a SQL aggregation query and returns the results.
        
        Method Validates if the query is an aggregation query before execution. 
        The results can be returned either as a DataFrame or as a raw cursor 
        result, based on the specified return type.

        Parameters
        ----------
        query : str
            The SQL aggregation query to be executed. Must contain aggregate 
            functions like COUNT, SUM, AVG, MAX, MIN.
        return_type : {'dataframe', 'raw'}, optional
            The format in which to return the query results. If 'dataframe', 
            the results are returned as a pandas DataFrame. If 'raw', the 
            results are returned as they are from the database cursor. 
            Defaults to 'dataframe'.

        Returns
        -------
        pandas.DataFrame or list
            Aggregation query results in the specified format. Either a 
            DataFrame or a raw list, depending on the 'return_type' parameter.

        Raises
        ------
        ValueError
            If 'return_type' is not one of the expected options 
            ('dataframe' or 'raw').If the query does not seem to be an 
            aggregation query.

        Examples
        --------
        >>> from gofast.query import DBAnalysis 
        >>> db_analysis = DBAnalysis('my_database.db').fit(table_name='my_table')
        >>> result = db_analysis.aggregate(
            'SELECT COUNT(*) FROM my_table', 'dataframe')
        >>> print(result)

        Notes
        -----
        - This method is specifically for aggregation queries. The query is 
          validated to check if it contains aggregate functions.
        - Ensure that the database connection is established before calling 
          this method.
        - The method assumes that the SQL query is correctly formatted and valid.
        - If 'return_type' is 'dataframe', ensure pandas is installed.
        
        """
        self.inspect 
        if not re.search(r'\b(COUNT|SUM|AVG|MAX|MIN)\b', query, re.IGNORECASE):
            raise ValueError("The query does not appear to be an aggregation query.")

        self._validate_return_type(return_type )
        result=self.execute_query(query)
        return self._format_result(result, return_type)

    def joinTables(self, query: str, return_type: str = 'dataframe'
                   ) -> DataFrame:
        """
        Executes a SQL join query and returns the results. 
        
        Function validates if the query is a join query before execution. The 
        results can be returned either as a DataFrame or as a raw cursor result, 
        based on the specified return type.

        Parameters
        ----------
        query : str
            The SQL join query to be executed. Must contain JOIN keywords like JOIN, 
            INNER JOIN, LEFT JOIN, RIGHT JOIN, FULL JOIN.
        return_type : {'dataframe', 'raw'}, optional
            The format in which to return the query results. If 'dataframe', 
            the results are returned as a pandas DataFrame. If 'raw', the 
            results are returned as they are from the database cursor. 
            Defaults to 'dataframe'.

        Returns
        -------
        pandas.DataFrame or list
            Join query results in the specified format. Either a DataFrame or a raw list, 
            depending on the 'return_type' parameter.

        Raises
        ------
        ValueError
            If 'return_type' is not one of the expected options 
            ('dataframe' or 'raw').
            If the query does not seem to be a join query.

        Examples
        --------
        >>> from gofast.query import DBAnalysis 
        >>> db_analysis = DBAnalysis('my_database.db').fit(table_name='my_table')
        >>> result = db_analysis.joinTables(
            'SELECT * FROM table1 INNER JOIN table2 ON table1.id = table2.id', 'dataframe')
        >>> print(result)

        Notes
        -----
        - This method is specifically for join queries. The query is validated 
          to check if it contains JOIN keywords.
        - Ensure that the database connection is established before calling 
          this method.
        - The method assumes that the SQL query is correctly formatted and valid.
        - If 'return_type' is 'dataframe', ensure pandas is installed.
        """
        self.inspect 
        if not re.search(r'\b(JOIN|INNER JOIN|LEFT JOIN|RIGHT JOIN|FULL JOIN)\b', 
                         query, re.IGNORECASE):
            raise ValueError("The query does not appear to be a join query.")

        self._validate_return_type(return_type)
        result=self.execute_query(query)
        return self._format_result(result, return_type)

    def subqueriesAndTempTables(
            self, queries: list, return_type: str = 'dataframe'
            ) -> DataFrame:
        """
        Executes a series of SQL queries typically used for creating subqueries 
        or temporary tables. 
        
        Function validates if the queries are suitable for these operations 
        before execution. The results of the last query can be returned either 
        as a DataFrame or as a raw cursor result.

        Parameters
        ----------
        queries : list of str
            A list of SQL queries to be executed in sequence.
        return_type : {'dataframe', 'raw'}, optional
            The format in which to return the results of the last query. 
            If 'dataframe', the results are returned as a pandas DataFrame. 
            If 'raw', the results are returned as they are from the database 
            cursor. Defaults to 'dataframe'.

        Returns
        -------
        pandas.DataFrame or list
            Results of the last query in the specified format. Either a 
            DataFrame or a raw list, depending on the 'return_type' parameter.

        Raises
        ------
        ValueError
            If 'return_type' is not one of the expected options ('dataframe' or 'raw').
            If the queries do not seem suitable for subqueries or temporary tables.

        Examples
        --------
        >>> from gofast.query import DBAnalysis 
        >>> db_analysis = DBAnalysis('my_database.db').fit(table_name='my_table')
        >>> queries = [
        ...     "CREATE TEMP TABLE TempTable AS SELECT * FROM MainTable WHERE condition",
        ...     "SELECT * FROM TempTable"
        ... ]
        >>> result = db_analysis.subqueriesAndTempTables(queries, 'dataframe')
        >>> print(result)

        Notes
        -----
        - This method is specifically for executing a series of queries typically used 
          for subqueries and temporary table operations.
        - Ensure that the database connection is established before calling this method.
        - The method assumes that the SQL queries are correctly formatted and valid.
        - If 'return_type' is 'dataframe', ensure pandas is installed.
        """
        self.inspect 
        subquery_pattern = r'\b(WITH|CREATE TEMP)\b|\bSELECT\b.*\bFROM\b.*\bSELECT\b'
        if not all(re.search(subquery_pattern, query, re.IGNORECASE) for query in queries):
            raise ValueError("One or more queries do not appear to be"
                             " suitable for subqueries or temporary tables.")

        if return_type not in ['dataframe', 'raw']:
            raise ValueError("Invalid return_type. Choose 'dataframe' or 'raw'.")
        results=[]
        for query in queries:
            result = self._format_result(self.execute_query(query), return_type)
            results.append(result)
        return results 

    def _format_result(self, cursor, return_type: str) -> DataFrame:
        """
        Formats the query results based on the specified return type.

        Parameters
        ----------
        cursor : Cursor
            The database cursor with the query results.
        return_type : str
            The format in which to return the query results. 'dataframe' returns a 
            pandas DataFrame, and 'raw' returns the raw results.

        Returns
        -------
        pandas.DataFrame or list
            The formatted query results.
        """
        return_type = self._validate_return_type(return_type)
        if return_type == 'dataframe':
            if hasattr(cursor, 'fetchall'):
                # Common method in both sqlite3 cursor and sqlalchemy ResultProxy
                data = cursor.fetchall()
                if hasattr(cursor, 'keys'):
                    # sqlalchemy ResultProxy
                    columns = cursor.keys()
                else:
                    # sqlite3 cursor
                    columns = [col[0] for col in cursor.description]
                return pd.DataFrame(data, columns=columns)
            else:
                raise AttributeError("The provided cursor result object does"
                                     " not support fetching data.")
        else:
            # Return raw results
            return cursor.fetchall() if hasattr(cursor, 'fetchall') else []

    def manipulate(self, query: str, auto_commit: bool = True, 
                       raise_error: bool = True ) -> None:
        """
        Executes a SQL query intended for data manipulation
        (INSERT, UPDATE, DELETE, etc.)
        and commits the transaction.
        
        The user has the option to control the commit behavior.

        Parameters
        ----------
        query : str
            The SQL data manipulation query to be executed.
        auto_commit : bool, optional
            Determines whether to automatically commit the transaction after 
            executing the query. Defaults to True. If False, the transaction 
            must be committed manually using the commit method.
        raise_error : bool, optional
            Determines whether to raise an exception if the query execution fails. 
            Defaults to True.
            
        Raises
        ------
        Exception
            If the SQL query execution fails.

        Examples
        --------
        >>> from gofast.query import DBAnalysis 
        >>> db_analysis = DBAnalysis('my_database.db').fit(table_name='my_table')
        >>> db_analysis.manipulate('INSERT INTO my_table (column1, column2)
                                       VALUES (value1, value2)')
        >>> # For manual commit
        >>> db_analysis.manipulate('INSERT INTO my_table (column1, column2) 
                                       VALUES (value1, value2)', auto_commit=False)
        >>> db_analysis.commit()

        Notes
        -----
        - This method is intended for SQL queries that manipulate data, 
          such as INSERT, UPDATE, or DELETE.
        - The auto_commit parameter allows for greater control over transaction
          management, especially useful in scenarios requiring multiple manipulation
          statements to be executed as a single transaction.
        """
        self.inspect 
        self._execute_and_commit(query, auto_commit, raise_error)
        
        return self 
            
    def commit(self) -> None:
       """
       Commits the current transaction. Useful when auto_commit is set to False in
       manipulate method.

       Examples
       --------
       >>> from gofast.query import DBAnalysis 
       >>> db_analysis = DBAnalysis('my_database.db').fit(table_name='my_table')
       >>> db_analysis.manipulate('INSERT INTO my_table (column1) VALUES (value1)',
                                      auto_commit=False)
       >>> db_analysis.commit()  # Committing the transaction manually
       """
       self.connection_.commit()
       
    def _execute_and_commit(
            self, query: str, auto_commit: bool = True, 
            raise_error: bool = True) -> None:
        """
        Helper method to execute a SQL query and handle transaction commit.

        Parameters
        ----------
        query : str
            The SQL query to be executed.
        auto_commit : bool, optional
            Determines whether to automatically commit the transaction 
            after executing the query. Defaults to True.
        raise_error : bool, optional
            Determines whether to raise an exception if the query execution fails. 
            Defaults to True.

        Raises
        ------
        ValueError
            If the query appears to be invalid.
        Exception
            If the query execution fails and raise_error is True.
        """
        # Basic check for query validity
        if not re.search(r'\b(INSERT|UPDATE|DELETE|ALTER|CREATE|DROP)\b',
                         query, re.IGNORECASE):
            raise ValueError("The query does not appear to be a valid"
                             " manipulation or transformation query.")
        try:
            self.execute_query(query)
            if auto_commit:
                self.connection_.commit()
        except Exception as e:
            if raise_error:
                raise e

    def transform(self, query: str, auto_commit: bool = True,
                      raise_error: bool = True) -> None:
        """
        Executes a SQL query intended for data transformation (ALTER, UPDATE, etc.)
        
        Method commits the transaction. The user has the option to control 
        the commit behavior.

        Parameters
        ----------
        query : str
            The SQL data transformation query to be executed.
        auto_commit : bool, optional
            Determines whether to automatically commit the transaction after
            executing the query. Defaults to True. If False, the transaction 
            must be committed  manually using the commit method.
        raise_error : bool, optional
            Determines whether to raise an exception if the query execution fails. 
            Defaults to True.
            
        Raises
        ------
        Exception
            If the SQL query execution fails.

        Examples
        --------
        >>> from gofast.query import DBAnalysis 
        >>> db_analysis = DBAnalysis('my_database.db').fit(table_name='my_table')
        >>> db_analysis.transform(
            'ALTER TABLE my_table ADD COLUMN new_column DataType')
        >>> # For manual commit
        >>> db_analysis.transform(
            'UPDATE my_table SET column1 = value1 WHERE condition', auto_commit=False)
        >>> db_analysis.commit()

        Notes
        -----
        - This method is intended for SQL queries that transform data structures, such as
          ALTER TABLE or complex UPDATE operations.
        - The auto_commit parameter allows for greater control over transaction
          management, which is particularly useful in scenarios where multiple
          transformation statements are to be executed as a single transaction.
        """
        self.inspect 
        self._execute_and_commit(query, auto_commit, raise_error)
        
        return self 

    def windowFunctions(
            self, query: str, return_type: str = 'dataframe', 
            validate_query: bool = True) -> DataFrame:
        """
        Executes a SQL query containing window functions and returns the 
        results. 
        
        The method can optionally validate if the query is likely to contain
        window functions.

        Parameters
        ----------
        query : str
            The SQL query containing window functions to be executed.
        return_type : {'dataframe', 'raw'}, optional
            The format in which to return the query results. If 'dataframe', 
            the results are returned as a pandas DataFrame. If 'raw', the 
            results are returned as they are from the database cursor. 
            Defaults to 'dataframe'.
            
        validate_query : bool, optional
            Determines whether to perform a basic validation check on the 
            query to ensure it contains window functions. Defaults to True.

        Returns
        -------
        pandas.DataFrame or list
            Query results in the specified format.

        Raises
        ------
        ValueError
            If the query does not seem to be a window function query and 
            validate_query is True.

        Examples
        --------
        >>> from gofast.query import DBAnalysis 
        >>> db_analysis = DBAnalysis('my_database.db').fit(table_name='my_table')
        >>> query = "SELECT AVG(column1) OVER (PARTITION BY column2) FROM my_table"
        >>> result = db_analysis.windowFunctions(query, 'dataframe')
        >>> print(result)

        Notes
        -----
        - Window functions in SQL are used for advanced data analysis, such 
          as calculating running totals, averages, or rankings within a 
          group of data.
        - The method assumes that the SQL query is correctly formatted and 
          valid for the specific database being used.
        """
        self.inspect 
        if validate_query and "OVER" not in query.upper():
            raise ValueError("The query does not appear to contain SQL window functions.")

        result=self.execute_query(query)
        return self._format_result(result, return_type)

    def storedProcedures(self, procedure_name: str, params: list,
                         return_type: str = 'dataframe') -> DataFrame:
        """
        Executes a stored procedure with the given parameters and returns 
        the results.
        
        The format of the results can be specified by the user.

        Parameters
        ----------
        procedure_name : str
            The name of the stored procedure to be executed.
        params : list
            A list of parameters to be passed to the stored procedure.
        return_type : {'dataframe', 'raw'}, optional
            The format in which to return the procedure results. If 'dataframe',
            the results are returned as a pandas DataFrame. If 'raw', the 
            results are returned as they are from the database cursor. 
            Defaults to 'dataframe'.

        Returns
        -------
        pandas.DataFrame or list
            The results from the stored procedure in the specified format.

        Raises
        ------
        Exception
            If the execution of the stored procedure fails.

        Examples
        --------
        >>> from gofast.query import DBAnalysis 
        >>> db_analysis = DBAnalysis('my_database.db').fit(table_name='my_table')
        >>> procedure_name = 'my_procedure'
        >>> params = [param1, param2]
        >>> result = db_analysis.storedProcedures(procedure_name, params,
                                                  'dataframe')
        >>> print(result)

        Notes
        -----
        - The method assumes that the stored procedure exists in the database 
          and the parameters are appropriate for the procedure.
        - The database user must have the necessary permissions to execute 
          stored procedures.
        - Error handling is included to catch any issues during the procedure 
          execution.
        """
        self.inspect 
        try:
            result=self.cursor_.callproc(procedure_name, params)
            return self._format_result(result, return_type)
        except Exception as e:
            raise e

    def ensureDataIntegrity(self, query: str, auto_commit: bool = True) -> None:
        """
        Executes a SQL query intended to ensure data integrity, such as 
        setting constraints or other data integrity operations, and optionally
        commits the transaction.

        Parameters
        ----------
        query : str
            The SQL query intended to ensure data integrity.
        auto_commit : bool, optional
            Determines whether to automatically commit the transaction after executing
            the query. Defaults to True. If False, the transaction must be committed
            manually using the commit method.

        Raises
        ------
        Exception
            If the SQL query execution fails.

        Examples
        --------
        >>> from gofast.query import DBAnalysis 
        >>> db_analysis = DBAnalysis('my_database.db').fit(table_name='my_table')
        >>> db_analysis.ensureDataIntegrity(
            'ALTER TABLE my_table ADD CONSTRAINT my_constraint UNIQUE (column1)')
        >>> # For manual commit
        >>> db_analysis.ensureDataIntegrity(
            'SET TRANSACTION ISOLATION LEVEL SERIALIZABLE', auto_commit=False)
        >>> db_analysis.commit()

        Notes
        -----
        - This method is intended for SQL queries that are crucial for maintaining data 
          integrity, such as adding constraints or setting transaction properties.
        - The auto_commit parameter provides control over transaction management, useful 
          in scenarios where multiple related integrity operations are to be executed 
          as a single transaction.
        """
        self.inspect 
        try:
            self.execute_query(query)
            if auto_commit:
                self.connection_.commit()
        except Exception as e:
            raise e

        return self 
    
    def scalabilityPerformance(
            self, query: str, return_type: str = 'dataframe'
            ) -> DataFrame:
        """
        Executes a SQL query intended for scalability and performance 
        analysis, returning the results in the specified format.

        Parameters
        ----------
        query : str
            The SQL query related to scalability and performance optimization.
        return_type : {'dataframe', 'raw'}, optional
            The format in which to return the query results. If 'dataframe', 
            the results are returned as a pandas DataFrame. If 'raw', the 
            results are returned as they are from the database cursor. 
            Defaults to 'dataframe'.

        Returns
        -------
        pandas.DataFrame or list
            The results of the query in the specified format.

        Examples
        --------
        >>> from gofast.query import DBAnalysis 
        >>> db_analysis = DBAnalysis('my_database.db').fit(table_name='my_table')
        >>> query = "EXPLAIN ANALYZE SELECT * FROM my_table"
        >>> result = db_analysis.scalabilityPerformance(query)
        >>> print(result)

        Notes
        -----
        - This method is particularly useful for analyzing and optimizing database 
          performance and scalability.
        - The method assumes that the SQL query is correctly formatted and 
          valid for the specific database being used.
        """
        self.inspect 
        result=self.execute_query(query)
        return self._format_result(result, return_type)

    def compatibilityIntegration(
            self, query: str, return_type: str = 'dataframe'
            ) -> DataFrame:
        """
        Executes a SQL query related to database compatibility and integration,
        returning the results in the specified format.

        Parameters
        ----------
        query : str
            The SQL query related to database compatibility and integration.
        return_type : {'dataframe', 'raw'}, optional
            The format in which to return the query results. If 'dataframe',
            the results 
            are returned as a pandas DataFrame. If 'raw', the results are 
            returned as they are from the database cursor. Defaults to 
            'dataframe'.

        Returns
        -------
        pandas.DataFrame or list
            The results of the query in the specified format.

        Examples
        --------
        >>> from gofast.query import DBAnalysis 
        >>> db_analysis = DBAnalysis('my_database.db').fit(table_name='my_table')
        >>> query = "SELECT * FROM information_schema.tables"
        >>> result = db_analysis.compatibilityIntegration(query)
        >>> print(result)

        Notes
        -----
        - This method is useful for queries that aid in assessing database 
          compatibility and integration with other systems or platforms.
        - The method assumes that the SQL query is correctly formatted and 
          valid for the specific database being used.
        """
        self.inspect 
        result=self.execute_query(query)
        return self._format_result(result, return_type)

    @property 
    def inspect(self): 
        """ Inspect data and trigger plot after checking the data entry. 
        Raises `NotFittedError` if `ExPlot` is not fitted yet."""
        
        msg = ( "{expobj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method."
               )
        if not hasattr ( self, 'cursor_') or not hasattr(self, "connection_"):  
            raise NotFittedError(msg.format(expobj=self))
        return 1
    
    def _validate_return_type (self, return_type , /): 
        """ Check the given ``return_type`` argument and returns either
        'dataframe' of 'raw'."""
        _, return_type = normalize_string(
            return_type, target_strs=['dataframe', 'raw'], 
            raise_exception= True, 
            match_method='contains', 
            return_target_str=True, 
            error_msg=("Invalid return_type. Choose 'dataframe' or 'raw'."), 
            )
        return return_type         


    def __repr__(self):
         conn_status = 'Connected' if self.connection_ else 'Disconnected'
         return (f"<DBAnalysis(db_path='{self.db_path}', verbose={self.verbose}, "
                 f"status='{conn_status}')>")     
         
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        