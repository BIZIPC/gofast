.. _simulate:

simulate
========

.. currentmodule:: gofast.datasets.simulate

The :mod:`gofast.datasets.simulate` module focuses on providing functions for simulating synthetic datasets across various domains, 
including environment, energy, customer behavior, and more. These datasets are useful for testing and validating machine learning models.

Key Features
------------
- **Environmental Simulations**:
  Generate synthetic datasets related to environmental studies.

  - :func:`~gofast.datasets.simulate.simulate_landfill_capacity`: Simulates data on landfill capacity and usage.
  - :func:`~gofast.datasets.simulate.simulate_water_reserves`: Creates a dataset for simulating water reserves.
  - :func:`~gofast.datasets.simulate.simulate_world_mineral_reserves`: Simulates world mineral reserves data.
  - :func:`~gofast.datasets.simulate.simulate_climate_data`: Generates climate data for various regions.
  - :func:`~gofast.datasets.simulate.simulate_weather_data`: Creates synthetic weather data for different locations.

- **Energy Consumption**:
  Generate datasets related to energy usage and consumption patterns.

  - :func:`~gofast.datasets.simulate.simulate_energy_consumption`: Simulates energy consumption data for households and industries.

- **Customer Behavior**:
  Simulate datasets that model customer behavior and interactions.

  - :func:`~gofast.datasets.simulate.simulate_customer_churn`: Creates a dataset to analyze customer churn.
  - :func:`~gofast.datasets.simulate.simulate_retail_sales`: Simulates retail sales data.
  - :func:`~gofast.datasets.simulate.simulate_transactions`: Generates synthetic transaction data for analysis.

- **Financial and Economic Data**:
  Create datasets related to financial markets and economic factors.

  - :func:`~gofast.datasets.simulate.simulate_stock_prices`: Simulates stock price data for multiple companies.
  - :func:`~gofast.datasets.simulate.simulate_real_estate_price`: Creates a dataset for real estate price analysis.
  - :func:`~gofast.datasets.simulate.simulate_default_loan`: Simulates loan default data for financial modeling.

- **Healthcare and Medical Data**:
  Generate synthetic datasets for healthcare and medical research.

  - :func:`~gofast.datasets.simulate.simulate_medical_diagnosis`: Simulates medical diagnosis data.
  - :func:`~gofast.datasets.simulate.simulate_patient_data`: Creates a dataset for patient health records.
  - :func:`~gofast.datasets.simulate.simulate_clinical_trials`: Simulates data for clinical trials.

- **Predictive Maintenance**:
  Generate datasets to analyze and predict maintenance needs.

  - :func:`~gofast.datasets.simulate.simulate_predictive_maintenance`: Simulates data for predictive maintenance scenarios.

- **Traffic and Transportation**:
  Create datasets for traffic flow and transportation analysis.

  - :func:`~gofast.datasets.simulate.simulate_traffic_flow`: Generates synthetic data for traffic flow studies.

- **Sentiment and Text Analysis**:
  Simulate datasets for sentiment analysis and other text-based analyses.

  - :func:`~gofast.datasets.simulate.simulate_sentiment_analysis`: Creates a dataset for sentiment analysis from text data.

Common Key Parameters
---------------------
- `<n_samples>`: Number of samples to generate.
- `<as_frame>`: If `True`, the data is returned as a pandas DataFrame. If `False`, a Bunch object is returned. By default, all functions 
  in the :mod:`~gofast.datasets.simulate` module return a Bunch object.
- `<return_X_y>`: If `True`, returns `(data, target)` instead of a Bunch object.
- `<target_name>`: Customizes the name of the target variable column.
- `<noise_level>`: Specifies the standard deviation of Gaussian noise to add to the numerical features.
- `<seed>`: Initializes the random number generator for reproducibility.

Data Handling and Attributes
----------------------------
By default, the `as_frame` parameter is `False`. However, if set to `True`, each function returns a Bunch object where 
information like `data`, `target`, `frame`, `target_names`, `feature_names`, `DESCR`, and `FDESCR` can be retrieved as attributes.

Here is an example of how to use the Bunch object:

.. code-block:: python

    from gofast.datasets.simulate import simulate_stock_prices

    stock_prices = simulate_stock_prices()

    stock_prices
    # Output: <Bunch object with keys: data, target, frame, target_names, feature_names, DESCR, FDESCR>

    print(stock_prices.DESCR)
    # Output:
    # |==============================================================================|
    # |                               Dataset Overview                               |
    # |------------------------------------------------------------------------------|
    # | Simulates stock price data for multiple companies over a specified time      |
    # | period. The dataset is generated using pseudo-random numbers to model        |
    # | stock price dynamics influenced by daily market fluctuations. The stock      |
    # | prices are generated using a geometric Brownian motion model, providing      |
    # | realistic simulations for financial analysis.                                |
    # |==============================================================================|

    print(stock_prices.FDESCR)
    # Output:
    # ================================================================================
    # |                               Dataset Features                               |
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # |Name                              | Description                               |
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # |date                              | Date of the stock price entry.            |
    # |company_id                        | Identifier for the company.               |
    # |price                             | Simulated stock price for the company.    |
    # =================================================================================


Function Descriptions
---------------------
Below are the descriptions and usage examples for each function in the 
:mod:`gofast.datasets.simulate` module:


simulate_stock_prices
~~~~~~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.simulate.simulate_stock_prices` function generates synthetic stock price data for multiple 
companies over a specified time period. This function is useful for testing and validating financial models and algorithms that 
require stock price data.

Key Parameters
^^^^^^^^^^^^^^
- `n_companies (int)`: Number of unique companies for which data will be generated. Default is 10.
- `start_date (str)`: The start date for the dataset in "YYYY-MM-DD" format. Default is "2024-01-01".
- `end_date (str)`: The end date for the dataset in "YYYY-MM-DD" format. Default is "2024-12-31".

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^
The :func:`~gofast.datasets.simulate.simulate_stock_prices` function is useful for generating synthetic stock price data to test and 
validate financial models and trading strategies. It provides realistic stock price simulations based on geometric Brownian motion, 
making it ideal for financial analysis and forecasting.

**Example 1**: Basic Stock Price Simulation

Generate a synthetic stock price dataset for 5 companies over a one-year period.

.. code-block:: python

    from gofast.datasets.simulate import simulate_stock_prices
    
    stock_prices = simulate_stock_prices(n_companies=5, start_date="2024-01-01", end_date="2024-12-31", as_frame=True)
    print(stock_prices.head())

**Example 2**: Adding Noise to Simulate Market Volatility

Generate a synthetic stock price dataset with added Gaussian noise to simulate market volatility.

.. code-block:: python

    from gofast.datasets.simulate import simulate_stock_prices
    
    stock_prices = simulate_stock_prices(n_companies=3, noise_level=0.02, as_frame=True)
    print(stock_prices.head())

simulate_transactions
~~~~~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.simulate.simulate_transactions` function generates synthetic financial transaction data for multiple 
accounts over a specified time period. This function is useful for testing and validating financial models and algorithms that 
require transaction data.

Key Parameters
^^^^^^^^^^^^^^
- `n_accounts (int)`: Number of unique accounts for which data will be generated. Default is 100.
- `start_date (str)`: The start date for the dataset in "YYYY-MM-DD" format. Default is "2024-01-01".
- `end_date (str)`: The end date for the dataset in "YYYY-MM-DD" format. Default is "2024-12-31".

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^
The :func:`~gofast.datasets.simulate.simulate_transactions` function is useful for generating synthetic financial transaction data 
to test and validate financial models and fraud detection algorithms. It provides realistic transaction simulations, including debits 
and credits, making it ideal for financial analysis and forecasting.

**Example 1**: Basic Transaction Simulation

Generate a synthetic transaction dataset for 10 accounts over a one-year period.

.. code-block:: python

    from gofast.datasets.simulate import simulate_transactions
    
    transactions = simulate_transactions(n_accounts=10, start_date="2024-01-01", end_date="2024-12-31", as_frame=True)
    print(transactions.head())

**Example 2**: Adding Noise to Simulate Real-World Financial Variations

Generate a synthetic transaction dataset with added Gaussian noise to simulate real-world financial variations.

.. code-block:: python

    from gofast.datasets.simulate import simulate_transactions
    
    transactions = simulate_transactions(n_accounts=5, noise_level=0.05, as_frame=True)
    print(transactions.head())

simulate_patient_data
~~~~~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.simulate.simulate_patient_data` function generates synthetic patient data including demographics, medical 
history, and test results over a specified time period. This function is useful for testing and validating healthcare models and 
algorithms that require patient data.

Key Parameters
^^^^^^^^^^^^^^
- `n_patients (int)`: Number of unique patients for whom data will be generated. Default is 100.
- `start_date (str)`: The start date for the dataset in "YYYY-MM-DD" format. Default is "2024-01-01".
- `end_date (str)`: The end date for the dataset in "YYYY-MM-DD" format. Default is "2024-12-31".

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^
The :func:`~gofast.datasets.simulate.simulate_patient_data` function is useful for generating synthetic patient data to test and validate 
healthcare models, patient monitoring systems, and medical algorithms. It provides realistic patient data simulations, making it ideal 
for healthcare analysis and forecasting.

**Example 1**: Basic Patient Data Simulation

Generate a synthetic patient dataset for 50 patients over a one-year period.

.. code-block:: python

    from gofast.datasets.simulate import simulate_patient_data
    
    patient_data = simulate_patient_data(n_patients=50, start_date="2024-01-01", end_date="2024-12-31", as_frame=True)
    print(patient_data.head())

**Example 2**: Adding Noise to Simulate Measurement Errors

Generate a synthetic patient dataset with added Gaussian noise to simulate measurement errors.

.. code-block:: python

    from gofast.datasets.simulate import simulate_patient_data
    
    patient_data = simulate_patient_data(n_patients=30, noise_level=0.05, as_frame=True)
    print(patient_data.head())

**Example 3**: Generating Data for a Shorter Period

Generate a synthetic patient dataset for 20 patients over a six-month period.

.. code-block:: python

    from gofast.datasets.simulate import simulate_patient_data
    
    patient_data = simulate_patient_data(n_patients=20, start_date="2024-01-01", end_date="2024-06-30", as_frame=True)
    print(patient_data.head())

simulate_clinical_trials
~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.simulate.simulate_clinical_trials` function generates synthetic clinical trial data for multiple patients 
over a specified time period. This function is useful for testing and validating clinical trial models and algorithms that require 
detailed patient and trial data.

Key Parameters
^^^^^^^^^^^^^^
- `n_patients (int)`: Number of unique patients for whom data will be generated. Default is 100.
- `start_date (str)`: The start date for the dataset in "YYYY-MM-DD" format. Default is "2024-01-01".
- `end_date (str)`: The end date for the dataset in "YYYY-MM-DD" format. Default is "2024-12-31".

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`~gofast.datasets.simulate.simulate_clinical_trials` function is useful for generating synthetic clinical trial data to test 
and validate clinical trial models, treatment effectiveness analysis, and medical research. It provides realistic trial data simulations, 
making it ideal for healthcare research and forecasting.

**Example 1**: Basic Clinical Trial Simulation

Generate a synthetic clinical trial dataset for 50 patients over a one-year period.

.. code-block:: python

    from gofast.datasets.simulate import simulate_clinical_trials
    
    clinical_trials = simulate_clinical_trials(n_patients=50, start_date="2024-01-01", end_date="2024-12-31", as_frame=True)
    print(clinical_trials.head())

**Example 2**: Adding Noise to Simulate Real-World Variations

Generate a synthetic clinical trial dataset with added Gaussian noise to simulate real-world variations.

.. code-block:: python

    from gofast.datasets.simulate import simulate_clinical_trials
    
    clinical_trials = simulate_clinical_trials(n_patients=30, noise_level=0.05, as_frame=True)
    print(clinical_trials.head())

**Example 3**: Generating Data for a Shorter Period

Generate a synthetic clinical trial dataset for 20 patients over a six-month period.

.. code-block:: python

    from gofast.datasets.simulate import simulate_clinical_trials
    
    clinical_trials = simulate_clinical_trials(n_patients=20, start_date="2024-01-01", end_date="2024-06-30", as_frame=True)
    print(clinical_trials.head())

simulate_weather_data
~~~~~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.simulate.simulate_weather_data` function generates synthetic weather data including temperature, humidity, 
and precipitation for multiple locations over a specified time period. This function is useful for testing and validating weather 
forecasting models and algorithms that require weather data.

Key Parameters
^^^^^^^^^^^^^^
- `n_locations (int)`: Number of unique locations for which data will be generated. Default is 100.
- `start_date (str)`: The start date for the dataset in "YYYY-MM-DD" format. Default is "2024-01-01".
- `end_date (str or None)`: The end date for the dataset in "YYYY-MM-DD" format. If `None`, the end date is set to the current date. Default is `None`.
- `n_days (int or None)`: Number of days for which data will be generated. If specified, `end_date` is ignored, and data is generated for `n_days` 
  starting from `start_date`. Default is `None`.

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^
The :func:`~gofast.datasets.simulate.simulate_weather_data` function is useful for generating synthetic weather data to test and validate 
weather forecasting models and climate studies. It provides realistic weather simulations based on typical daily variations, making it 
ideal for environmental analysis and prediction.

**Example 1**: Basic Weather Data Simulation

Generate a synthetic weather dataset for 50 locations over a one-year period.

.. code-block:: python

    from gofast.datasets.simulate import simulate_weather_data
    
    weather_data = simulate_weather_data(n_locations=50, start_date="2024-01-01", end_date="2024-12-31", as_frame=True)
    print(weather_data.head())

**Example 2**: Adding Noise to Simulate Measurement Errors

Generate a synthetic weather dataset with added Gaussian noise to simulate measurement errors.

.. code-block:: python

    from gofast.datasets.simulate import simulate_weather_data
    
    weather_data = simulate_weather_data(n_locations=30, noise_level=0.05, as_frame=True)
    print(weather_data.head())

**Example 3**: Generating Data for a Specific Number of Days

Generate a synthetic weather dataset for 20 locations over a six-month period.

.. code-block:: python

    from gofast.datasets.simulate import simulate_weather_data
    
    weather_data = simulate_weather_data(n_locations=20, start_date="2024-01-01", n_days=180, as_frame=True)
    print(weather_data.head())

simulate_climate_data
~~~~~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.simulate.simulate_climate_data` function generates synthetic climate data for long-term environmental 
studies. This function is useful for testing and validating climate models and algorithms that require long-term climate data.

Key Parameters
^^^^^^^^^^^^^^
- `n_locations (int)`: Number of unique locations for which data will be generated. Default is 100.
- `n_years (int)`: Number of years for which data will be generated. Default is 30.
- `start_year (int)`: The starting year for the dataset. Default is 1990.

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^
The :func:`~gofast.datasets.simulate.simulate_climate_data` function is useful for generating synthetic climate data to test 
and validate climate models and environmental studies. It provides realistic long-term climate simulations based on typical annual 
variations, making it ideal for long-term environmental analysis and prediction.

**Example 1**: Basic Climate Data Simulation

Generate a synthetic climate dataset for 50 locations over 30 years starting from 1990.

.. code-block:: python

    from gofast.datasets.simulate import simulate_climate_data
    
    climate_data = simulate_climate_data(n_locations=50, n_years=30, start_year=1990, as_frame=True)
    print(climate_data.head())

**Example 2**: Adding Noise to Simulate Environmental Fluctuations

Generate a synthetic climate dataset with added Gaussian noise to simulate environmental fluctuations.

.. code-block:: python

    from gofast.datasets.simulate import simulate_climate_data
    
    climate_data = simulate_climate_data(n_locations=30, noise_level=0.05, as_frame=True)
    print(climate_data.head())

**Example 3**: Generating Data for a Different Time Period

Generate a synthetic climate dataset for 20 locations over 20 years starting from 2000.

.. code-block:: python

    from gofast.datasets.simulate import simulate_climate_data
    
    climate_data = simulate_climate_data(n_locations=20, n_years=20, start_year=2000, as_frame=True)
    print(climate_data.head())

simulate_landfill_capacity
~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.simulate.simulate_landfill_capacity` function generates synthetic datasets of landfill capacity measurements across various locations over a specified time period. This function is useful for testing and validating machine learning models and algorithms that require landfill data.

Key Parameters
^^^^^^^^^^^^^^
- `n_landfills (int)`: The number of unique landfill sites for which data will be generated. Default is 100.
- `start_date (str)`: The start date for the dataset in "YYYY-MM-DD" format. Default is "2024-01-01".
- `end_date (str)`: The end date for the dataset in "YYYY-MM-DD" format. Default is "2024-01-31".
- `task (str)`: Determines the nature of the target variable(s) for the dataset. "regression" or "classification". Default is "regression".

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^
The :func:`~gofast.datasets.simulate.simulate_landfill_capacity` function is useful for generating synthetic landfill data to test and validate waste management models and environmental sustainability studies. It provides realistic landfill capacity simulations, making it ideal for machine learning applications in these domains.

**Example 1**: Basic Landfill Capacity Simulation

Generate a synthetic landfill dataset for 10 landfills over a one-month period.

.. code-block:: python

    from gofast.datasets.simulate import simulate_landfill_capacity
    
    landfill_data = simulate_landfill_capacity(n_landfills=10, start_date="2024-01-01", end_date="2024-01-31", as_frame=True)
    print(landfill_data.head())

**Example 2**: Generating Data for Classification Task

Generate a synthetic landfill dataset with categorical targets for classification.

.. code-block:: python

    from gofast.datasets.simulate import simulate_landfill_capacity
    
    landfill_data = simulate_landfill_capacity(n_landfills=50, start_date="2024-06-01", end_date="2024-06-30", task="classification", as_frame=True)
    print(landfill_data.head())

**Example 3**: Adding Noise to Simulate Measurement Errors

Generate a synthetic landfill dataset with added Gaussian noise to simulate measurement errors.

.. code-block:: python

    from gofast.datasets.simulate import simulate_landfill_capacity
    
    landfill_data = simulate_landfill_capacity(noise_level=0.05, as_frame=True)
    print(landfill_data.head())

simulate_water_reserves
~~~~~~~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.simulate.simulate_water_reserves` function generates synthetic datasets of water reserve measurements across various locations over a specified time period. This function is useful for testing and validating machine learning models and algorithms that require water reserve data.

Key Parameters
^^^^^^^^^^^^^^
- `n_locations (int)`: Number of unique locations for which data will be generated. Default is 100.
- `start_date (str)`: The start date for the dataset in "YYYY-MM-DD" format. Default is "2024-01-01".
- `end_date (str)`: The end date for the dataset in "YYYY-MM-DD" format. Default is "2024-01-31".

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^
The :func:`~gofast.datasets.simulate.simulate_water_reserves` function is useful for generating synthetic water reserve data to test 
and validate environmental models and water resource management strategies. It provides realistic water reserve simulations, making 
it ideal for environmental analysis and prediction.

**Example 1**: Basic Water Reserve Data Simulation

Generate a synthetic water reserve dataset for 10 locations over a one-month period.

.. code-block:: python

    from gofast.datasets.simulate import simulate_water_reserves
    
    water_reserves = simulate_water_reserves(n_locations=10, start_date="2024-01-01", end_date="2024-01-31", as_frame=True)
    print(water_reserves.head())

**Example 2**: Generating Data with Specific Number of Samples

Generate a synthetic water reserve dataset with a specified number of samples.

.. code-block:: python

    from gofast.datasets.simulate import simulate_water_reserves
    
    water_reserves = simulate_water_reserves(n_samples=1000, as_frame=True)
    print(water_reserves.head())

**Example 3**: Adding Noise to Simulate Measurement Errors

Generate a synthetic water reserve dataset with added Gaussian noise to simulate measurement errors.

.. code-block:: python

    from gofast.datasets.simulate import simulate_water_reserves
    
    water_reserves = simulate_water_reserves(noise_level=0.05, as_frame=True)
    print(water_reserves.head())

simulate_world_mineral_reserves
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.simulate.simulate_world_mineral_reserves` function generates synthetic datasets of world mineral reserves, providing insights into global mineral production. This function allows for the generation of data reflecting mineral reserve quantities across different countries and regions, incorporating economic impact factors and statistical noise to mimic real-world variability.

Key Parameters
^^^^^^^^^^^^^^
- `regions (list of str)`: Filters the simulation to include countries within specified geographical regions. Default is None.
- `distributions (dict)`: Custom mapping of regions to mineral types for targeted simulation scenarios. Default is None.
- `mineral_types (list of str)`: Filters the dataset to simulate reserves for specified minerals. Default is None.
- `countries (list of str)`: Specifies a list of countries to be included in the simulation. Default is None.
- `economic_impact_factor (float)`: Adjusts the simulated quantity of reserves based on economic conditions. Default is 0.05.
- `default_location (str)`: Placeholder for the location when a mineral's producing country is undetermined. Default is 'Global HQ'.

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^
The :func:`~gofast.datasets.simulate.simulate_world_mineral_reserves` function is useful for generating synthetic mineral reserve data to test and validate geoscientific models and economic impact analyses. It provides realistic mineral reserve simulations, making it ideal for geoscientific research and policy-making.

**Example 1**: Basic Mineral Reserves Simulation

Generate a simple dataset of mineral reserves.

.. code-block:: python

    from gofast.datasets.simulate import simulate_world_mineral_reserves
    
    mineral_reserves = simulate_world_mineral_reserves(n_samples=100, as_frame=True)
    print(mineral_reserves.head())

**Example 2**: Focusing on Specific Minerals and Regions

Generate a synthetic mineral reserve dataset focusing on gold and diamonds in Africa and Asia.

.. code-block:: python

    from gofast.datasets.simulate import simulate_world_mineral_reserves
    
    mineral_reserves = simulate_world_mineral_reserves(regions=['Africa', 'Asia'], mineral_types=['gold', 'diamond'], n_samples=100, as_frame=True)
    print(mineral_reserves.head())

**Example 3**: Handling Undetermined Production Countries

Generate a synthetic mineral reserve dataset with a custom default location for undetermined production countries.

.. code-block:: python

    from gofast.datasets.simulate import simulate_world_mineral_reserves
    
    X, y = simulate_world_mineral_reserves(default_location='Research Center', noise_level=0.1, seed=42, return_X_y=True)
    print(len(y))

simulate_energy_consumption
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.simulate.simulate_energy_consumption` function generates synthetic datasets representing energy 
consumption across multiple households over a specified time frame. This function is useful for testing and validating machine 
learning models and algorithms that require energy consumption data.

Key Parameters
^^^^^^^^^^^^^^
- `n_households (int)`: The total number of households for which the energy consumption is to be simulated. Default is 10.
- `days (int)`: The span of days across which the energy consumption is simulated, starting from `start_date`. Default is 365.
- `start_date (str)`: The commencement date of the simulation period, formatted as 'YYYY-MM-DD'. Default is '2021-01-01'.

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^
The :func:`~gofast.datasets.simulate.simulate_energy_consumption` function is useful for generating synthetic energy consumption data 
to test and validate predictive models and energy usage analyses. It provides realistic energy consumption simulations, making it ideal 
for energy studies and smart grid research.

**Example 1**: Basic Energy Consumption Simulation

Generate a synthetic energy consumption dataset for 100 households over a 30-day period.

.. code-block:: python

    from gofast.datasets.simulate import simulate_energy_consumption
    
    energy_data = simulate_energy_consumption(n_households=100, days=30, as_frame=True)
    print(energy_data.head())

**Example 2**: Generating Data with Specific Number of Samples

Generate a synthetic energy consumption dataset with a specified number of samples.

.. code-block:: python

    from gofast.datasets.simulate import simulate_energy_consumption
    
    energy_data = simulate_energy_consumption(n_samples=1000, as_frame=True)
    print(energy_data.head())

**Example 3**: Adding Noise to Simulate Measurement Errors

Generate a synthetic energy consumption dataset with added Gaussian noise to simulate measurement errors.

.. code-block:: python

    from gofast.datasets.simulate import simulate_energy_consumption
    
    energy_data = simulate_energy_consumption(noise_level=0.05, as_frame=True)
    print(energy_data.head())
    
simulate_customer_churn
~~~~~~~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.simulate.simulate_customer_churn` function generates synthetic datasets for customer 
churn prediction based on demographic information, service usage patterns, and other relevant customer features. 
This simulation aims to provide a realistic set of data for modeling customer churn in various services or 
subscription-based business models.

Key Parameters
^^^^^^^^^^^^^^
- `n_customers (int)`: The number of customer records to generate in the dataset. Default is 1000.

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^
The :func:`~gofast.datasets.simulate.simulate_customer_churn` function is useful for generating synthetic customer churn data to test and validate machine learning models and algorithms. It provides realistic churn prediction data, making it ideal for customer retention studies and business strategy development.

**Example 1**: Basic Customer Churn Simulation

Generate a synthetic customer churn dataset for 500 customers.

.. code-block:: python

    from gofast.datasets.simulate import simulate_customer_churn
    
    churn_data = simulate_customer_churn(n_customers=500, as_frame=True)
    print(churn_data.head())

**Example 2**: Generating Data with Specific Noise Level

Generate a synthetic customer churn dataset with added Gaussian noise to simulate data variability.

.. code-block:: python

    from gofast.datasets.simulate import simulate_customer_churn
    
    churn_data = simulate_customer_churn(noise_level=0.1, as_frame=True)
    print(churn_data.head())

**Example 3**: Splitting Data into Features and Target

Generate a synthetic customer churn dataset and split it into features and target for machine learning.

.. code-block:: python

    from gofast.datasets.simulate import simulate_customer_churn
    
    X, y = simulate_customer_churn(return_X_y=True)
    print(X.shape, y.shape)

simulate_predictive_maintenance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.simulate.simulate_predictive_maintenance` function generates synthetic datasets tailored for predictive maintenance tasks, offering detailed insights into the operational dynamics and maintenance requirements of a fleet of machines over a specified period.

Key Parameters
^^^^^^^^^^^^^^
- `n_machines (int)`: Specifies the number of individual machines included in the simulation. Default is 25.
- `n_sensors (int)`: Denotes the quantity of distinct sensors installed per machine. Default is 5.
- `operational_params (int)`: Represents the count of operational parameters critical to assessing machine performance. Default is 2.
- `days (int)`: Defines the total number of days across which the simulation spans. Default is 30.
- `start_date (str)`: Marks the commencement date of the dataset. Default is '2021-01-01'.
- `failure_rate (float)`: The estimated daily probability of a failure occurrence per machine. Default is 0.02.
- `maintenance_frequency (int)`: Specifies the interval in days between scheduled maintenance activities. Default is 45.
- `task (str)`: Determines the primary objective of the simulated dataset ('classification' or 'regression'). Default is 'classification'.

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^
The :func:`~gofast.datasets.simulate.simulate_predictive_maintenance` function is useful for generating synthetic predictive 
maintenance data to test and validate machine learning models and algorithms. It provides realistic maintenance data, making it 
ideal for maintenance scheduling and fault prediction studies.

**Example 1**: Basic Predictive Maintenance Simulation

Generate a synthetic predictive maintenance dataset for 100 machines for a regression task.

.. code-block:: python

    from gofast.datasets.simulate import simulate_predictive_maintenance
    
    maintenance_data = simulate_predictive_maintenance(n_machines=100, task='regression', as_frame=True)
    print(maintenance_data.head())

**Example 2**: Generating Data with Specific Number of Samples

Generate a synthetic predictive maintenance dataset with a specified number of samples.

.. code-block:: python

    from gofast.datasets.simulate import simulate_predictive_maintenance
    
    maintenance_data = simulate_predictive_maintenance(n_samples=1000, as_frame=True)
    print(maintenance_data.head())

**Example 3**: Adding Noise to Simulate Measurement Errors

Generate a synthetic predictive maintenance dataset with added Gaussian noise to simulate measurement errors.

.. code-block:: python

    from gofast.datasets.simulate import simulate_predictive_maintenance
    
    maintenance_data = simulate_predictive_maintenance(noise_level=0.05, as_frame=True)
    print(maintenance_data.head())

simulate_real_estate_price
~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.simulate.simulate_real_estate_price` function generates synthetic datasets for real estate price prediction. This dataset includes various property features, economic indicators, and temporal dynamics to create realistic market value estimations for residential properties over a specified time frame.

Key Parameters
^^^^^^^^^^^^^^
- `n_properties (int)`: The number of residential properties to include in the simulation. Default is 1000.
- `features (list of str)`: Characteristics of properties to simulate, such as size, number of bedrooms, and age. Defaults to common features.
- `economic_indicators (list of str)`: Economic indicators to simulate alongside property features, like interest rates and GDP growth rate.
- `start_year (int)`: The starting year for the simulated price data. Default is 2000.
- `years (int)`: The duration in years over which the real estate price data is simulated. Default is 20.
- `price_increase_rate (float)`: An annual rate at which property prices increase. Default is 0.03.

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^
The :func:`~gofast.datasets.simulate.simulate_real_estate_price` function is useful for generating synthetic real estate price data to test and validate machine learning models for price prediction and trend analysis. It provides realistic property price simulations, making it ideal for real estate market studies.

**Example 1**: Basic Real Estate Price Simulation

Generate a synthetic real estate price dataset for 500 properties over 10 years.

.. code-block:: python

    from gofast.datasets.simulate import simulate_real_estate_price
    
    real_estate_data = simulate_real_estate_price(n_properties=500, years=10, as_frame=True)
    print(real_estate_data.head())

**Example 2**: Adding Economic Indicators

Generate a synthetic real estate price dataset with specific economic indicators.

.. code-block:: python

    from gofast.datasets.simulate import simulate_real_estate_price
    
    real_estate_data = simulate_real_estate_price(economic_indicators=['interest_rate', 'GDP_growth_rate'], as_frame=True)
    print(real_estate_data.head())

**Example 3**: Splitting Data into Features and Target

Generate a synthetic real estate price dataset and split it into features and target for machine learning.

.. code-block:: python

    from gofast.datasets.simulate import simulate_real_estate_price
    
    X, y = simulate_real_estate_price(return_X_y=True)
    print(X.shape, y.shape)

simulate_sentiment_analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.simulate.simulate_sentiment_analysis` function generates synthetic datasets for sentiment analysis tasks. It focuses on classifying the sentiment of product reviews into positive, neutral, or negative categories, creating realistic review texts and corresponding sentiment labels.

Key Parameters
^^^^^^^^^^^^^^
- `n_reviews (int)`: The number of product reviews to generate in the dataset. Default is 1000.
- `review_length_range (tuple of int)`: Specifies the minimum and maximum length of the reviews, measured in number of words. Default is (50, 300).
- `sentiment_distribution (tuple of float)`: Represents the distribution of positive, neutral, and negative sentiments among the reviews. Default is (0.4, 0.2, 0.4).

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^
The :func:`~gofast.datasets.simulate.simulate_sentiment_analysis` function is useful for generating synthetic sentiment analysis data to test and validate natural language processing models. It provides realistic product review texts, making it ideal for sentiment classification studies.

**Example 1**: Basic Sentiment Analysis Simulation

Generate a synthetic sentiment analysis dataset for 500 reviews.

.. code-block:: python

    from gofast.datasets.simulate import simulate_sentiment_analysis
    
    sentiment_data = simulate_sentiment_analysis(n_reviews=500, as_frame=True)
    print(sentiment_data.head())

**Example 2**: Custom Review Length Range

Generate a synthetic sentiment analysis dataset with a custom review length range.

.. code-block:: python

    from gofast.datasets.simulate import simulate_sentiment_analysis
    
    sentiment_data = simulate_sentiment_analysis(review_length_range=(100, 500), as_frame=True)
    print(sentiment_data.head())

**Example 3**: Splitting Data into Features and Target

Generate a synthetic sentiment analysis dataset and split it into features and target for machine learning.

.. code-block:: python

    from gofast.datasets.simulate import simulate_sentiment_analysis
    
    X, y = simulate_sentiment_analysis(return_X_y=True)
    print(X.shape, y.shape)

simulate_weather_forecasting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.simulate.simulate_weather_forecasting` function generates synthetic datasets tailored for weather 
forecasting tasks, simulating atmospheric conditions and weather variables over a defined period.

Key Parameters
^^^^^^^^^^^^^^
- `n_days (int)`: The total number of days for which weather data is simulated. Default is 365.
- `weather_variables (list of str)`: A list of weather variables to include in the simulation, such as 'temperature', 'humidity', and 'precipitation'. 
  Default is a set of common variables.
- `start_date (str)`: The starting date of the weather data simulation. Default is "2020-01-01".
- `include_extreme_events (bool)`: If True, the simulation includes random extreme weather events. Default is False.

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^
The :func:`~gofast.datasets.simulate.simulate_weather_forecasting` function is useful for generating synthetic weather data to test and validate machine learning models 
for weather prediction and analysis. It provides realistic weather simulations, making it ideal for atmospheric and climate studies.

**Example 1**: Basic Weather Forecasting Simulation

Generate a synthetic weather dataset for 30 days with extreme weather events included.

.. code-block:: python

    from gofast.datasets.simulate import simulate_weather_forecasting
    
    weather_data = simulate_weather_forecasting(n_days=30, include_extreme_events=True, as_frame=True)
    print(weather_data.head())

**Example 2**: Custom Weather Variables

Generate a synthetic weather dataset with specific weather variables.

.. code-block:: python

    from gofast.datasets.simulate import simulate_weather_forecasting
    
    weather_data = simulate_weather_forecasting(weather_variables=['temperature', 'wind_speed'], as_frame=True)
    print(weather_data.head())

**Example 3**: Splitting Data into Features and Target

Generate a synthetic weather dataset and split it into features and target for machine learning.

.. code-block:: python

    from gofast.datasets.simulate import simulate_weather_forecasting
    
    X, y = simulate_weather_forecasting(return_X_y=True)
    print(X.shape, y.shape)

simulate_default_loan
~~~~~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.simulate.simulate_default_loan` function generates synthetic datasets for loan default prediction based on 
borrower profiles and loan characteristics. This dataset supports binary classification tasks where the target is predicting the 
likelihood of loan defaults.

Key Parameters
^^^^^^^^^^^^^^
- `n_samples (int)`: The number of loan samples to generate. Default is 1000.
- `credit_score_range (tuple of int)`: The range of credit scores to generate. Default is (300, 850).
- `age_range (tuple of int)`: The range of borrower ages to generate. Default is (18, 70).
- `loan_amount_range (tuple of int)`: The range of loan amounts ($) to generate. Default is (5000, 50000).
- `interest_rate_range (tuple of float)`: The range of interest rates (as a percentage) to generate. Default is (5, 20).
- `loan_term_months (list of int)`: A list of possible loan terms in months. Default is [12, 24, 36, 48, 60].
- `employment_length_range (tuple of int)`: The range of employment lengths (in years) to generate. Default is (0, 30).
- `annual_income_range (tuple of int)`: The range of annual incomes ($) to generate. Default is (20000, 150000).
- `default_rate (float)`: The probability that a loan will default. Default is 0.15.

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^
The :func:`~gofast.datasets.simulate.simulate_default_loan` function is useful for generating synthetic loan default data to test and 
validate machine learning models for credit risk assessment and loan default prediction. It provides realistic borrower profiles and 
loan characteristics, making it ideal for financial studies.

**Example 1**: Basic Loan Default Simulation

Generate a synthetic loan default dataset for 500 samples.

.. code-block:: python

    from gofast.datasets.simulate import simulate_default_loan
    
    loan_data = simulate_default_loan(n_samples=500, as_frame=True)
    print(loan_data.head())

**Example 2**: Custom Loan Characteristics

Generate a synthetic loan default dataset with specific ranges for loan characteristics.

.. code-block:: python

    from gofast.datasets.simulate import simulate_default_loan
    
    loan_data = simulate_default_loan(credit_score_range=(400, 800), loan_amount_range=(10000, 40000), as_frame=True)
    print(loan_data.head())

**Example 3**: Splitting Data into Features and Target

Generate a synthetic loan default dataset and split it into features and target for machine learning.

.. code-block:: python

    from gofast.datasets.simulate import simulate_default_loan
    
    X, y = simulate_default_loan(return_X_y=True)
    print(X.shape, y.shape)
simulate_traffic_flow
~~~~~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.simulate.simulate_traffic_flow` function generates synthetic datasets for traffic 
flow prediction, covering various aspects of urban traffic dynamics over a specified period.

Key Parameters
^^^^^^^^^^^^^^
- `n_samples (int)`: The total number of traffic observations to generate. Default is 10,000.
- `start_date (str)`: The beginning of the simulation period. Default is "2021-01-01 00:00:00".
- `end_date (str)`: The end of the simulation period. Default is "2021-12-31 23:59:59".
- `traffic_flow_range (tuple)`: The range of traffic flow rates to simulate. Default is (100, 1000).
- `time_increments (str)`: The granularity of time intervals for each traffic observation. Default is 'hour'.
- `special_events_probability (float)`: The likelihood of a special event occurring. Default is 0.05.
- `road_closure_probability (float)`: The chance of road closures. Default is 0.01.
- `noise_level (float)`: The standard deviation of Gaussian noise added to the traffic data. Default is 0.1.

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^
The :func:`~gofast.datasets.simulate.simulate_traffic_flow` function is useful for generating synthetic traffic data to test and 
validate machine learning models for traffic prediction and congestion management. It provides realistic traffic flow simulations, 
making it ideal for urban planning and transportation studies.

**Example 1**: Basic Traffic Flow Simulation

Generate a synthetic traffic dataset with 500 samples and a higher probability of special events.

.. code-block:: python

    from gofast.datasets.simulate import simulate_traffic_flow
    
    traffic_data = simulate_traffic_flow(n_samples=500, special_events_probability=0.1, as_frame=True)
    print(traffic_data.head())

**Example 2**: Custom Time Increments

Generate a synthetic traffic dataset with quarter-hour time increments.

.. code-block:: python

    from gofast.datasets.simulate import simulate_traffic_flow
    
    traffic_data = simulate_traffic_flow(time_increments='quarter-hour', as_frame=True)
    print(traffic_data.head())

**Example 3**: Splitting Data into Features and Target

Generate a synthetic traffic dataset and split it into features and target for machine learning.

.. code-block:: python

    from gofast.datasets.simulate import simulate_traffic_flow
    
    X, y = simulate_traffic_flow(return_X_y=True)
    print(X.shape, y.shape)

simulate_medical_diagnosis
~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.simulate.simulate_medical_diagnosis` function generates synthetic datasets to simulate patient medical 
diagnosis data, including symptoms, lab test results, and diagnosis outcomes.

Key Parameters
^^^^^^^^^^^^^^
- `n_patients (int)`: The number of patients to simulate. Default is 1,000.
- `n_symptoms (int)`: The number of different symptoms to simulate for each patient. Default is 10.
- `n_lab_tests (int)`: The number of lab test results to simulate for each patient. Default is 5.
- `diagnosis_options (list of str)`: The list of possible diagnosis outcomes. Default is ['Disease A', 'Disease B', 'Healthy'].

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^
The :func:`~gofast.datasets.simulate.simulate_medical_diagnosis` function is useful for generating synthetic medical data to test 
and validate machine learning models for medical diagnostics and decision support. It provides realistic patient data, making it 
ideal for healthcare studies.

**Example 1**: Basic Medical Diagnosis Simulation

Generate a synthetic medical diagnosis dataset for 500 patients.

.. code-block:: python

    from gofast.datasets.simulate import simulate_medical_diagnosis
    
    medical_data = simulate_medical_diagnosis(n_patients=500, as_frame=True)
    print(medical_data.head())

**Example 2**: Custom Diagnosis Options

Generate a synthetic medical diagnosis dataset with specific diagnosis options.

.. code-block:: python

    from gofast.datasets.simulate import simulate_medical_diagnosis
    
    medical_data = simulate_medical_diagnosis(diagnosis_options=['Disease X', 'Disease Y', 'Healthy'], as_frame=True)
    print(medical_data.head())

**Example 3**: Splitting Data into Features and Target

Generate a synthetic medical diagnosis dataset and split it into features and target for machine learning.

.. code-block:: python

    from gofast.datasets.simulate import simulate_medical_diagnosis
    
    X, y = simulate_medical_diagnosis(return_X_y=True)
    print(X.shape, y.shape)

simulate_retail_sales
~~~~~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.simulate.simulate_retail_sales` function generates a synthetic dataset for retail sales forecasting, incorporating factors like product promotions, seasonal trends, and economic indicators. This dataset is designed for regression tasks aimed at predicting future sales volumes based on historical data and external influences.

Key Parameters
^^^^^^^^^^^^^^
- `n_products (int)`: Specifies the total number of unique products included in the simulation. Default is 100.
- `n_days (int)`: Defines the duration, in days, for which the sales data is generated. Default is 365.
- `include_promotions (bool)`: Introduces promotional activities into the dataset when set to True. Default is True.
- `include_seasonality (bool)`: Simulates seasonal variations in sales data when set to True. Default is True.
- `complex_seasonality (bool)`: Simulates complex seasonal patterns in sales data when set to True. Default is False.
- `include_economic_factors (bool)`: Includes economic variables that might affect retail sales when set to True. Default is True.
- `as_frame (bool)`: Returns the dataset as a pandas DataFrame when set to True. Default is False.
- `target_name (str, optional)`: Customizes the column name for the target variable. Default is 'future_sales'.

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`~gofast.datasets.simulate.simulate_retail_sales` function is useful for generating synthetic retail sales data to 
test and validate machine learning models for sales forecasting. It provides realistic sales simulations, making it ideal for 
retail analytics and market analysis.

**Example 1**: Basic Retail Sales Simulation

Generate a synthetic retail sales dataset for 50 products over 180 days.

.. code-block:: python

    from gofast.datasets.simulate import simulate_retail_sales
    
    retail_data = simulate_retail_sales(n_products=50, n_days=180, as_frame=True)
    print(retail_data.head())

**Example 2**: Including Promotions and Seasonality

Generate a synthetic retail sales dataset with promotions and seasonality included.

.. code-block:: python

    from gofast.datasets.simulate import simulate_retail_sales
    
    retail_data = simulate_retail_sales(
        include_promotions=True, include_seasonality=True, as_frame=True)
    print(retail_data.head())

**Example 3**: Splitting Data into Features and Target

Generate a synthetic retail sales dataset and split it into features and target for machine learning.

.. code-block:: python

    from gofast.datasets.simulate import simulate_retail_sales
    
    X, y = simulate_retail_sales(return_X_y=True)
    print(X.shape, y.shape)

**Example 4**: Custom Noise Level

Generate a synthetic retail sales dataset with a custom noise level.

.. code-block:: python

    from gofast.datasets.simulate import simulate_retail_sales
    
    retail_data = simulate_retail_sales(noise_level=0.1, as_frame=True)
    print(retail_data.head())
