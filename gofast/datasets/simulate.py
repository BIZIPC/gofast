# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
import inspect 
import pandas as pd
import numpy as np

from ..tools.coreutils import is_iterable, unpack_list_of_dicts  
from ..tools.validator import validate_dates, validate_positive_integer 
from ..tools.validator import validate_distribution, validate_length_range
from .util import validate_region, adjust_parameters_to_fit_samples 
from .util import find_mineral_distributions, find_countries_by_minerals
from .util import extract_minerals_from_regions, build_distributions_from
from .util import check_distributions, select_location_for_mineral, manage_data
from .util import generate_ore_infos, build_reserve_details_by_country
from .util import get_last_day_of_current_month, adjust_households_and_days
from .util import validate_noise_level, validate_loan_parameters 
from .util import select_diagnostic_options, fetch_simulation_metadata 

__all__= [
    "simulate_landfill_capacity",
    "simulate_water_reserves", 
    "simulate_world_mineral_reserves", 
    "simulate_energy_consumption",
    "simulate_customer_churn",  
    "simulate_predictive_maintenance",
    "simulate_real_estate_price", 
    "simulate_sentiment_analysis", 
    "simulate_weather_forecasting", 
    "simulate_default_loan",
    "simulate_traffic_flow",
    "simulate_medical_diagnosis",
    "simulate_retail_sales"
    ]


def simulate_landfill_capacity(
    *, n_landfills=100, 
    start_date="2024-01-01", 
    end_date="2024-01-31",
    n_samples=None, 
    task="regression", 
    as_frame=False, 
    return_X_y=False,
    target_name=None, 
    noise_level=None, 
    seed=None
    ):
    """
    Generate a simulated dataset of landfill capacity measurements across
    various locations over a specified time period. This function creates
    synthetic data that mimics real-world landfill operations, incorporating
    factors like total capacity, waste accumulation, and environmental impact.
    It's designed to support machine learning applications in waste management
    and environmental sustainability, offering customizable options for
    regression or classification tasks.

    Parameters
    ----------
    n_landfills : int, optional
        The number of unique landfill sites for which data will be generated.
        More sites allow for a richer, more diverse dataset but increase
        computational requirements. Default value is set to 100, providing
        a balance between dataset complexity and generation speed.

    start_date : str, optional
        Specifies the start date for data generation in "YYYY-MM-DD" format.
        This parameter sets the temporal bounds of the dataset, enabling
        simulations over specific periods of interest. Default is "2024-01-01".

    end_date : str, optional
        Specifies the end date for data generation in "YYYY-MM-DD" format,
        inclusive. This allows for tailoring the dataset's time range to
        match specific study periods or scenarios. Default is "2024-01-31".

    n_samples : int or None, optional
        If specified, the function aims to generate a total number of samples
        close to this value by adjusting the `n_landfills` or data density.
        Useful for creating datasets with a predetermined size. Default is None,
        where the sample size is determined by `n_landfills` and the date range.

    task : str, optional
        Determines the nature of the target variable(s) for the dataset.
        - "regression": Targets are continuous values, suitable for regression.
        - "classification": Targets are categorical, suitable for classification.
        Default is "regression".

    as_frame : bool, optional
        If True, the dataset is returned as a pandas DataFrame, facilitating
        exploratory data analysis and integration with data science tools.
        Default is False, returning a Bunch object or ndarray tuples.

    return_X_y : bool, optional
        When True, returns the dataset as a tuple (X, y) of feature matrix and
        target vector, ready for use with machine learning algorithms.
        Ignored if `as_frame` is True. Default is False.

    target_name : str or None, optional
        Customizes the target variable's name. For regression, defaults to
        "capacity_usage_percent". For classification, defaults to "usage_category".
        Specifying a name directly selects or renames the target variable.

    noise_level : float or None, optional
        Adds Gaussian noise to the feature values to simulate measurement errors
        or natural variability. Defined by the standard deviation of the noise.
        Default is None, indicating no noise addition.

    seed : int or None, optional
        Seed for the random number generator to ensure reproducibility of
        the dataset across different runs. Default is None, resulting in
        different data each time the function is called.

    Returns
    -------
    Depending on the combination of `as_frame` and `return_X_y`, the function
    returns either a pandas DataFrame, a tuple of arrays (X, y), or a Bunch
    object containing the dataset and additional information.

    Notes
    -----
    The simulated data represents an idealized version of landfill operations
    and should not be used for precise engineering calculations without
    validation against real-world data. The synthetic nature of the dataset
    allows for flexible exploration of waste management scenarios but may
    not capture all complexities of actual landfill sites.

    Examples
    --------
    Generate a simple dataset for regression:
    >>> from gofast.datasets.simulate import simulate_landfill_capacity
    >>> data = simulate_landfill_capacity(n_samples=10 )
    >>> data.frame.head() 
        landfill_id       date  ...  soil_contamination_index  capacity_usage_percent
     0            1 2024-01-01  ...                  2.919168               99.969783
     1            1 2024-01-02  ...                  4.935421               37.903027
     2            1 2024-01-03  ...                  3.106227               75.606607
     3            2 2024-01-01  ...                  4.474257               71.741477
     4            2 2024-01-02  ...                  4.234219               62.955622
    
     [5 rows x 9 columns]

    Create a more complex dataset for classification, returned as a DataFrame:

    >>> df = simulate_landfill_capacity(task="classification", as_frame=True, 
    ...                                 n_landfills=50, start_date="2024-06-01",
    ...                                 end_date="2024-06-30")
    >>> df.head()
       landfill_id       date  ...  capacity_usage_percent  usage_category
    0            1 2024-06-01  ...               82.702655            High
    1            1 2024-06-02  ...               25.623501             Low
    2            1 2024-06-03  ...               83.977406            High
    3            1 2024-06-04  ...               88.319349            High
    4            1 2024-06-05  ...               97.274844            High
   
    [5 rows x 10 columns]
    Generate data for machine learning, with added noise:

    >>> X, y = simulate_landfill_capacity(return_X_y=True, task="regression", 
    ...                                   noise_level=0.05)
    >>> X.shape, y.shape 
    ((3100, 8), (3100,))
    """
    np.random.seed(seed)
    func_name = inspect.currentframe().f_code.co_name
    dataset_descr, features_descr= fetch_simulation_metadata (func_name)
    
    start_date, end_date = validate_dates(
        start_date, end_date, return_as_date_str= True )
    date_range = pd.date_range(start=start_date, end=end_date)
    n_days = len(date_range)
    
    if n_samples: 
        # Adjust n_landfills to fit the number of samples. 
        adjust_params= adjust_parameters_to_fit_samples(
            n_samples, initial_guesses= {'n_landfills': n_landfills, 
                                         "n_days":len(date_range)}
            )
        n_landfills = adjust_params.get("n_landfills", 10 )
        n_days= adjust_params.get("n_days", 7 )
        # now take the date from start_date to fit n_days. 
        date_range = date_range[: n_days]

    data = []

    for landfill_id in range(1, n_landfills + 1):
        for date in date_range:
            total_capacity = np.random.uniform(50000, 100000)  # in tons
            current_waste = np.random.uniform(10000, total_capacity)
            daily_inflow = np.random.uniform(50, 500)  # daily waste inflow in tons
            daily_outflow = np.random.uniform(0, 100)  # daily waste outflow in tons
            water_pollution_level = np.random.uniform(0, 10)  # Arbitrary pollution scale
            soil_contamination_index = np.random.uniform(0, 5)  # Arbitrary contamination scale
            
            capacity_usage = (current_waste / total_capacity) * 100  # as percentage

            record = {
                "landfill_id": landfill_id,
                "date": date,
                "total_capacity_tons": total_capacity,
                "current_waste_tons": current_waste,
                "daily_inflow_tons": daily_inflow,
                "daily_outflow_tons": daily_outflow,
                "water_pollution_level": water_pollution_level,
                "soil_contamination_index": soil_contamination_index,
                "capacity_usage_percent": capacity_usage,
            }
            data.append(record)
    
    landfill_df = pd.DataFrame(data)

    target_name = target_name or  "capacity_usage_percent"
    if noise_level is not None:
        # validate noise level 
        noise_level = validate_noise_level(noise_level) 
        numeric_cols = landfill_df.select_dtypes(include=np.number).columns
        landfill_df[numeric_cols] += np.random.normal(
            0, noise_level, landfill_df[numeric_cols].shape)

    # Adjust target based on the task
    if task == "classification":
        # Define categories for classification based on capacity usage
        bins = [0, 33, 66, 100]
        labels = ["Low", "Medium", "High"]
        landfill_df['usage_category'] = pd.cut(
            landfill_df['capacity_usage_percent'], bins=bins,
            labels=labels, include_lowest=True)
        if target_name == "capacity_usage_percent":
            target_name = "usage_category"

     # Select the default target if target_name is None
    return manage_data(
         data=landfill_df, as_frame=as_frame, return_X_y=return_X_y,
         target_names=target_name,
         DESCR=dataset_descr, features_descr= features_descr, 
         noise=noise_level, 
     )
    

def simulate_water_reserves(
    *, n_locations=100, 
    start_date="2024-01-01", 
    end_date="2024-01-31", 
    n_samples=None, 
    as_frame=False, 
    return_X_y=False, 
    target_name=None, 
    noise_level=None, 
    seed=None, 
  ):
    """
    Generate a simulated dataset of water reserve measurements across various
    locations over a specified time period. The dataset includes measurements
    such as total capacity, current volume, and water usage, along with 
    environmental factors affecting the reserves like rainfall and evaporation.

    Parameters
    ----------
    n_locations : int, optional
        Number of unique locations for which data will be generated. Each 
        location will have entries for each day within the specified date range,
        thus creating a comprehensive dataset across time and space. 
        Default is 100.
        
    start_date, end_date : str, optional
        The time range for the dataset, specified in "YYYY-MM-DD" format. 
        The function will generate data for each day within this range, 
        inclusive of both start and end dates. 
        Default range is from ``"2024-01-01"`` to ``"2024-01-31"``.
        
    n_samples : int or None, optional
        Specifies the target number of total samples to generate in the dataset. 
        This parameter allows for dynamic adjustment of the `n_locations` parameter 
        to match the desired number of samples, considering the number of days 
        between `start_date` and `end_date`. For instance, if `n_samples` is set 
        to 100 and the date range includes 31 days, the function will adjust 
        `n_locations` such that approximately 100 samples are generated across 
        all locations and days. This is particularly useful for scenarios where 
        a specific dataset size is required, whether for model training, testing, 
        or performance evaluation. If `None`, the function will generate data for 
        the number of locations specified by `n_locations` without adjustment, 
        which may result in a total sample size different from `n_samples`. 
        Default is `None`.
        
        It's important to note that the actual number of samples generated can 
        slightly vary from the specified `n_samples` due to rounding during the 
        adjustment process. The aim is to approximate the desired sample size 
        as closely as possible while maintaining a logical and realistic 
        distribution of data across locations and time.

    as_frame : bool, optional
        Determines the format of the returned dataset. If True, the dataset
        is returned as a pandas DataFrame, which is useful for data analysis 
        and manipulation with pandas. If ``False``, the dataset is returned in
        a Bunch object or as arrays, based on the `return_X_y` parameter.
        Default is ``False``.
        
    return_X_y : bool, optional
        If True, the function returns a tuple `(X, y)` where `X` is the array
        of feature values and `y` is the array of targets. This is useful 
        for directly feeding data into machine learning models. If ``False``, 
        returns a Bunch object containing the dataset, targets, and 
        additional information. This parameter is ignored if `as_frame` is 
        ``True``. Default is False.
        
    target_name : list of str, optional
        Specifies the names of the target variables to be included in the 
        output. By default (`None`), the function uses ``"percentage_full"`` as 
        the target variable, which represents the percentage of the water 
        reserve's capacity that is currently filled.
        
    noise : float or None, optional
        Adds Gaussian noise with standard deviation equal to `noise` to the 
        numerical features of the dataset to simulate measurement errors or 
        environmental fluctuations. If None, no noise is added. This can be 
        useful for creating more realistic datasets for robustness testing of 
        models. Default is ``None``.
        
    seed : int or None, optional
        Seed for the random number generator to ensure reproducibility of 
        the dataset. If ``None``, the random number generator is initialized
        without a fixed seed. Setting a seed is useful when conducting 
        experiments that require consistent results. Default is ``None``.

    **kws : dict
        Additional keyword arguments not explicitly listed above. This allows
        for future expansion or customization of the dataset generation without
        modifying the function signature.

    Returns
    -------
    Depending on the combination of `as_frame` and `return_X_y` parameters, 
    the function can return a pandas DataFrame, a tuple of `(X, y)` arrays, or 
    a Bunch object encapsulating the  dataset, targets, and additional 
    metadata.

    Notes
    -----
    The simulated dataset is generated using pseudo-random numbers to model 
    the dynamics of water reserves, including environmental impacts and human
    usage. It's intended for testing, educational purposes, and as a placeholder
    in the development of data analysis pipelines.

    Examples
    --------
    Generating a DataFrame of simulated water reserve data for 10 locations 
    over January 2024:
    
    >>> from gofast.datasets.simulate import simulate_water_reserves
    >>> data_obj= simulate_water_reserves(n_samples=10,)
    
    Generating `(X, y)` arrays suitable for use with scikit-learn models:
    
    >>> X, y = simulate_water_reserves(return_X_y=True)

    See Also
    --------
    simulate.simulate_world_mineral_reserves : 
        The primary data structure used to simulate world mineral reserves.
    sklearn.utils.Bunch : Used to package the dataset when arrays are returned.
    """
    from ._globals import WATER_RESERVES_LOC
    func_name = inspect.currentframe().f_code.co_name
    dataset_descr, features_descr= fetch_simulation_metadata (func_name)
    
    np.random.seed(seed)
    start_date, end_date = validate_dates(
        start_date, end_date, return_as_date_str= True )
    dates_origin = pd.date_range(start=start_date, end=end_date)
    
    if n_samples: 
        # Adjust n_locaions to fit the number of samples. 
        adjust_params= adjust_parameters_to_fit_samples(
            n_samples, initial_guesses= {'n_locations': n_locations, 
                                         "n_dates":len(dates_origin)}
            )
        n_locations = adjust_params.get("n_locations", 10 )
        n_dates= adjust_params.get("n_dates", 7 )
        # now take the date from start_date to fit n_dates. 
        dates = dates_origin[: n_dates]
    else: 
        dates = dates_origin 
        
    data = []
    # Generate a unique location name for each location_id before generating data
    location_names = {i+1: np.random.choice(WATER_RESERVES_LOC) for i in range(n_locations)}

    for i in range(n_locations):
        for date in dates:
            total_capacity_ml = np.random.uniform(5000, 10000)
            current_volume_ml = np.random.uniform(1000, total_capacity_ml)
            percentage_full = (current_volume_ml / total_capacity_ml) * 100
            rainfall_mm = np.random.uniform(0, 50)
            evaporation_mm = np.random.uniform(0, 10)
            inflow_ml = np.random.uniform(0, 500)
            outflow_ml = np.random.uniform(0, 500)
            usage_ml = np.random.uniform(0, 300)

            data.append({
                "location_id": i + 1,
                "location_name": location_names[i + 1],  
                "date": date,
                "total_capacity": total_capacity_ml,
                "current_volume": current_volume_ml,
                "rainfall": rainfall_mm,
                "evaporation": evaporation_mm,
                "inflow": inflow_ml,
                "outflow": outflow_ml,
                "usage": usage_ml,
                "percentage_full": percentage_full,
            })

    water_reserves_df = pd.DataFrame(data)
    if target_name is None:
        target_name = ["percentage_full"]

    return manage_data(
        data=water_reserves_df, as_frame=as_frame, return_X_y=return_X_y,
        target_names=target_name, noise=noise_level, seed=seed,
        features_descr=features_descr,
        DESCR=dataset_descr,
    )

def simulate_world_mineral_reserves(
    *, n_samples=100, 
    as_frame=False, 
    return_X_y=False, 
    target_name=None, 
    noise_level=None,
    seed=None, 
    regions=None, 
    distributions=None, 
    mineral_types=None, 
    countries=None, 
    economic_impact_factor=0.05, 
    default_location='Global HQ',
):
    """
    Simulates a dataset of world mineral reserves, providing insights into global 
    mineral production. This function allows for the generation of data reflecting 
    mineral reserve quantities across different countries and regions, incorporating
    economic impact factors and statistical noise to mimic real-world variability.

    The simulation process involves selecting countries known for producing specified
    minerals, calculating reserve quantities, and optionally assigning a default location 
    for minerals with undetermined origins. The function offers flexibility in focusing 
    the simulation on specific minerals, regions, or countries, catering to diverse 
    geoscientific research needs.

    Parameters
    ----------
    n_samples : int, optional
        Number of data points to generate, representing individual mineral reserve 
        instances. Default is 300.
    as_frame : bool, optional
        If set to True, outputs the dataset as a pandas DataFrame, facilitating 
        data analysis and visualization. Default is False.
    return_X_y : bool, optional
        If True, separates the features and target variable into two entities, 
        supporting machine learning applications. Default is False.
    target_name : str or list of str, optional
        Specifies the names for the target variables, enhancing dataset 
        interpretability. Default is None.
    noise_level : float, optional
        Specifies the standard deviation of Gaussian noise added to reserve quantities,
        simulating measurement errors or estimation uncertainties. Default is None.
    seed : int, optional
        Seed for the random number generator, ensuring reproducibility of the simulated
        dataset. Default is None.
    regions : list of str, optional
        Filters the simulation to include countries within specified geographical regions,
        reflecting regional mineral production characteristics. Default is None.
    distributions : dict, optional
        Custom mapping of regions to mineral types for targeted simulation scenarios.
        Default is None.
    mineral_types : list of str, optional
        Filters the dataset to simulate reserves for specified minerals, aiding in 
        focused geoscientific studies. Default is None.
    countries : list of str, optional
        Specifies a list of countries to be included in the simulation, enabling 
        country-specific mineral production analysis. Default is None.
    economic_impact_factor : float, optional
        Adjusts the simulated quantity of reserves based on economic conditions, 
        introducing variability in mineral production estimates. Default is 0.05.
    default_location : str, optional
        Placeholder for the location when a mineral's producing country is 
        undetermined, typically a headquarters or primary research center. 
        Default is 'Global HQ'.

    Returns
    -------
    pandas.DataFrame or tuple
        The simulated dataset of mineral reserves. The structure of the returned
        data depends on the `as_frame` and `return_X_y` parameters.

    Raises
    ------
    ValueError
        Raised if required parameters are not provided or if the simulation 
        process encounters an error due to invalid parameter values.

    Examples
    --------
    Generate a simple dataset of mineral reserves:
    
    >>> from gofast.datasets.simulate import simulate_world_mineral_reserves
    >>> min_reserves = simulate_world_mineral_reserves()
    
    Simulate reserves focusing on gold and diamonds in Africa and Asia:
    
    >>> df=simulate_world_mineral_reserves(regions=['Africa', 'Asia'],
    ...                                 mineral_types=['gold', 'diamond'],
    ...                                 n_samples=100, as_frame=True)
    >>> df.head()
        sample_id  ...      quantity
     0          1  ...    317.714995
     1          2  ...  10009.865385
     2          3  ...   6234.385021
     3          4  ...   1406.044839
     4          5  ...   7581.913884
    
     [5 rows x 6 columns]
    Handle undetermined production countries with a custom default location:
    >>> X, y= simulate_world_mineral_reserves(default_location='Research Center',
    ...                                 noise=0.1, seed=42, return_X_y=True )
    >>> len(y)
    100
    """
    func_name = inspect.currentframe().f_code.co_name
    dataset_descr, features_descr= fetch_simulation_metadata (func_name)
    
    # Initialize country dict 
    mineral_countries_map ={}
    # Set a seed for reproducibility of results
    np.random.seed(seed)
    # Normalize region input and validate regions if provided
    if regions:
        regions=is_iterable(regions, exclude_string=True, transform =True)
        regions = [validate_region(region, mode='soft') for region in regions] 

    # Normalize mineral_types to a list and find corresponding countries if specified
    if mineral_types:
        mineral_types= is_iterable(
            mineral_types, exclude_string=True, transform=True)
        countries, mineral_countries_map = find_countries_by_minerals(
            mineral_types, return_minerals_and_countries= True
        )
    # Extract minerals from regions if mineral_types are not explicitly provided
    if mineral_types is None and regions:  
        mineral_types = extract_minerals_from_regions(regions) 
    
    # Create distributions mapping if both regions and mineral types are available
    if regions and mineral_types: 
        distributions = build_distributions_from(regions, mineral_types)
        
    # Validate the structure of distributions, falling back to default if incorrect
    if distributions:
        if not check_distributions(distributions):
            # If distributions are invalid, revert to None and possibly warn the user
            distributions = None 

    # Use default distributions if none provided or after fallback
    if distributions is None:
        regions = ["Africa", "Asia", "Europe", "America", "Oceania"]
        distributions = unpack_list_of_dicts(
            [find_mineral_distributions(region) for region in regions])
        mineral_types = extract_minerals_from_regions(regions) 
    # Find countries based on mineral types if countries are not explicitly provided
    if not countries:
        countries, mineral_countries_map = find_countries_by_minerals(
            mineral_types, return_minerals_and_countries= True 
        )
    # Generate ore informations related to each country
    infos_dict = generate_ore_infos(countries, error="ignore")
    
    # validate noise level 
    noise_level = validate_noise_level(
        noise_level) if noise_level else noise_level 
   
    # Simulate mineral reserve data for each sample
    data = []
    for i in range(n_samples):
        selected_region = np.random.choice(list(distributions.keys()))
        available_minerals = distributions[selected_region]
        mineral_type = np.random.choice(available_minerals)
        base_quantity = np.random.uniform(100, 10000)
        economic_impact = 1 + (np.random.rand() * economic_impact_factor - (
            economic_impact_factor / 2)) # make sure to not have negative quantities
        quantity = max(0, base_quantity * economic_impact + (
            np.random.normal(0, noise_level) if noise_level else 0))
        # Select location 
        selected_region, location= select_location_for_mineral (
            mineral_type, mineral_countries_map,
            fallback_countries=countries, 
            selected_region =selected_region, 
            substitute_for_missing= default_location 
        )
        info = infos_dict.get(location, "No information available")
        reserve_details = build_reserve_details_by_country (location)

        data.append({
            'sample_id': i + 1,
            'region': selected_region,
            'location': location,
            'mineral_type': mineral_type,
            'info': info,
            'quantity': quantity, 
            **reserve_details
        }
        )
    # Convert simulated data into a DataFrame
    mineral_reserves_df = pd.DataFrame(data)
    # Handle data return format based on function parameters
    return manage_data(
        data=mineral_reserves_df, as_frame=as_frame, return_X_y=return_X_y,
        target_names=target_name if target_name else ['quantity'],
        features_descr= features_descr,
        DESCR=dataset_descr, seed=seed,
    )

def simulate_energy_consumption(
    *, n_households=10, 
    days=365, 
    start_date="2021-01-01", 
    n_samples=None, 
    as_frame=False, 
    return_X_y=False, 
    target_name=None, 
    noise_level=None, 
    seed=None, 
    **kwargs
    ):
    """
    Generates a simulated dataset representing energy consumption across multiple
    households over a specified time frame. This simulation integrates several 
    factors, including household size, the presence of energy-saving appliances, 
    electric vehicles, solar panels, and the influence of daily temperature 
    fluctuations, to model realistic energy usage patterns.

    Parameters
    ----------
    n_households : int, default=10
        The total number of households for which the energy consumption
        is to be simulated.
    days : int, default=365
        The span of days across which the energy consumption is simulated,
        starting from `start_date`.
    start_date : str, default='2021-01-01'
        The commencement date of the simulation period, formatted as 'YYYY-MM-DD'.
    as_frame : bool, default=False
        Determines if the output dataset is provided as a pandas DataFrame.
        Useful for data analysis and manipulation within pandas.
    return_X_y : bool, default=False
        If set to True, the function returns the predictors (`X`) and the target 
        variable (`y`) as separate entities, facilitating ease of use in 
        machine learning models.
    target_name : str or None, default=None
        Specifies the column name of the target variable within the dataset. If 
        left as None, the default target variable 'energy_consumption_kwh' is used.
    noise_level : float or None, optional
        The standard deviation of Gaussian noise added to the energy consumption
        figures to simulate real-world unpredictability and measurement errors.
    seed : int or None, optional
        A seed value to initialize the random number generator, ensuring the
        reproducibility of the dataset across multiple function calls.

    Returns
    -------
    Depending on the combination of `as_frame` and `return_X_y` flags, the 
    function can return:
        - A Bunch object (similar to scikit-learn datasets) containing the data 
          and metadata.
        - A pandas DataFrame if `as_frame` is True.
        - A tuple containing the features matrix `X` and the target vector `y` if 
          `return_X_y` is True.

    Examples
    --------
    >>> from gofast.datasets.simulate import simulate_energy_consumption
    >>> dataset = simulate_energy_consumption(n_households=100, days=30, as_frame=True)
    >>> print(dataset.head())

    Notes
    -----
    This dataset is primarily intended for use in predictive modeling tasks, such 
    as forecasting household energy consumption based on historical data and 
    various influencing factors. It provides a rich set of features that mirror 
    the complexities of real-world energy usage patterns, making it suitable for 
    both regression and classification tasks in machine learning.
    """
    func_name = inspect.currentframe().f_code.co_name
    dataset_descr, features_descr= fetch_simulation_metadata (func_name)
    
    if n_samples:
        n_samples = validate_positive_integer(n_samples, "n_samples")
        # recompute the n_households and days to fit the 
        # number of samples. 
        n_households, days = adjust_households_and_days( n_samples)
       
    # Validate input parameters
    n_households = validate_positive_integer(n_households, "n_households")
    days = validate_positive_integer(days, "days")
    
    start_date, _= validate_dates(start_date =start_date ,
                   end_date= get_last_day_of_current_month(return_today= True), 
                   return_as_date_str= True
                   )
    np.random.seed(seed) # For reproducibility

    # electric vehicles in households
    electric_vehicles = np.random.choice([0, 1], size=n_households, p=[0.8, 0.2])
    
    # Additional feature: households with solar panels
    solar_panels = np.random.choice([0, 1], size=n_households, p=[0.75, 0.25])
    
    dates = pd.date_range(start=start_date, periods=days, freq='D')
    time_features = np.tile(dates, n_households)
    
    household_ids = np.repeat(np.arange(1, n_households + 1), len(dates))
    household_sizes = np.random.choice(
        [1, 2, 3, 4, 5], size=n_households, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    energy_saving_appliances = np.random.choice(
        [0, 1], size=n_households, p=[0.7, 0.3])
    
    avg_temperature = np.sin(np.linspace(0, 2 * np.pi, days)) * 10 + 15
    temperatures = np.repeat(avg_temperature, n_households)
    
    base_consumption = household_sizes[household_ids - 1] * 5
    temperature_effect = (temperatures - 15) / 5
    energy_saving_effect = energy_saving_appliances[household_ids - 1] * -1
    # Increase consumption for EV households
    ev_effect = electric_vehicles[household_ids - 1] * 2  
    # Decrease consumption for solar panel households
    solar_panel_effect = solar_panels[household_ids - 1] * -2  
    
    energy_consumption = ( 
        base_consumption 
        + temperature_effect 
        + energy_saving_effect 
        + ev_effect 
        + solar_panel_effect
        )
    energy_consumption = np.abs(energy_consumption)
    # Construct the DataFrame
    energy_data = pd.DataFrame({
        'date': time_features,
        'household_id': household_ids,
        'household_size': household_sizes[household_ids - 1],
        'energy_saving_appliances': energy_saving_appliances[household_ids - 1],
        'electric_vehicles': electric_vehicles[household_ids - 1],
        'solar_panels': solar_panels[household_ids - 1],
        'temperature': temperatures,
        'energy_consumption_kwh': energy_consumption
    })

    # Select the default target if target_name is None
    return manage_data(
         data=energy_data, as_frame=as_frame, return_X_y=return_X_y,
         target_names=target_name if target_name else ['energy_consumption_kwh'],
         DESCR=dataset_descr, features_descr= features_descr, 
         noise=noise_level, 
     )


def simulate_customer_churn(
    *, n_customers=1000, 
    as_frame=False, 
    return_X_y=False, 
    target_name=None, 
    noise_level=None, 
    seed=None
    ):
    """
    Simulates a dataset for customer churn prediction based on demographic
    information, service usage patterns, and other relevant customer features. 
    
    This simulation aims to provide a realistic set of data for modeling customer 
    churn in various services or subscription-based business models.

    Parameters
    ----------
    n_customers : int, default=1000
        The number of customer records to generate in the dataset.
    as_frame : bool, default=False
        If True, the function returns the dataset as a pandas DataFrame,
        facilitating data manipulation and analysis.
    return_X_y : bool, default=False
        If True, the dataset is split into features (`X`) and the target variable 
        (`y`), returned as separate objects. This is useful for direct use in 
        machine learning algorithms.
    target_name : str or None, default=None
        The name of the target variable in the dataset. For churn prediction, 
        it is typically a binary variable indicating whether a customer has churned. 
        If None, defaults to 'churn'.
    noise_level : float or None, optional
        Specifies the standard deviation of Gaussian noise to be added to the 
        churn variable, simulating prediction uncertainty and data variability.
    seed : int or None, optional
        A seed for the random number generator, ensuring reproducibility of the 
        dataset across different function calls.

    Returns
    -------
    Depending on the specified parameters, this function can return:
        - A pandas DataFrame if `as_frame` is set to True.
        - Two arrays, features (`X`) and target (`y`), if `return_X_y` is True.
        - A dictionary-like object with data arrays, feature names, and target if 
          none of the above flags are set.

    Examples
    --------
    >>> from customer_churn_simulation import simulate_customer_churn
    >>> dataset = simulate_customer_churn(n_customers=500, as_frame=True)
    >>> print(dataset.head())

    Notes
    -----
    The simulated dataset can be used for binary classification tasks, especially 
    for building and validating customer churn prediction models. The features 
    include both categorical and continuous variables, reflecting a wide range 
    of factors that could influence a customer's decision to leave a service.
    """
    func_name = inspect.currentframe().f_code.co_name
    dataset_descr, features_descr= fetch_simulation_metadata (func_name)
    
    np.random.seed(seed)

    n_customers= validate_positive_integer(n_customers, "n_customers")
    # Customer demographic features
    ages = np.random.randint(18, 70, size=n_customers)
    tenure_months = np.random.randint(1, 72, size=n_customers)
    monthly_charges = np.random.uniform(10, 120, size=n_customers)
    total_charges = monthly_charges * tenure_months
    gender = np.random.choice(['Male', 'Female'], size=n_customers)
    senior_citizen = np.random.choice([0, 1], size=n_customers, p=[0.85, 0.15])
    partner = np.random.choice([0, 1], size=n_customers, p=[0.5, 0.5])
    dependents = np.random.choice([0, 1], size=n_customers, p=[0.7, 0.3])

    # Customer service features
    multiple_lines = np.random.choice([0, 1], size=n_customers, p=[0.6, 0.4])
    online_security = np.random.choice([0, 1], size=n_customers, p=[0.5, 0.5])
    online_backup = np.random.choice([0, 1], size=n_customers, p=[0.45, 0.55])
    device_protection = np.random.choice([0, 1], size=n_customers, p=[0.48, 0.52])
    tech_support = np.random.choice([0, 1], size=n_customers, p=[0.49, 0.51])
    streaming_tv = np.random.choice([0, 1], size=n_customers, p=[0.4, 0.6])
    streaming_movies = np.random.choice([0, 1], size=n_customers, p=[0.42, 0.58])

    # Target variable: Churn
    churn = np.random.choice([0, 1], size=n_customers, p=[0.73, 0.27])

    # Constructing DataFrame
    data = pd.DataFrame({
        'age': ages,
        'tenure_months': tenure_months,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'gender': gender,
        'senior_citizen': senior_citizen,
        'partner': partner,
        'dependents': dependents,
        'multiple_lines': multiple_lines,
        'online_security': online_security,
        'online_backup': online_backup,
        'device_protection': device_protection,
        'tech_support': tech_support,
        'streaming_tv': streaming_tv,
        'streaming_movies': streaming_movies,
        'churn': churn
    })

    # Apply noise if specified
    if noise_level:
        # validate noise level 
        noise_level = validate_noise_level(
            noise_level, .01 ) if noise_level else noise_level 
        noise_amount = int(noise_level * n_customers)
        noise_indices = np.random.choice(data.index, size=noise_amount, replace=False)
        data.loc[noise_indices, 'churn'] = 1 - data.loc[noise_indices, 'churn']

    return manage_data(
         data=data, as_frame=as_frame, return_X_y=return_X_y,
         target_names=target_name if target_name else ['churn'],
         DESCR=dataset_descr, features_descr=features_descr  
     )

def simulate_predictive_maintenance(
    *, 
    n_machines=25, 
    n_sensors=5, 
    operational_params=2, 
    days=30,
    start_date="2021-01-01", 
    failure_rate=0.02, 
    maintenance_frequency=45,
    task="classification",
    n_samples=None, 
    as_frame=False, 
    return_X_y=False,
    target_name=None, 
    noise_level=None, 
    seed=None
    ):
    """
    Generates a synthetic dataset tailored for predictive maintenance tasks, 
    offering detailed insights into the operational dynamics and maintenance 
    requirements of a fleet of machines over a specified period. 
    
    The dataset incorporates various attributes such as sensor readings, 
    operational parameters, maintenance schedules, and failure events, making
    it conducive for developing and evaluating machine learning models aimed 
    at either classifying potential machine failures or regressing on the time
    until the next maintenance event.

    Parameters
    ----------
    n_machines : int, default=25
        Specifies the number of individual machines included in the simulation,
        each uniquely identified and monitored over the simulation period.

    n_sensors : int, default=5
        Denotes the quantity of distinct sensors installed per machine, each
        generating continuous operational data reflective of the machine's state.

    operational_params : int, default=5
        Represents the count of operational parameters that are critical to
        assessing the performance and efficiency of the machines.

    days : int, default=30
        Defines the total number of days across which the simulation spans,
        creating a longitudinal dataset that captures seasonal variations and
        operational trends.

    start_date : str, default='2021-01-01'
        Marks the commencement date of the dataset, providing a temporal anchor
        from which all simulated events are chronologically generated.

    failure_rate : float, default=0.02
        The estimated daily probability of a failure occurrence per machine,
        influencing the generation of failure events within the simulation.

    maintenance_frequency : int, default=90
        Specifies the interval, in days, between scheduled maintenance activities,
        introducing periodic resets in the simulation of machine health and failure
        probabilities.

    task : str, default='classification'
        Determines the primary objective of the simulated dataset, with options
        including 'classification' for binary outcomes of machine failure, and
        'regression' for continuous outcomes depicting the time until an impending
        maintenance requirement.

    n_samples : int, optional
        If specified, dynamically adjusts the `n_machines`, `days`, `n_sensors`,
        and `operational_params` to collectively satisfy a predefined number of
        samples, enabling fine-tuning of the dataset's scale and granularity.

    as_frame : bool, default=False
        When set to True, the generated dataset is presented as a pandas DataFrame,
        facilitating direct manipulations and exploratory data analyses within
        pandas' ecosystem.

    return_X_y : bool, default=False
        If True, decomposes the dataset into separate entities, with the predictors
        (`X`) and the target variable (`y`) returned as distinct components, streamlining
        their application in machine learning pipelines.

    target_name : str, optional
        Customizes the label of the target variable within the dataset, defaulting
        to 'failure' for classification scenarios and 'days_until_maintenance' for
        regression contexts.

    noise_level : float, optional
        Introduces a layer of Gaussian noise to the sensor and operational data,
        mimicking real-world inaccuracies and measurement discrepancies.

    seed : int, optional
        Seeds the random number generator to ensure the reproducibility of the dataset
        across different simulation runs, preserving the consistency of results.

    Returns
    -------
    Depending on the combination of `as_frame` and `return_X_y`, this function
    can return a Bunch object, a pandas DataFrame, or a tuple containing the
    features matrix `X` and the target vector `y`.

    Examples
    --------
    >>> from gofast.datasets.simulate import simulate_predictive_maintenance
    >>> dataset = simulate_predictive_maintenance(
    ... n_machines=100, task='regression', as_frame=True)
    >>> print(dataset.head())
        sensor_1  sensor_2  ...  machine_id  days_until_maintenance
     0  0.068893 -2.536880  ...           1                    30.0
     1  0.988240 -0.304479  ...           1                    30.0
     2  0.534793  0.067235  ...           1                    29.0
     3  0.074354 -0.807021  ...           1                    29.0
     4 -1.840403 -0.106911  ...           1                    28.0
    
    Notes
    -----
    The simulated dataset offers a rich source of data for building and testing
    predictive maintenance algorithms. It captures various factors that influence
    machine health and maintenance needs, providing a realistic foundation for
    developing models that can predict maintenance events and prevent machine failures.
    """
    func_name = inspect.currentframe().f_code.co_name
    dataset_descr, features_descr= fetch_simulation_metadata (func_name)
    
    if n_samples: 
        n_samples= validate_positive_integer(n_samples, "n_samples")
        # recompute params to fit the number of samples.
        initial_guesses = {
            'n_machines': n_machines,
            'n_sensors': n_sensors,
            'operational_params': operational_params,
            'days': days
        }
        params = adjust_parameters_to_fit_samples(n_samples, initial_guesses)
        n_machines = params.get("n_machines", 6 )
        n_sensors = params.get("n_sensors", 2) 
        operational_params= params.get("operational_params", 1)
        days= params.get("days", 11)

    # Validate input parameters
    n_machines = validate_positive_integer(n_machines, "n_machines")
    days = validate_positive_integer(days, "days")
    n_sensors = validate_positive_integer(n_sensors, "n_sensors")
    operational_params = validate_positive_integer(days, "operational_params")
    maintenance_frequency = validate_positive_integer(maintenance_frequency,
                                                      "maintenance_frequency")
    np.random.seed(seed)
    
    # Generate dates
    start_date, _= validate_dates(start_date =start_date ,
                   end_date= get_last_day_of_current_month(return_today= True), 
                   return_as_date_str= True
                   )
    dates = pd.date_range(start=start_date, periods=days, freq='D')
    machine_ids = np.arange(1, n_machines + 1)
    
    sensor_data = np.random.randn(days * n_machines, n_sensors)
    operational_data = np.random.randn(days * n_machines, operational_params)
    
    # Simulate failures and maintenance events
    failures = np.random.binomial(1, failure_rate, (days, n_machines))
    maintenance_dates = np.arange(maintenance_frequency, days, maintenance_frequency)
    maintenance = np.isin(
        np.tile(np.arange(days), (n_machines, 1)).T, maintenance_dates).astype(int)
    
    # Combine data
    data = np.hstack((sensor_data, operational_data))
    if task == "classification":
        # Label 1 on failure dates, considering maintenance resets
        # failure probability
        target = np.cumsum(failures, axis=0)
        target[maintenance == 1] = 0
        target = np.where(target > 0, 1, 0).flatten()
    else:  # Regression task to predict days until next maintenance
        days_until_maintenance = np.zeros((days, n_machines))
        # Iterate over each machine
        for machine_idx in range(n_machines):
            # Calculate maintenance days for this machine
            machine_maintenance_days = np.arange(
                maintenance_frequency - 1, days,maintenance_frequency)
            for day_idx in range(days):
                # Find the next maintenance day for this machine
                next_maintenance = machine_maintenance_days[machine_maintenance_days >= day_idx]
                if len(next_maintenance) > 0:
                    days_until_next_maintenance = next_maintenance[0] - day_idx
                else:
                    # If no next maintenance day is found, set a default value 
                    # (e.g., days till end of period)
                    days_until_next_maintenance = days - day_idx
                # Assign the days until next maintenance to the matrix
                days_until_maintenance[day_idx, machine_idx] = days_until_next_maintenance
        
        # Flatten the matrix to match the DataFrame's expected target column length
        target = days_until_maintenance.flatten()
    
    # Add noise if specified
    if noise_level:
        # validate noise level 
        noise_level = validate_noise_level(
            noise_level, 1. ) if noise_level else noise_level 
        data += np.random.normal(0, noise_level, data.shape)
    
    columns = [f'sensor_{i}' for i in range(1, n_sensors + 1)] + \
              [f'op_param_{i}' for i in range(1, operational_params + 1)]
    df = pd.DataFrame(data, columns=columns)
    df['date'] = np.tile(dates, n_machines)
    df['machine_id'] = np.repeat(machine_ids, days)
    
    if not target_name:
        target_name = ( 
            'failure' if task == "classification" 
            else 'days_until_maintenance'
       )
    df[target_name] = target
    
    # Return data
    return manage_data(
        data=df, as_frame=as_frame, return_X_y=return_X_y,
        target_names=target_name if target_name else None,
        seed=seed,
        DESCR=dataset_descr, features_descr=features_descr  
     )

def simulate_real_estate_price(
    *, n_properties=1000, 
    features=None,
    economic_indicators=None,
    start_year=2000,
    years=20,
    price_increase_rate=0.03,
    n_samples=None, 
    as_frame=False, 
    return_X_y=False, 
    target_name=None, 
    noise_level=None, 
    seed=None
    ):
    """
    Simulates a synthetic dataset for real estate price prediction, incorporating 
    various property features, economic indicators, and temporal dynamics to 
    generate realistic market value estimations of residential properties over a 
    specified time frame. This dataset is specifically designed to support the 
    development and testing of machine learning models aimed at regression tasks 
    in the real estate domain, such as predicting property prices based on their 
    attributes and broader economic conditions.

    Parameters
    ----------
    n_properties : int, default=1000
        The number of residential properties to include in the simulation, each 
        with its unique characteristics and price history over the specified years.

    features : list of str, optional
        A list specifying the characteristics of properties to simulate, such as 
        size, number of bedrooms, and age. Defaults to common features like size 
        in square meters, bedrooms, and bathrooms.

    economic_indicators : list of str, optional
        A list of economic indicators to simulate alongside property features, 
        reflecting broader economic conditions that might impact real estate 
        prices, like interest rates and GDP growth rate.

    start_year : int, default=2000
        The starting year for the simulated price data, providing a temporal 
        foundation for the dataset's timeline.

    years : int, default=20
        The duration in years over which the real estate price data is simulated, 
        allowing for long-term price trend analysis.

    price_increase_rate : float, default=0.03
        An annual rate at which property prices are simulated to increase, 
        representing general market appreciation or inflation.

    n_samples : int, optional
        If specified, dynamically adjusts the `n_properties` and `years` to 
        achieve a target number of samples, offering flexibility in dataset size.

    as_frame : bool, default=False
        If True, returns the dataset as a pandas DataFrame, enabling direct use 
        within pandas for data exploration and manipulation.

    return_X_y : bool, default=False
        If set to True, splits the dataset into predictors (`X`) and the target 
        variable (`y`), simplifying its integration into machine learning workflows.

    target_name : str or None, default=None
        Customizes the name of the target variable column, which by default is 
        set to 'price', representing the market value of properties.

    noise_level : float or None, optional
        Specifies the standard deviation of Gaussian noise to add to the property 
        prices, simulating measurement errors or market volatility.

    seed : int or None, optional
        Sets the seed for the random number generator to ensure the reproducibility 
        of the dataset across simulations.

    Returns
    -------
    The output format of the simulated dataset depends on the `as_frame` and 
    `return_X_y` parameters, providing versatility in how the data is returned. 
    Options include a structured Bunch object, a pandas DataFrame, or a tuple of 
    arrays (`X`, `y`) for the predictors and target variable.

    Examples
    --------
    >>> from gofast.datasets.simulate import simulate_real_estate_price
    >>> dataset = simulate_real_estate_price(n_properties=500, years=10, as_frame=True)
    >>> print(dataset.head())

    Notes
    -----
    This dataset offers a rich framework for exploring the dynamics of real estate 
    markets through a machine learning lens, encompassing a wide range of features 
    and economic indicators that influence property values. It is particularly 
    valuable for regression analyses aiming to predict real estate prices based on 
    quantitative attributes and temporal trends.
    """
    func_name = inspect.currentframe().f_code.co_name
    dataset_descr, features_descr= fetch_simulation_metadata (func_name)
    
    # Default features if None specified
    if features is None:
        features = ['size_sqm', 'bedrooms', 'bathrooms',
                    'age_years', 'distance_city_center_km']
    if economic_indicators is None:
        economic_indicators = ['interest_rate', 'unemployment_rate',
                               'GDP_growth_rate']
    
    features = is_iterable(features, exclude_string= True, transform =True)
    economic_indicators= is_iterable(
        economic_indicators, exclude_string=True, transform=True )
    # Adjust properties and years to fit n_samples
    if n_samples:
        n_samples = validate_positive_integer(years, "n_samples")
        initial_guesses = {'n_properties': n_properties, 'years': years}
        adjusted_values = adjust_parameters_to_fit_samples(n_samples, initial_guesses)
        n_properties = adjusted_values.get('n_properties', n_properties)
        years = adjusted_values.get('years', years)
    
    start_year, _ = validate_dates(start_year, end_date= get_last_day_of_current_month())
    years = validate_positive_integer(years, "years")
    n_properties = validate_positive_integer(n_properties, "n_properties")
    
    np.random.seed(seed)
    
    # Generate synthetic feature data
    feature_data = np.random.rand(n_properties * years, len(features))
    economic_data = np.random.rand(n_properties * years, len(economic_indicators))
    
    # Generate years
    year_data = np.repeat(np.arange(start_year, start_year + years), n_properties)
    
    # Generate price data with a yearly increase rate
    base_prices = np.random.randint(50000, 500000, n_properties)
    price_data = np.tile(base_prices, (years, 1)).T * (
        1 + price_increase_rate) ** np.arange(years)
    price_data = price_data.flatten()
    
    # Add noise to price data if specified
    if noise_level:
        # validate noise level 
        noise_level = validate_noise_level(
            noise_level) if noise_level else noise_level 
        price_data += np.random.normal(0, noise_level, price_data.shape)
    
    # Combine all data
    data = np.hstack((feature_data, economic_data, year_data.reshape(-1, 1),
                      price_data.reshape(-1, 1)))
    
    columns = features + economic_indicators + ['year', 'price']
    df = pd.DataFrame(data, columns=columns)

    # Handle return types
    return manage_data(
        data=df, as_frame=as_frame, return_X_y=return_X_y,
        target_names=target_name if target_name else 'price',
        seed=seed,
        DESCR=dataset_descr, features_descr=features_descr  
    )

def simulate_sentiment_analysis(
    *, n_reviews=1000, 
    review_length_range=(50, 300), 
    sentiment_distribution='auto', 
    n_samples=None, 
    as_frame=False, 
    return_X_y=False, 
    target_name=None, 
    noise_level=None, 
    seed=None
):
    """
    Simulates a dataset for sentiment analysis tasks, focusing on classifying 
    the sentiment of product reviews into positive, neutral, or negative 
    categories. This dataset is generated using text data processing techniques 
    to create realistic review texts and corresponding sentiment labels, making 
    it suitable for training and evaluating machine learning models specialized 
    in natural language understanding and sentiment classification.

    Parameters
    ----------
    n_reviews : int, default=1000
        The number of product reviews to generate in the simulated dataset.
    review_length_range : tuple of int, default=(50, 300)
        A tuple specifying the minimum and maximum length of the reviews, measured 
        in number of words.
    sentiment_distribution : tuple of float, default=(0.4, 0.2, 0.4)
        A tuple representing the distribution of positive, neutral, and negative 
        sentiments among the reviews, where each value must sum to 1.
    n_samples : int, optional
        If specified, adjusts `n_reviews` to meet the desired number of samples, 
        affecting the total number of reviews generated.
    as_frame : bool, default=False
        If True, returns the dataset as a pandas DataFrame, with reviews as text 
        entries and sentiment labels as categorical values.
    return_X_y : bool, default=False
        If set to True, separates the dataset into predictors (`X`) comprising the 
        review texts and the target variable (`y`) containing sentiment labels.
    target_name : str or None, default=None
        The name of the target variable column, defaulting to 'sentiment' if not 
        specified. This column contains the sentiment classification for each review.
    seed : int or None, optional
        Initializes the random number generator to ensure the reproducibility of 
        the dataset across different simulations.

    Returns
    -------
    Depending on the combination of `as_frame` and `return_X_y`, the function 
    can return a pandas DataFrame, a tuple of (`X`, `y`), or a Bunch object 
    containing the dataset and descriptive metadata.

    Examples
    --------
    >>> from gofast.datasets.simulate import simulate_sentiment_analysis
    >>> dataset = simulate_sentiment_analysis(n_reviews=500, as_frame=True)
    >>> print(dataset.head())
                                              review sentiment
    0  amet amet sit ipsum lorem lorem ipsum adipisci...  negative
    1  lorem consectetur sit adipiscing ipsum adipisc...  positive
    2  amet ipsum amet consectetur elit sit lorem lor...  negative
    3  elit ipsum adipiscing elit consectetur amet am...  negative
    4  adipiscing elit ipsum lorem ipsum consectetur ...  positive
    
    Notes
    -----
    This simulated dataset provides a foundation for exploring and refining 
    sentiment analysis techniques within the domain of natural language processing. 
    It covers a spectrum of sentiments in customer product reviews, offering a 
    diverse range of text data for model training and performance evaluation.
    """
    func_name = inspect.currentframe().f_code.co_name
    dataset_descr, features_descr= fetch_simulation_metadata (func_name)
    
    def generate_random_text(length, seed=None):
        np.random.seed(seed)
        words = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", 
                 "adipiscing", "elit"]
        return ' '.join(np.random.choice(words, length))
    
    np.random.seed(seed)

    if n_samples:
        # Adjust n_reviews based on n_samples 
        n_samples = validate_positive_integer(n_samples, "n_samples")
        initial_guesses = {'n_reviews': n_reviews}
        adjusted_values = adjust_parameters_to_fit_samples(n_samples, initial_guesses)
        n_reviews = adjusted_values.get('n_reviews', 700)
    
    sentiment_distribution = validate_distribution(
        sentiment_distribution, elements = [ 'positive', 'neutral', 'negative']
        )
    review_length_range = validate_length_range(review_length_range)
    
    review_lengths = np.random.randint(*review_length_range, n_reviews)
    sentiments = np.random.choice(["positive", "neutral", "negative"],
                                  p=sentiment_distribution, size=n_reviews)
    
    reviews = [generate_random_text(length, seed) for length in review_lengths]

    if noise_level:
        # validate noise level 
        noise_level = validate_noise_level(
            noise_level) if noise_level else noise_level 
        noisy_indices = np.random.choice(
            range(n_reviews), size=int(noise_level * n_reviews), replace=False)
        for idx in noisy_indices:
            sentiments[idx] = np.random.choice(["positive", "neutral", "negative"])
    
    data = pd.DataFrame({"review": reviews, "sentiment": sentiments})

    return manage_data(
        data=data, as_frame=as_frame, return_X_y=return_X_y,
        target_names=target_name if target_name else "sentiment",
        seed=seed,
        DESCR=dataset_descr, features_descr=features_descr  
    )

def simulate_weather_forecasting(
    *, n_days=365, 
    weather_variables=None,
    start_date="2020-01-01", 
    n_samples=None, 
    as_frame=False, 
    return_X_y=False, 
    target_name=None, 
    noise_level=0.05, 
    seed=None,
    include_extreme_events=False
    ):
    """
    Generates a synthetic dataset tailored for weather forecasting tasks, simulating
    atmospheric conditions and weather variables over a defined period. This dataset
    can serve as a foundation for developing and testing machine learning models focused
    on predicting weather conditions such as temperature, humidity, and precipitation.

    Parameters
    ----------
    n_days : int, default=365
        The total number of days for which weather data is simulated, providing daily
        observations of the specified weather variables.
    weather_variables : list of str, optional
        A list of weather variables to include in the simulation, such as 'temperature',
        'humidity', and 'precipitation'. If None, a default set of variables is used.
    start_date : str, default="2020-01-01"
        The starting date of the weather data simulation, formatted as 'YYYY-MM-DD'.
    n_samples : int, optional
        If provided, specifies a desired total number of samples in the dataset,
        adjusting `n_days` accordingly to meet this target.
    as_frame : bool, default=False
        Determines if the dataset is returned as a pandas DataFrame, facilitating
        further analysis and manipulation within the pandas ecosystem.
    return_X_y : bool, default=False
        When True, the function returns the predictors (`X`) and the target variable
        (`y`) as separate entities, simplifying their use in machine learning pipelines.
    target_name : str or None, default=None
        Specifies the name of the target variable within the dataset, e.g., 'temperature'
        for temperature prediction tasks. If None, a default target variable is selected.
    noise_level : float, default=0.05
        The standard deviation of Gaussian noise added to the simulated weather data,
        representing measurement inaccuracies or minor variations.
    seed : int or None, optional
        A seed for the random number generator, ensuring reproducible results across
        different executions of the simulation.
    include_extreme_events : bool, default=False
        If set to True, the simulation includes random extreme weather events, adding
        anomalies to the weather variables to represent occurrences like heatwaves or
        heavy rainfall.

    Returns
    -------
    Depending on the specified `as_frame` and `return_X_y` flags, the output can be
    a Bunch object containing the dataset and metadata, a pandas DataFrame, or a tuple
    of arrays (`X`, `y`) for the features and target variable.

    Examples
    --------
    >>> from gofast.datasets.simulate import simulate_weather_forecasting
    >>> dataset = simulate_weather_forecasting(
    ...     n_days=30, include_extreme_events=True, as_frame=True)
    >>> print(dataset.head())
            date  temperature  ...  air_pressure  temperature_next_day
    0 2020-01-01    26.514123  ...   1024.981290             24.305035
    1 2020-01-02    24.479263  ...   1010.193422             17.464121
    2 2020-01-03    17.121617  ...   1015.476951             27.989124
    3 2020-01-04    28.176325  ...   1000.415557             21.561623
    4 2020-01-05    21.669945  ...   1024.878211             15.727065
    
    [5 rows x 7 columns]
    
    Notes
    -----
    This synthetic dataset is ideal for conducting exploratory data analysis and
    developing predictive models in the domain of weather forecasting. It enables
    practitioners to experiment with various modeling approaches, from time series
    analysis to deep learning, in predicting future weather conditions based on
    historical patterns and atmospheric observations.
    """
    func_name = inspect.currentframe().f_code.co_name
    dataset_descr, features_descr= fetch_simulation_metadata (func_name)
    
    np.random.seed(seed)
    if not weather_variables:
        weather_variables = ['temperature', 'humidity', 'wind_speed',
                             'precipitation', 'air_pressure']
    if not target_name:
        target_name = ['temperature_next_day']

    weather_variables= is_iterable(
        weather_variables, exclude_string= True, transform=True )
    if n_samples:
        # Adjust n_days based on n_samples
        n_samples = validate_positive_integer(n_samples, "n_samples")
        adjusted_values = adjust_parameters_to_fit_samples(
            n_samples, {'n_days': n_days})
        n_days = adjusted_values.get('n_days', 60)

    start_date, _= validate_dates(
        start_date, end_date= get_last_day_of_current_month(return_today=True), 
        return_as_date_str= True 
        )
    date_range = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    # Generate synthetic weather data
    temperature = np.random.normal(20, 5, n_days)  # Mean temp of 20°C with std dev of 5
    humidity = np.random.uniform(40, 90, n_days)  # Humidity ranging from 40% to 90%
    wind_speed = np.random.uniform(0, 15, n_days)  # Wind speed 0 to 15 km/h
    precipitation = np.random.choice([0, 1], size=n_days, p=[0.8, 0.2]
                                     ) * np.random.uniform(0, 10, n_days)  # 20% chance of rain
    air_pressure = np.random.normal(1013, 10, n_days)  # Average air pressure in hPa

    weather_data = pd.DataFrame({
        'date': date_range,
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'precipitation': precipitation,
        'air_pressure': air_pressure
    })

    # Introduce extreme weather events if specified
    if include_extreme_events:
        extreme_days = np.random.choice(
            n_days, size=n_days // 20, replace=False)  # 5% of days
        weather_data.loc[extreme_days, 'temperature'] += np.random.normal(
            0, 10, len(extreme_days))
        weather_data.loc[extreme_days, 'precipitation'] *= np.random.uniform(
            2, 5, len(extreme_days))
        weather_data.loc[extreme_days, 'wind_speed'] *= np.random.uniform(
            2, 3, len(extreme_days))

    # Next day's temperature for regression target
    weather_data['temperature_next_day'] = np.roll(weather_data['temperature'], -1)

    # Add noise
    if noise_level:
        # validate noise level 
        noise_level = validate_noise_level( noise_level) if noise_level else noise_level 
        for var in weather_variables:
            weather_data[var] += np.random.normal(0, noise_level * np.std(
                weather_data[var]), n_days)
    
    return manage_data(
        data=weather_data, as_frame=as_frame, return_X_y=return_X_y,
        target_names=target_name if target_name else "temperature_next_day",
        seed=seed,
        DESCR=dataset_descr, features_descr=features_descr  
    )


def simulate_default_loan(
    *, n_samples=1000, 
    credit_score_range=None,
    age_range=None, 
    loan_amount_range=None,
    interest_rate_range=None, 
    loan_term_months=None,
    employment_length_range=None, 
    annual_income_range=None,
    default_rate=0.15, 
    as_frame=False, 
    return_X_y=False,
    target_name='default', 
    noise_level=None, 
    seed=None
):
    """
    Simulates a dataset for loan default prediction based on borrower profiles,
    loan characteristics, and the likelihood of defaulting. This dataset supports
    binary classification tasks where the target is predicting the likelihood of
    loan defaults. The dataset creation process is highly customizable, allowing
    for simulation under various scenarios and borrower profiles.

    Parameters
    ----------
    n_samples : int, default=1000
        The number of loan samples to generate in the dataset.
        
    credit_score_range : tuple of int, optional
        The range (inclusive) of credit scores to randomly generate for borrowers.
        Default value is (300, 850) when None.
        
    age_range : tuple of int, optional
        The range (inclusive) of borrower ages to randomly generate.
        Default value is (18, 70) when None.
        
    loan_amount_range : tuple of int, optional
        The range (inclusive) of loan amounts ($) to randomly generate.
        Default value is (5000, 50000) when None.
        
    interest_rate_range : tuple of float, optional
        The range (inclusive) of interest rates (as a percentage) to randomly generate.
        Default value is (5, 20) when None.
        
    loan_term_months : list of int, optional, 
        A list of possible loan terms in months that a borrower can choose from.
        Default=[12, 24, 36, 48, 60] when None
        
    employment_length_range : tuple of int, optional
        The range (inclusive) of employment lengths (in years) to randomly generate.
        Default value is (0, 30) when None.
        
    annual_income_range : tuple of int, optional
        The range (inclusive) of annual incomes ($) to randomly generate.
        Default value is (20000, 150000) when None.
        
    default_rate : float, default=0.15
        The probability that a loan will default. This rate is used to generate
        the binary default indicator (0 for non-default, 1 for default) for 
        each loan in the dataset.
        
    as_frame : bool, default=False
        If True, the generated dataset is returned as a pandas DataFrame, enhancing
        readability and convenience for further analysis and model training.
        
    return_X_y : bool, default=False
        If True, separates the dataset into features (X) and the target variable (y),
        returning them as separate entities, suitable for direct use in machine 
        learning models.
        
    target_name : str, default='default'
        The name assigned to the target variable column in the generated dataset,
        representing the default status of each loan.
        
    noise_level : float, default=0.05
        Specifies the standard deviation of Gaussian noise added to all numerical
        features in the dataset to simulate measurement errors or variations in data.
        
    seed : int, optional
        A seed for the random number generator to ensure reproducibility of the dataset
        across different runs or simulations.

    Returns
    -------
    Depending on the `as_frame` and `return_X_y` parameters, this function can return
    either a pandas DataFrame, a tuple containing the features matrix `X` and the target
    vector `y`, or a Bunch object containing the dataset and metadata.
    
    Examples
    --------
    >>> from gofast.datasets.simulate import simulate_default_loan
    >>> dataset = simulate_default_loan(
    ...   n_samples=500, as_frame=True, noise_level=.05)
    >>> print(dataset.head())
       credit_score        age  ...  annual_income  default
    0    757.994122  59.986021  ...   94578.987783        0
    1    448.931573  62.046920  ...   64234.094531        0
    2    520.973391  61.016563  ...   69659.998526        0
    3    845.878802  67.170633  ...   82933.929032        0
    4    453.111953  50.014340  ...   24343.994096        0
       
    [5 rows x 8 columns]

    Notes
    -----
    The generated dataset aims to provide a realistic approximation of factors influencing
    loan defaults. It is suitable for training and evaluating machine learning models tasked
    with binary classification of loan default risks. Customizable ranges and parameters
    allow for simulation under diverse scenarios, making it a versatile tool for model
    development and testing in financial analytics.
    """
    func_name = inspect.currentframe().f_code.co_name
    dataset_descr, features_descr= fetch_simulation_metadata (func_name)
    np.random.seed(seed)
    
    loan_term_months= ( 
        loan_term_months if loan_term_months else [12, 24, 36, 48, 60]
      ) 
    # convert to iterable 
    loan_term_months= is_iterable(
        loan_term_months, exclude_string= True,transform= True  ) 
    
    if any(isinstance(s, str) for s in loan_term_months):
        raise TypeError(
            "The 'loan_term_months' parameter expects a list of integers"
            " representing loan terms in months. Each entry in the list"
            " should be a positive integer indicating the duration of the"
            " loan term. Please ensure that 'loan_term_months' contains only"
            " integer values, without any strings or non-integer types."
        )

    validated_params = validate_loan_parameters(
        {
            'credit_score_range': credit_score_range,
            'age_range': age_range, 
            'loan_amount_range': loan_amount_range, 
            'interest_rate_range': interest_rate_range,
            'employment_length_range': employment_length_range, 
            'annual_income_range': annual_income_range
         },
         error='ignore'
    )
    # validate ranges 
    credit_score_range= validated_params.get("credit_score_range") 
    age_range= validated_params.get("age_range")
    loan_amount_range= validated_params.get("loan_amount_range")
    interest_rate_range= validated_params.get("interest_rate_range")
    employment_length_range= validated_params.get("employment_length_range")
    annual_income_range= validated_params.get("annual_income_range")
    
    # Generate synthetic data
    n_samples= validate_positive_integer(n_samples, "n_samples")
    credit_scores = np.random.randint(*credit_score_range, n_samples)
    ages = np.random.randint(*age_range, n_samples)
    loan_amounts = np.random.randint(*loan_amount_range, n_samples)
    interest_rates = np.random.uniform(*interest_rate_range, n_samples)
    loan_terms = np.random.choice(loan_term_months, n_samples)
    employment_lengths = np.random.randint(*employment_length_range, n_samples)
    annual_incomes = np.random.randint(*annual_income_range, n_samples)
    defaults = np.random.binomial(1, default_rate, n_samples)
    
    # Compile data into a DataFrame
    target_name= target_name or 'default'
    data = pd.DataFrame({
        'credit_score': credit_scores,
        'age': ages,
        'loan_amount': loan_amounts,
        'interest_rate': interest_rates,
        'loan_term': loan_terms,
        'employment_length': employment_lengths,
        'annual_income': annual_incomes,
        target_name: defaults
    })
    
    # Add Gaussian noise to numerical features if specified
    if noise_level:
        noise_level= validate_noise_level(noise_level, default_value=.05)
        numerical_features = ['credit_score', 'age', 'loan_amount', 'interest_rate',
                              'employment_length', 'annual_income']
        for feature in numerical_features:
            data[feature] += np.random.normal(0, noise_level, n_samples)
    
    return manage_data(
        data=data, as_frame=as_frame, return_X_y=return_X_y,
        target_names=target_name,
        seed=seed,
        DESCR=dataset_descr, features_descr=features_descr  
    )

def simulate_traffic_flow(
    *, n_samples=10000, 
    start_date="2021-01-01 00:00:00",
    end_date="2021-12-31 23:59:59", 
    traffic_flow_range=(100, 1000),
    time_increments='hour', 
    special_events_probability=0.05,
    road_closure_probability=0.01, 
    as_frame=False, 
    return_X_y=False,
    target_name='traffic_flow', 
    noise_level=0.1,
    seed=None
    ):
    """
    Generates a comprehensive synthetic dataset for traffic flow prediction, 
    covering various aspects of urban traffic dynamics. The dataset simulates
    detailed traffic conditions over a specified period, taking into account
    the fluctuating impact of time increments, special events, and road closures 
    on traffic flow. It's designed to aid in the development of robust 
    traffic forecasting models for urban planning and congestion management.

    Parameters:
    ----------
    n_samples : int, default=10000
        The total number of traffic observations to generate. Each sample 
        represents a unique combination of conditions affecting traffic flow.

    start_date : str, default="2021-01-01 00:00:00"
        Marks the beginning of the simulation period. The datetime format allows 
        for precise modeling of traffic conditions from this starting point.

    end_date : str, default="2021-12-31 23:59:59"
        Defines the end of the simulation period. This parameter ensures the 
        temporal scope of the dataset encompasses a full range of traffic patterns 
        over time, including seasonal variations.

    traffic_flow_range : tuple, default=(100, 1000)
        Specifies the minimum and maximum range of traffic flow rates to simulate.
        This range allows for diverse traffic conditions, from light to heavy flow.

    time_increments : str, default='hour'
        Dictates the granularity of time intervals for each traffic observation.
        Finer increments such as 'half-hour' can capture rapid fluctuations in 
        traffic flow, enhancing the dataset's fidelity.

    special_events_probability : float, default=0.05
        Sets the likelihood of a special event (e.g., concerts, sports events) 
        occurring within a given time increment, which typically leads to spikes 
        in traffic volume.

    road_closure_probability : float, default=0.01
        Determines the chance of road closures due to construction or accidents,
        significantly altering usual traffic routes and flow rates.

    as_frame : bool, default=False
        When True, outputs the dataset in a structured pandas DataFrame format,
        facilitating immediate data exploration and analysis in Python.

    return_X_y : bool, default=False
        If set to True, the function outputs the dataset split into features (X) 
        and the target variable (y), aligning with the common format for machine 
        learning model inputs.

    target_name : str, default='traffic_flow'
        Names the target variable reflecting traffic flow rates, enabling easy 
        identification within the dataset for modeling purposes.

    noise_level : float, default=0.1
        Introduces a realistic variance to the traffic data through Gaussian noise, 
        simulating measurement errors or unexpected disruptions in traffic flow.

    seed : int, optional
        Optional seed value to initialize the random number generation, ensuring
        consistent replication of the dataset across multiple simulations.

    Returns:
    -------
    The function's output format is configurable through `as_frame` and 
    `return_X_y`, offering a versatile dataset representation for a variety of 
    analytical and modeling needs. The dataset can be used directly for training 
    and testing machine learning models focused on predicting traffic conditions.

    Examples:
    --------
    >>> from gofast.datasets.simulate import simulate_traffic_flow
    >>> dataset = simulate_traffic_flow(n_samples=500, 
    ...                                    special_events_probability=0.1,
    ...                                    as_frame=True)
    >>> print(dataset.head())
                 datetime  special_event  road_closure  traffic_flow
    0 2021-01-01 00:00:00              0             0    648.455524
    1 2021-01-01 01:00:00              0             0    330.535273
    2 2021-01-01 02:00:00              0             0    263.264355
    3 2021-01-01 03:00:00              0             0    927.035411
    4 2021-01-01 04:00:00              0             0    674.648777
        
    Notes:
    -----
    The simulated traffic dataset encapsulates a rich set of features impacting
    urban traffic flow, designed to challenge and validate forecasting models.
    It offers an invaluable tool for traffic analysts and modelers aiming to 
    improve predictive accuracy and understand the multifaceted nature of 
    traffic dynamics.
    """

    func_name = inspect.currentframe().f_code.co_name
    dataset_descr, features_descr= fetch_simulation_metadata (func_name)
    
    np.random.seed(seed)
    n_samples= validate_positive_integer(n_samples, "n_samples")
    
    start_date, end_date = validate_dates(
        start_date, end_date, return_as_date_str=True, 
        date_format= "%Y-%m-%d %H:%M:%S"
        )
    # adjust na_samples accordingly 
    
    # Map user-friendly time increment strings to pandas frequency strings
    increments_map = {
        'hour': 'H',  # Map 'hour' to 'H'
        'half-hour': '30T',  # Example: map 'half-hour' to '30T'
        'quarter-hour': '15T',  # Every 15 minutes
        'minute': 'T',  # Every minute
        'day': 'D',  # Daily frequency
        'week': 'W',  # Weekly frequency
        'month': 'M',  # Monthly frequency
        'quarter': 'Q',  # Quarterly frequency
        'year': 'A',  # Annual frequency
    } 
    # Validate and convert the time increment to pandas frequency string
    # Default to hourly if not found
    pd_freq = increments_map.get(time_increments.lower(), 'H')  
    
    # Now use the correct pandas frequency string
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date)
    date_range = pd.date_range(start=start_datetime, end=end_datetime, freq=pd_freq)
    n_days = len(date_range)
    
    # Adjust date_range to fit the number of samples. 
    adjust_params= adjust_parameters_to_fit_samples(
        n_samples, initial_guesses= {"n_days":len(date_range)}
        )
    n_days= adjust_params.get("n_days", 30 )
    # now take the date from start_date to fit n_days. 
    date_range = date_range[: n_days]

    traffic_flow_range= validate_length_range(traffic_flow_range )
    traffic_flows = np.random.randint(*traffic_flow_range, len(date_range))
    special_events = np.random.binomial(1, special_events_probability, len(date_range))
    road_closures = np.random.binomial(1, road_closure_probability, len(date_range))

    # Compiling the dataset
    data = pd.DataFrame({
        'datetime': date_range,
        'special_event': special_events,
        'road_closure': road_closures, 
        'traffic_flow': traffic_flows,
    })

    # Adding noise to the traffic_flow data if specified
    if noise_level and noise_level > 0:
        noise_level = validate_noise_level( noise_level, default_value= .1 ) 
        data['traffic_flow'] += np.random.normal(
            0, noise_level * data['traffic_flow'].mean(), len(date_range))
    
    # Setting the target
    if not target_name:
        target_name = 'traffic_flow'
   
    return manage_data(
        data=data, as_frame=as_frame, return_X_y=return_X_y,
        target_names=target_name,
        seed=seed,
        DESCR=dataset_descr, features_descr=features_descr  
    )

def simulate_medical_diagnosis(
    *, n_patients=1000, 
    n_symptoms=10, 
    n_lab_tests=5, 
    diagnosis_options=None, 
    n_samples=None, 
    as_frame=False, 
    return_X_y=False, 
    target_name=None, 
    noise_level=None, 
    seed=None
    ):
    """
    Creates a synthetic dataset to simulate patient medical diagnosis data. 
    
    This dataset includes patient symptoms, lab test results, and diagnosis
    outcomes, suitable for training and testing machine learning models in 
    medical diagnostics and decision support.

    Parameters:
    ----------
    n_patients : int, default=1000
        The number of patients to simulate in the dataset. Each patient will 
        have a set of symptoms and lab test results associated with them.

    n_symptoms : int, default=10
        The number of different symptoms to simulate for each patient. Symptoms
        are binary variables (1 for presence, 0 for absence).

    n_lab_tests : int, default=5
        The number of lab tests results to simulate for each patient. Lab test
        results are continuous variables, represented as floats.

    diagnosis_options : list of str, optional
        The list of possible diagnosis outcomes for each patient. This list 
        can be customized to include any number of diseases or health conditions.
        If None. default is ['Disease A', 'Disease B', 'Healthy']
    n_samples : int, optional
        If specified, dynamically adjusts `n_patients` to meet the desired 
        number of samples, allowing for flexible dataset sizing.

    as_frame : bool, default=False
        If True, the dataset is returned as a pandas DataFrame, facilitating 
        direct data manipulation and analysis using pandas.

    return_X_y : bool, default=False
        If True, separates the dataset into feature vectors (X) and the target
        variable (y), simplifying integration into machine learning workflows.

    target_name : str or None, default=None
        Names the target variable column. If None, defaults to 'diagnosis',
        which is the label indicating the patient's diagnosis outcome.

    noise_level : float or None, optional
        Specifies the standard deviation of Gaussian noise to add to lab test
        results, simulating variability and measurement error in medical tests.

    seed : int or None, optional
        A seed value for the random number generator to ensure the 
        reproducibility of the dataset across simulations.

    Returns
    -------
    Depending on `as_frame` and `return_X_y`, the function can return a pandas
    DataFrame, a tuple of `(X, y)`, or a Bunch object containing the dataset 
    and metadata.

    Examples
    --------
    >>> from gofast.datasets.simulate import simulate_medical_diagnosis
    >>> dataset = simulate_medical_diagnosis(n_patients=500, as_frame=True)
    >>> print(dataset.head())
        symptom_1  symptom_2  ...  lab_test_5                diagnosis
     0        1.0        1.0  ...    0.441992  Coronary Artery Disease
     1        0.0        1.0  ...    0.613877              Chicken Pox
     2        1.0        1.0  ...    0.580177                Dyspepsia
     3        1.0        1.0  ...    0.637615                 HIV/AIDS
     4        1.0        1.0  ...    0.449666  Coronary Artery Disease
    
     [5 rows x 16 columns]
     
    Notes
    -----
    This dataset provides a basis for developing machine learning models that 
    can assist in medical diagnosis by analyzing patterns in symptoms and lab 
    test results. It simulates realistic scenarios encountered in medical 
    practice, allowing for the exploration of diagnostic models and their 
    potential to improve patient outcomes.
    """
    func_name = inspect.currentframe().f_code.co_name
    dataset_descr, features_descr= fetch_simulation_metadata (func_name)
    
    diagnosis_options= select_diagnostic_options(diagnosis_options, 6)
    np.random.seed(seed)
    n_patients= validate_positive_integer (n_patients, "n_patients")
    n_symptoms= validate_positive_integer (n_symptoms, "n_symptoms")
    n_lab_tests= validate_positive_integer (n_lab_tests, "n_lab_tests")
    # Adjust the number of patients and symptoms/lab tests 
    # if n_samples is provided
    if n_samples:
        n_samples = validate_positive_integer(n_samples, "n_samples")
        adjusted_values = adjust_parameters_to_fit_samples(
            n_samples, {'n_patients': n_patients, 'n_symptoms': n_symptoms, 
                        'n_lab_tests': n_lab_tests})
        n_patients = adjusted_values.get('n_patients', 100)
        n_lab_tests = adjusted_values.get('n_lab_tests', 3)
        n_symptoms = adjusted_values.get('n_symptoms', 7)
        
    # Generate synthetic symptoms data (1 for presence, 0 for absence)
    symptoms_data = np.random.randint(0, 2, size=(n_patients, n_symptoms))
    
    # Generate synthetic lab test results (floats within a reasonable range)
    lab_tests_data = np.random.normal(loc=0.5, scale=0.1, size=(n_patients, n_lab_tests))
    if noise_level:
        noise_level=validate_noise_level(noise_level )
        lab_tests_data += np.random.normal(0, noise_level, lab_tests_data.shape)

    # Generate diagnosis labels
    diagnosis_labels = np.random.choice(diagnosis_options, size=n_patients)

    # Combine all data into a DataFrame if as_frame=True
    columns = [f'symptom_{i+1}' for i in range(n_symptoms)] + \
              [f'lab_test_{i+1}' for i in range(n_lab_tests)]
    data = np.hstack((symptoms_data, lab_tests_data))
    df = pd.DataFrame(data, columns=columns)
    df['diagnosis'] = diagnosis_labels

    return manage_data(
        data=df, as_frame=as_frame, return_X_y=return_X_y,
        target_names=target_name if target_name else "diagnosis",
        seed=seed,
        DESCR=dataset_descr, features_descr=features_descr  
    )

def simulate_retail_sales(
    *, n_products=100, 
    n_days=365, 
    include_promotions=True, 
    include_seasonality=True, 
    complex_seasonality=False, 
    include_economic_factors=True, 
    n_samples=None, 
    as_frame=False, 
    return_X_y=False, 
    target_name=None, 
    noise_level=0.05,
    seed=None
    ):
    """
    Simulates a dataset for retail sales forecasting, incorporating factors 
    like product promotions, seasonal trends, and economic indicators. This 
    dataset is designed for regression tasks aimed at predicting future sales 
    volumes based on historical data and external influences.

    Parameters
    ----------
    n_products : int, default=100
        This parameter specifies the total number of unique products included
        in the simulation. Each product will have its sales data generated 
        across the specified number of days. Adjusting this parameter allows
        for the simulation of datasets with varying product diversities, 
        simulating scenarios from small retail outlets with a limited product
        range to large supermarkets or online retailers offering a wide array
        of items.

    n_days : int, default=365
        Defines the duration, in days, for which the sales data is generated.
        Setting this parameter allows the dataset to encompass a range of 
        timeframes, from short-term sales analyses to longer-term trends and 
        seasonal patterns. This temporal dimension is crucial for modeling 
        and forecasting exercises that require understanding how sales volumes
        change over time.
    
    include_promotions : bool, default=True
        When set to True, this parameter introduces promotional activities 
        into the dataset, simulating the impact of sales, discounts, and 
        marketing campaigns on product sales volumes. Promotions are a critical
        factor in retail sales dynamics, often leading to significant increases
        in consumer purchases. Including this factor enables the simulation 
        of more realistic sales patterns and provides an opportunity to analyze
        the effectiveness of promotional strategies.
    
    include_seasonality : bool, default=True
        This parameter controls the simulation of seasonal variations in sales
        data. Seasonality is a fundamental aspect of retail sales, with consumer
        purchasing behaviors showing marked fluctuations in response to seasons,
        holidays, and special occasions. By enabling seasonality, the dataset
        reflects the cyclical nature of sales, facilitating the development of
        forecasting models that can anticipate periodic sales peaks and troughs.

    include_economic_factors : bool, default=True
        This parameter determines the inclusion of economic variables that
        might affect retail sales, such as GDP growth rates, inflation, or
        consumer confidence indices. Incorporating these factors into the
        dataset allows for a more holistic analysis of sales dynamics, as
        economic conditions significantly influence consumer spending
        behavior. Enabling this feature enhances the dataset's realism and
        applicability for models that account for macroeconomic trends.

    n_samples : int or None, optional
        This parameter allows for the specification of the total number of
        samples to be generated in the dataset. When provided, it
        dynamically adjusts the `n_products` and `n_days` parameters to
        achieve the desired sample size. This flexibility is particularly
        useful for creating datasets of specific sizes to meet the
        computational or methodological requirements of different analysis
        or modeling tasks.

    as_frame : bool, default=False
        Setting this parameter to True returns the simulated dataset as a
        pandas DataFrame, facilitating easy manipulation, exploration, and
        visualization of the data within the pandas ecosystem. This format
        is particularly useful for data scientists and analysts who prefer
        working with DataFrame structures for data processing and analysis
        tasks.

    return_X_y : bool, default=False
        When enabled, this parameter separates the dataset into a features
        matrix `X` and a target vector `y`, conforming to the standard input
        format expected by many machine learning algorithms. This
        separation simplifies the process of feeding the data into
        predictive models, making it an essential utility for machine
        learning workflows.

    target_name : str or None, default=None
        This parameter allows for the customization of the column name for
        the target variable in the dataset. By default, the target variable
        is named 'sales'. However, users can specify an alternative name to
        align with their specific analytical or modeling context. This
        feature adds a layer of personalization to the dataset, catering to
        diverse project requirements.

    noise_level : float, default=0.05
        The `noise_level` parameter specifies the standard deviation of
        Gaussian noise to be added to the numerical features in the dataset.
        This addition introduces a realistic degree of variation and
        uncertainty into the data, simulating the inherent randomness and
        measurement errors present in real-world retail sales data. Tuning
        the noise level allows researchers and modelers to assess the
        robustness of their analytical methods under different levels of
        data quality and variability.

    seed : int or None, optional
        Providing a seed value initializes the random number generator used
        to create the dataset, ensuring that the dataset generation process
        is reproducible. This reproducibility is crucial for experimental
        consistency, as it allows for the exact replication of datasets
        across different runs, facilitating comparative analyses and the
        validation of modeling approaches.

    Returns
    -------
    Depending on the combination of `as_frame` and `return_X_y`, this function 
    can return a structured Bunch object, a DataFrame, or a tuple containing 
    the features matrix `X` and the target vector `y`.

    Examples:
    --------
    >>> from gofast.datasets.simulate import simulate_retail_sales
    >>> dataset = simulate_retail_sales(n_products=50, n_days=180, as_frame=True)
    >>> print(dataset.head())
            sales  promotions  seasonality  economic_factors  future_sales
    0   95.024977           1     1.000000          1.001834           113
    1   81.036977           0     1.017213          1.034794           115
    2  107.985552           0     1.034422          1.037967           101
    3  108.047064           0     1.051620          1.069003           109
    4  110.070907           0     1.068802          0.979844           112
    Notes:
    -----
    This simulated dataset provides a comprehensive basis for developing and 
    testing predictive models in the retail sector. It models complex real-world 
    influences on sales, offering valuable insights for forecasting future 
    trends and understanding the impact of promotions, seasonality, and economic 
    conditions on retail performance.
    """  
    func_name = inspect.currentframe().f_code.co_name
    dataset_descr, features_descr= fetch_simulation_metadata (func_name)
    
    np.random.seed(seed)

    # Validate and adjust parameters
    n_products = validate_positive_integer(n_products, "n_products")
    n_days = validate_positive_integer(n_days, "n_days")
    noise_level = validate_noise_level(noise_level)

    if n_samples:
        adjusted_values = adjust_parameters_to_fit_samples(
            n_samples, {'n_products': n_products, 'n_days': n_days})
        n_products, n_days = ( 
            adjusted_values.get('n_products'), adjusted_values.get('n_days')
            )

    # Generate sales data
    sales_data = np.random.poisson(lam=100, size=(n_products, n_days)).astype(float)
    sales_data += np.random.normal(0, noise_level, sales_data.shape)

    # Simulate promotional data
    if include_promotions:
        promotions = np.random.binomial(1, 0.1, size=(n_products, n_days))

    # Simulate seasonal effects
    if include_seasonality:
        days = np.arange(n_days)
        if complex_seasonality:
            seasonality = np.cos(2 * np.pi * days / 365) + np.sin(
                4 * np.pi * days / 365) + 1
        else:
            seasonality = np.sin(2 * np.pi * days / 365) + 1

    # Simulate economic factors
    if include_economic_factors:
        economic_factors = np.random.normal(1.0, 0.05, size=(n_days))

    # Combine all data
    data_dict = {'sales': sales_data}
    if include_promotions:
        data_dict['promotions'] = promotions
    if include_seasonality:
        seasonality_effect = np.tile(seasonality, (n_products, 1))
        data_dict['seasonality'] = seasonality_effect
    if include_economic_factors:
        economic_effect = np.tile(economic_factors, (n_products, 1))
        data_dict['economic_factors'] = economic_effect

    # Construct DataFrame
    df = pd.DataFrame({key: np.ravel(value) for key, value in data_dict.items()})
    future_sales = np.random.poisson(lam=np.mean(sales_data, axis=1) * 1.1, size=n_products)
    target_name = target_name or 'future_sales'
    df[target_name] = np.tile(future_sales, n_days)

    # Manage and return data according to specified parameters
    return manage_data(
        data=df, as_frame=as_frame, return_X_y=return_X_y,
        target_names=target_name, seed=seed,
        DESCR=dataset_descr, features_descr=features_descr  
    )
        
        
        
        
        
        
        