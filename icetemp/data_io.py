# data_io.py
# modeled on https://github.com/phys201/example/blob/main/example/data_io.py
import os
import numpy as np
import pandas as pd

# load data in an os-independent way
def load_ice_data(filename, data_year, temp_errors, depth_errors, data_dir='south_pole_ice_temperature_data_release'):
    """
    Given a filename, create a pandas DataFrame object containing data and other metadata specified when calling function

    Parameters
    ----------
    filename : string
        string containing name of file containing data INCLUDING extention
        EX <filename>.txt
    data_year : int
        year in which data was taken
    temp_errors : float
        error on temperature measurements (constant across all data points)
    depth_errors : float
        error on depth measurements (constant across all data points)
    data_dir : string
        string that specifies directory in which data lives in relation to top directory

    Returns
    -------
    data : pandas DataFrame
        pandas DataFrame containing data and metadata
        
    """
    # default: data located in south_pole_ice_temperature_data_release directory
    # need to navigate there 
    start = os.path.abspath(__file__)
    start_dir = os.path.dirname(start)
    data_dir = os.path.join(start_dir, data_dir)
    data_path = os.path.join(start_dir, data_dir, filename)
    
    # create pandas DataFrame object from our data
    data = pd.read_csv(data_path, header=None, delim_whitespace=True, names=["Temperature", "Depth"], engine='python')
    # specifying the engine prevents a warning message, since we are using regex separators

    # depths should be positive
    data['Depth'] = np.abs(data['Depth'].values)

    #add metadata
    data['data_year'] = data_year
    data['temp_errors'] = temp_errors 
    data['depth_errors'] = depth_errors

    return data

# create temperature vs. time data
def timeify(dfs, years):
    '''
    Function that aggregates temperature vs. depth data over a period of years

    Parameters
    ----------
    dfs : list of pd.DataFrames
        each data frame has columns Temperature, Depth, prediction_errors for predicted temperature at a certain depth
    years : list
        list of years that each data frame in 'dfs' was collected

    Returns
    -------
    timetable: pd.DataFrame
        concatenated data frame that now incorporates timing information
    '''

    # append years to each dataframe
    for y, df in enumerate(dfs):
        df['year'] = years[y]

    # concatenate dataframes together
    return pd.concat(dfs)
