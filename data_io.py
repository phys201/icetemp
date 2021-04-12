# data_io.py
# modeled on https://github.com/phys201/example/blob/main/example/data_io.py
import os
import pandas as pd

# load data in an os-independent way
def load_ice_data(filename, data_year, temp_errors, depth_errors, data_dir='south_pole_ice_temperature_data_release'):
    '''
    Returns data contained in filename (.txt extension included) as a Pandas dataframe including the year the data was taken and the errors on temp & depth 

    '''

    # default: data located in south_pole_ice_temperature_data_release directory
    # need to navigate there 
    start = os.path.abspath(__file__)
    start_dir = os.path.dirname(start)
    data_dir = os.path.join(start_dir, data_dir)
    data_path = os.path.join(start_dir, data_dir, filename)



    data = pd.read_csv(data_path, header=None, sep=' |\t', names = ["Temperature", "Depth"])

    #add metadata
    data['data_year'] = data_year
    data['temp_errors'] = temp_errors 
    data['depth_errors'] = depth_errors 

    return data
