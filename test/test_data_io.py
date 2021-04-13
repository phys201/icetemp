# test_data_io.py
# unit testing to be run with nosetests to ensure proper data handling
from icetemp import data_io as di

# load up test sets
line_test_df = di.load_ice_data("test_data_linear.txt", data_year=0, temp_errors=0.1, depth_errors=2, data_dir='test_data')
quad_test_df = di.load_ice_data('test_data_quadratic.txt', data_year=2002, temp_errors=0.4, depth_errors=1, data_dir='test_data')

print(line_test_df.shape)
print(quad_test_df.columns)
