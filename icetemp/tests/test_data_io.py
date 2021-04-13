# unit testing for data_io
from unittest import TestCase

import pandas as pd
import icetemp.data_io as di

# arbitrary column entries for testing
data_year = 0
temp_error = 0.5
depth_error = 17

# load test data
line_df = di.load_ice_data('test_data_linear.txt', data_year=data_year, temp_errors=temp_error, depth_errors=depth_error, data_dir='test_data')
quad_df = di.load_ice_data('test_data_quadratic.txt', data_year=data_year, temp_errors=temp_error, depth_errors=depth_error, data_dir='test_data')

# get column list
line_columns = line_df.columns
quad_columns = quad_df.columns

# object to handle unit testing using nosetests
class TestDataIO(TestCase):
	def test_is_dataframe(self):
		'''
		Makes sure that when we load the test data, we are created a Pandas DataFrame as expected
		'''
		self.assertTrue(isinstance(line_df, pd.DataFrame))
		self.assertTrue(isinstance(quad_df, pd.DataFrame))
	
	def test_number_columns(self):
		'''
		Makes sure that the dataframes we load have the correct number of columns
		'''
		self.assertTrue(len(line_columns) == 5)
		self.assertTrue(len(quad_columns) == 5)

	def test_column_entries(self):
		'''
		Makes sure that the dataframe we create have the correct column entries
		'''
		self.assertTrue(set(line_columns) == set(quad_columns))
		self.assertTrue(set(line_columns) == {'Temperature', 'Depth', 'data_year', 'temp_errors', 'depth_errors'})

	def test_column_values(self):
		'''
		Makes sure we see the correct entries in the data_year, temp_errors, and depth_errors columns
		'''
		# all entries are the same
		self.assertTrue(len(set(line_df['data_year'].values)) == 1)
		self.assertTrue(len(set(quad_df['data_year'].values)) == 1)
		self.assertTrue(len(set(line_df['temp_errors'].values)) == 1)
		self.assertTrue(len(set(quad_df['temp_errors'].values)) == 1)
		self.assertTrue(len(set(line_df['depth_errors'].values)) == 1)
		self.assertTrue(len(set(quad_df['temp_errors'].values)) == 1)

		# entries are what we'd expect
		self.assertTrue(line_df['data_year'].values[0] == data_year)
		self.assertTrue(quad_df['data_year'].values[0] == data_year)
		self.assertTrue(line_df['temp_errors'].values[0] == temp_error)
		self.assertTrue(quad_df['temp_errors'].values[0] == temp_error)
		self.assertTrue(line_df['depth_errors'].values[0] == depth_error)
		self.assertTrue(quad_df['depth_errors'].values[0] == depth_error)
