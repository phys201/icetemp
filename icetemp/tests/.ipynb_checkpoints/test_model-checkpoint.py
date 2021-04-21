# unit testing for data_io
from unittest import TestCase

import numpy as np
import pandas as pd
import icetemp.model as mod

# arbitrary column entries for testing
data_year = 0
temp_error = 1 / np.sqrt(2*np.pi)

# generate test DataFrame
test_size = 30
m = 1.
b = 0.
q = 3.
depths = np.random.rand(test_size) * 1000 # random numbers between 0-1000
line_temps = m*depths + b
quad_temps = q*(depths**2) + m*depths + b

line_test_df = pd.DataFrame({'Temperature': line_temps, 'Depth': depths, 'temp_errors': np.array([temp_error]*test_size)})
quad_test_df = pd.DataFrame({'Temperature': quad_temps, 'Depth': depths, 'temp_errors': np.array([temp_error]*test_size)})

# object to handle unit testing using nosetests
class TestModel(TestCase):
	def test_linear_likelihood(self):
		'''
		Computes and tests result of linear likelihood fit of the random test data
		'''
		self.assertTrue(np.abs(mod.calc_linear_likelihood(line_test_df, m, b) - 1) < 1e-6)

	def test_quadratic_likelihood(self):
		'''
		Computes and tests result of quadratic likelihood fit of the test data
		'''
		self.assertTrue(np.abs(mod.calc_quad_likelihood(quad_test_df, q, m, b) - 1) < 1e-6)

