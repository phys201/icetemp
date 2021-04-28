# unit testing for data_io
from unittest import TestCase

import numpy as np
import pandas as pd
import icetemp.model as mod

# arbitrary column entries for testing
data_year = 0
temp_error_llh = 1 / np.sqrt(2*np.pi)

# generate test DataFrames for likelihood calculation
test_size = 30
m = 1.
b = 0.
q = 3.
depths = np.random.rand(test_size) * 1000 # random numbers between 0-1000
line_temps = m*depths + b
quad_temps = q*(depths**2) + m*depths + b

line_llh_test_df = pd.DataFrame({'Temperature': line_temps, 'Depth': depths, 'temp_errors': np.array([temp_error_llh]*test_size)})
quad_llh_test_df = pd.DataFrame({'Temperature': quad_temps, 'Depth': depths, 'temp_errors': np.array([temp_error_llh]*test_size)})

# grenerate test DataFrames for fitting
temp_error_fit = 0.0001
quad_fit_test_df = pd.DataFrame({'Temperature': quad_temps, 'Depth': depths, 'temp_errors': np.array([temp_error_fit]*test_size)})

# generate timetable for GPR
timetable_test = pd.DataFrame({'year': [2001, 2005, 2008, 2009], 
                          'Temperature': [-40., -39.5, -38.2, -37.], 
                          'prediction_errors': [0.1, 0.1, 0.1, 0.1]})

# object to handle unit testing using nosetests
class TestModel(TestCase):
	def test_linear_likelihood(self):
		'''
		Computes and tests result of linear likelihood fit of the random test data
		'''
		self.assertTrue(np.abs(mod.calc_linear_likelihood(line_llh_test_df, m, b) - 1) < 1e-6)

	def test_quadratic_likelihood(self):
		'''
		Computes and tests result of quadratic likelihood fit of the test data
		'''
		self.assertTrue(np.abs(mod.calc_quad_likelihood(quad_llh_test_df, q, m, b) - 1) < 1e-6)
	
	def test_quadratic_algebraic_fit(self): 
		'''
		Tests algebraic quadatic regression result on dummy data
		'''
		params, _ = mod.fit_quad(quad_fit_test_df)
		self.assertTrue(np.abs(params[0] - b) < 1e-3)
		self.assertTrue(np.abs(params[1] - m) < 1e-3)
		self.assertTrue(np.abs(params[2] - q) < 1e-3)

	def test_gpr_fit_compiles(self):
		'''
		Tests whether or not the GPR sampling model actually compiles
		'''
		try:
			gpr = mod.fit_GPR(timetable_test, nosetest=True)
			self.assertTrue(True)
		except:
			self.assertTrue(False)


