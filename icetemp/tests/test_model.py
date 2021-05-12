# unit testing for model.py
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

# generate test DataFrames for fitting
temp_error_fit = 0.0001
quad_fit_test_df = pd.DataFrame({'Temperature': quad_temps, 'Depth': depths, 'temp_errors': np.array([temp_error_fit]*test_size)})

# generate timetable for GPR
timetable_test = pd.DataFrame({'year': [2001, 2005, 2008, 2009], 
    'temperature': [-40., -39.5, -38.2, -37.], 
    'prediction_errors': [0.1, 0.1, 0.1, 0.1]})

# object to handle unit testing using nosetests
class TestModel(TestCase):
    # testing helper functions
    def test_input_get_timetable(self):
        '''
        Give get_timetable() invalid input and make sure exception is thrown
        '''
        # define data to be wrong type
        data = [5, 10]  # list of wrong type (ints instead of pandas dataframe)
        params_list = ["test"]
        params_errors_list = ["test"]

        self.assertRaises(TypeError, mod.get_timetable, data, params_list, params_errors_list)
        self.assertRaises(TypeError, mod.get_timetable,  quad_fit_test_df, params_list, params_errors_list)

    # testing non-helper functions
    def test_linear_likelihood(self):
        '''
        Computes and tests result of linear likelihood fit of the random test data
        Tests if exception is thrown when given invalid data input 
        '''
        data = "not_data"
        self.assertTrue(np.abs(mod.calc_linear_likelihood(line_llh_test_df, b, m) - 1) < 1e-6)
        self.assertRaises(TypeError, mod.calc_linear_likelihood, data, b, m)
    def test_quadratic_likelihood(self):
        '''
        Computes and tests result of quadratic likelihood fit of the test data
        Tests if exception is thrown when given invalid data input
        '''
        data = [5, 10, 15, 20]
        self.assertTrue(np.abs(mod.calc_quad_likelihood(quad_llh_test_df, b, m, q) - 1) < 1e-6)
        self.assertRaises(TypeError, mod.calc_quad_likelihood, data, b, m, q)  

    def test_quadratic_algebraic_fit(self): 
        '''
        Tests algebraic quadratic regression result on dummy data
        Tests if exception is thrown when given invalid data input
        '''
        params, _ = mod.fit_quad(quad_fit_test_df)
        self.assertTrue(np.abs(params[0] - b) < 1e-3)
        self.assertTrue(np.abs(params[1] - m) < 1e-3)
        self.assertTrue(np.abs(params[2] - q) < 1e-3)
        data = 5576567.88
        self.assertRaises(TypeError, mod.fit_quad, data)

    def test_quadratic_MCMC_fit(self):
        '''
        Tests whether or not the pymc3 model in fit_quad_MCMC compiles 
        Does not perform inference/sampling in order to save time on tests 
        Tests if exception is thrown when given invalid data input 
        '''
        #dictionary of guesses
        init_guess = {'C_0':1.0,'C_1':0.0,'C_2':3.0} 
        try:
            _ = mod.fit_quad_MCMC(quad_fit_test_df, init_guess, nosetest=True)
        except:
            self.fail("fit_quad_MCMC() raised ExceptionType unexpectedly!")
        data = 110101
        self.assertRaises(TypeError, mod.fit_quad_MCMC, data, init_guess, nosetest=True)  

    def test_n_polyfit_MCMC(self):
        '''
        Tests whether or not the pymc3 model in n_polyfit_MCMC compiles for polynomials of degree 1 through 10
        Does not perform inference/sampling in order to save time on tests
        '''
        for i in range(2, 12):
            param_names = ['C_{}'.format(j) for j in range(i)]
            init_guess = {param: {0.0} for param in param_names}
            try: 
                _ = mod.n_polyfit_MCMC(i, quad_fit_test_df, init_guess, nosetest=True)
            except:
                self.fail("n_polyfit_MCMC() raised ExceptionType unexpectedly with n = {}!".format(i - 1))

    def test_gpr_fit_compiles(self):
        '''
        Tests whether or not the GPR sampling model actually compiles
        '''
        try:
            _ = mod.fit_GPR(timetable_test, nosetest=True)
        except:
            self.fail("fit_GPR() raised ExceptionType unexpectedly!")

    def test_get_odds_ratio(self):
        '''
        Tests to make sure correct exceptions are raised with incorrect dtypes for get_odds_ratio
        '''
        n_M1, n_M2, data, best_fit1, best_fit2 = [], [], [], [], []
        self.assertRaises(TypeError, mod.get_odds_ratio, n_M1, n_M2, data, best_fit1, best_fit2)

    def test_plot_ployfit(self):
        '''
        Tests to make sure correct exceptions are thrown with incorrect dtypes for plot_polyfit
        '''
        data = 0
        best_fit_list = [np.array([1])]
        self.assertRaises(TypeError, mod.plot_polyfit, data, best_fit_list)

