# model.py
# contains functions to calculate the likelihood based on a linear and quadratic model given data and parameters
import numpy as np
import pymc3 as pm
import arviz as az

def calc_linear_likelihood(data, m, b):
    """
    Calculates the likelihood based on a linear model given the data and parameters (m, b)
    model: temp = slope*depth + intercept

    Parameters
    ----------
    data : pandas DataFrame
        data and metadata contained in pandas DataFrame
        Format described in tutotial notebook
    m, b : floats
        parameter values used in calculation of likelihood 
    
    Returns
    -------
    likelihood : double
        The likelihood for a linear model given the data and specified parameters 
    """
    
    # prepare data
    depth = data['Depth'].values
    temp = data['Temperature'].values
    temp_error = data['temp_errors'].values
    likelihood =  np.prod(1. / np.sqrt(2 * np.pi * temp_error ** 2) * np.exp(-(temp - m * depth - b)**2 / (2 * temp_error ** 2) ) )
    return likelihood


def calc_quad_likelihood(data, q, m, b):
    """
    Calculates the likelihood based on a quadratic model given the data and parameters (m, b)
    model: temp = q*depth^2 + m*depth + b

    Parameters
    ----------
    data : pandas DataFrame
        data and metadata contained in pandas DataFrame
        Format described in tutotial notebook
    q, m, b : floats
        parameter values used in calculation of likelihood 
    
    Returns
    -------
    likelihood : double
        The likelihood for a quadratic model given the data and specified parameters 
    """
    
    # prepare data
    depth = data['Depth'].values
    temp = data['Temperature'].values
    temp_error = data['temp_errors'].values
    likelihood =  np.prod(1. / np.sqrt(2 * np.pi * temp_error ** 2) * np.exp(-(temp - q*depth**2 - m * depth - b)**2 / (2 * temp_error ** 2) ) )
    return likelihood


def fit_quad(data):
    """
    Fits the data to a quadratic function
    Only errors on temperature are considered in this model
    Based on Hogg, Bovy, and Lang section 1 (https://arxiv.org/abs/1008.4686)
    model: temp = q*depth^2 + m*depth + b

    Parameters
    ----------
    data : pandas DataFrame
        data and metadata contained in pandas DataFrame
        Format described in tutotial notebook

    Returns
    -------
    q, m, b (parameters, 1D array of length 3), covariance matrix (3 by 3) : floats, matrix of floats
        parameters and covariance matrix from fit to quadratic function

    """

    # prepare data
    depth = data['Depth'].values
    temp = data['Temperature'].values
    sigma_y = data['temp_errors'].values

    # define quantities from HBL equation 2, 3, and 4
    Y = temp
    A = depth[:, np.newaxis] ** (0, 1, 2)
    C = np.diag(sigma_y ** 2)


    C_inv = np.linalg.inv(C)
    cov = np.linalg.inv(A.T @ C_inv @ A)
    params = cov @ (A.T @ C_inv @ Y)
    return params, cov


def fit_quad_MCMC(data):
    """
    Fits the data to a quadratic function using pymc3
    Errors on temperature are considered in the model
    model: temp = q*depth^2 + m*depth + b

    Parameters
    ----------
    data : pandas DataFrame
        data and metadata contained in pandas DataFrame
        Format described in tutotial notebook

    Returns
    -------
    q, m, b (1D array of parameters), covariance matrix (3x3 matrix) : floats
        parameter values from quadratic model, covariance matrix

    """
    
    # prepare data
    depth = data['Depth'].values
    temp = data['Temperature'].values
    sigma_y = data['temp_errors'].values

    # initial guess
    initial_guess = {'m':0.00, 'b':0.00, 'q':0.00}

    with pm.Model() as quad_model:
        # define quadratic model
        q = pm.Flat('q')
        m = pm.Flat('m')
        b = pm.Flat('b')
        line = q**2 * depth + m * depth + b

        # define likelihood
        likelihood = pm.Normal("temp_pred", mu = line, sd = sigma_y, observed=temp)

        # unleash the inference
        n_tuning_steps = 2500
        ndraws = 500
        traces = pm.sample(tune=n_tuning_steps, draws=ndraws, chains=1)
        az.plot_trace(trace)

        #params = pm.find_MAP(model=quad_model, start = initial_guess, return_raw=True)
        #covariance_matrix = np.flip(scipy_output.hess_inv.todense()/uncertainty)


    return 0, 0#params, cov_mat
