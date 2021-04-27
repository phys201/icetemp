# model.py
# contains functions to calculate the likelihood based on a linear and quadratic model given data and parameters
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import theano.tensor as tt

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


def fit_quad_MCMC(data, init_guess):
    """
    Fits the data to a quadratic function using pymc3
    Errors on temperature are considered in the model
    model: temp = q*depth^2 + m*depth + b
    Plots the traces in the MCMC

    Parameters
    ----------
    data : pandas DataFrame
        data and metadata contained in pandas DataFrame
        Format described in tutotial notebook
    init_guess : dict
        dictionary containing initial values for each of the parameters in the model

    Returns
    -------
    b, m, q (1D array of parameters) : floats
        parameter values from the model 

    """
    # prepare data
    depth = data['Depth'].values
    temp = data['Temperature'].values
    sigma_y = data['temp_errors'].values

    with pm.Model() as _:
        # define priors for each parameter in the quadratic fit
        m = pm.Uniform('m', -100, 100)
        b = pm.Uniform('b', -100, 100)
        q = pm.Uniform('q', -100, 100)
        line = q * depth**2 + m * depth + b

        # define likelihood
        y_obs = pm.Normal("temp_pred", mu = line, sd = sigma_y, observed=temp)

        # unleash the inference
        n_tuning_steps = 2000
        ndraws = 2500
        traces = pm.sample(start=init_guess, tune=n_tuning_steps, draws=ndraws, chains=2) # need at least two chains to use following arviz function
        az.plot_trace(traces)

        # extract parameters using arviz
        q = az.summary(traces, round_to=9)['mean']['q']
        m = az.summary(traces, round_to=9)['mean']['m']
        b = az.summary(traces, round_to=9)['mean']['b']
        
        # Output as dataframe
        # samples = pm.trace_to_dataframe(traces)
    return b, m, q

def fit_GPR(timetable):
    '''
    Performs a Gaussian Process Regression to infer temperature v. time dependence

    Parameters
    ----------
    timetable: pandas DataFrame
        DataFrame of data and metadata for temperatures at a certain depth over a large period of time
        Incoporates the following columns: year, Temperature, prediction_errors (error on regressions from above)

    Returns
    -------
    gpr_model: pymc3 model
        model that will later allow us to sample and plot the posterior predictive distribution of
        temperature vs. time
    '''
    
    # data extraction
    years = timetable['year'].values
    temps = timetable['Temperature'].values
    pred_errors = timetable['prediction_errors'].values

    # defining the model and sampling
    with pm.Model() as gpr_model:

        l = pm.Uniform("l", 0, 10)
        eta = pm.HalfCauchy("η", beta=5)
        
        cov = eta**2 * pm.gp.cov.ExpQuad(1, l)
        gp = pm.gp.Latent(cov_func=cov)
        
        # the gaussian process prior
        f = gp.prior("f", X=years)
        
        y_obs = pm.Normal("y", mu=f, sd = pred_errors, observed=temps)

        # sampling
        trace = pm.sample(2000)

    # traceplotting
    pm.traceplot(trace[1000:], varnames=['l', 's2_f', 's2_n'])

    # compute samples from posterior predictive distribution
    year_range = np.max(years) - np.min(years)
    Z = np.linspace(np.min(years) - year_range/0.15, np.max(years) + year_range/0.15, 100)
    with gpr_model:
        gp_samples = pm.gp.sample_gp(trace[1000:], y_obs, Z, samples=50, random_seed=42)

    # plottting PPD
    _, ax = plt.subplots(figsize=(14,5))
    [ax.plot(Z, x, color=cm(0.3), alpha=0.3) for x in gp_samples]
    ax.plot(years, temps, 'ok', ms=10)
    ax.set_xlabel('Time [years]')
    ax.set_ylabel('Temperature [$^\\circ C$]')
    ax.set_title('Posterior predictive distribution')

    return gpr_model

    