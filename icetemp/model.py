# model.py
# contains functions to calculate the likelihood based on a linear and quadratic model given data and parameters
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import pandas as pd
from scipy import stats

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
    params, param_errors: 1D numpy arrays of floats
        parameter values from the model
        standard deviations of each parameter

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

        # extract parameters and uncertainty using arviz
        params_list = []
        params_uncert = []
        for parameter in ['b', 'm', 'q']:
            params_list.append(az.summary(traces, round_to=9)['mean'][parameter])
            params_uncert.append(az.summary(traces, round_to=9)['sd'][parameter])

        params = np.array(params_list)
        param_errors = np.array(params_uncert)
    return params, param_errors

def n_polyfit_MCMC(n, data):
    """
    Fits the data to a quadratic function using pymc3
    Errors on temperature are considered in the model
    model: temp = q*depth^2 + m*depth + b
    Plots the traces in the MCMC

    Parameters
    ----------
    n: integer
        indicates the power of the polynomial fit
    data : pandas DataFrame
        data and metadata contained in pandas DataFrame
        Format described in tutotial notebook

    Returns
    -------
    params, param_errors: 1D numpy arrays of floats
        parameter values from the model
        standard deviations of each parameter

    """
    # prepare data
    depth = data['Depth'].values
    temp = data['Temperature'].values
    sigma_y = data['temp_errors'].values

    with pm.Model() as _:
        # define priors for each parameter in the polynomial fit (e.g C_0 + C_1*x + C_2*x^2 + ...)
        C_n = [pm.Uniform('C_{}'.format(i), -100, 100) for i in range(n+1)]
        polynomial = np.sum([C_n[i] * depth**i for i in range(n+1)])

        # define likelihood
        y_obs = pm.Normal("temp_pred", mu = polynomial, sd = sigma_y, observed=temp)

        # unleash the inference
        n_tuning_steps = 2000
        ndraws = 2500
        traces = pm.sample(start=pm.find_MAP(), tune=n_tuning_steps, draws=ndraws, chains=4) # need at least two chains to use following arviz function
        az.plot_trace(traces)

        # extract parameters and uncertainty using arviz
        params_list = []
        params_uncert = []
        for parameter in ['C_{}'.format(i) for i in range(n+1)]:
            params_list.append(az.summary(traces, round_to=9)['mean'][parameter])
            params_uncert.append(az.summary(traces, round_to=9)['sd'][parameter])

        params = np.array(params_list)
        param_errors = np.array(params_uncert)
    return params, param_errors

def get_timetable(n, data):
    """
    Fits the data to a quadratic function using pymc3
    Errors on temperature are considered in the model
    model: temp = q*depth^2 + m*depth + b
    Plots the traces in the MCMC

    Parameters
    ----------
    n: integer
        indicates the power of the polynomial fit
    data : pandas DataFrame
        data and metadata contained in pandas DataFrame
        Format described in tutotial notebook

    Returns
    -------
    timetable: pandas DataFrame
        data and metadata contained in pandas DataFrame
        Format described in tutotial notebook

    """
    # range of depth locations
    x = np.linspace(800,2500)

    year_list = []
    temp_list = []
    pred_errs_list = []
    for year in range(len(data)):
        params, errors = n_polyfit_MCMC(n, data[year]) # returns params in order C_0, C_1, C_2,...
        print("Paremters from MCMC for the year {}".format(data[year]['data_year'][0]))
        print(params)

        year_list.append(data[year]['data_year'][0])
        temp_list.append(params[0])
        pred_errs_list.append(errors[0])

        polynomial = np.sum([params[i] * x**i for i in range(n+1)], axis = 0)
        data[year].plot(x='Depth', y='Temperature', kind='scatter', yerr=0.1,color='orange')
        plt.plot(x, polynomial, linestyle='dashed', color='blue')
        plt.title(r'Real data with polynomial [$x^{}$] fit (parameters from MCMC) for {}'.format(n, data[year]['data_year'][0]))

    timetable = pd.DataFrame({'year': year_list, 'temperature': temp_list, 'prediction_errors': pred_errs_list})
    return timetable

def get_odds_ratio(timetable1, timetable2):
    """
    Computes the odds ratio between two models based on the normal distribution of the ground level temperature.

    Parameters
    ----------
    timetable1, timetable2: pandas DataFrame
        data and metadata contained in pandas DataFrame
        Format described in tutotial notebook


    Returns
    -------
    odds_ratio: float
        Determines a favorable model out of the two models.

    """
    # range of depth locations
    x = np.linspace(800,2500)

    likelihood1 = np.prod(stats.norm.pdf(x, timetable1['temperature'], timetable1['prediction_errors']))
    likelihood2 = np.prod(stats.norm.pdf(x, timetable2['temperature'], timetable2['prediction_errors']))

    return np.divide(likelihood1/likelihood2)

def fit_GPR(timetable, plot_post_pred_samples=False, num_post_pred_samples=150):
    '''
    Performs a Gaussian Process Regression to infer temperature v. time dependence

    Parameters
    ----------
    timetable: pandas DataFrame
        DataFrame of data and metadata for temperatures at a certain depth over a large period of time
        Incoporates the following columns: year, temperature, prediction_errors (error on regressions from above)
    pplot_post_pred_samples: bool
        If false, the posterior predictive distribution is plotted by visualizing the mean PPC values + 1-sigma std
        devs at each X-value. If true, posterior predictive distribution is plotted by visualizing samples from the
        posterio predictive distribution.
    num_post_pred_samples: int
        Number of poster predictive samples to take when visualizing posterior predictive distribution of fit

    Returns
    -------
    marginal_gp_model: pymc3 model
        model that will later allow us to sample and plot the posterior predictive distribution of
        temperature vs. time
    '''

    # data extraction
    X = timetable['year'].values[:, None]
    y = timetable['temperature'].values
    sigma = timetable['prediction_errors'].values

    # compute mean of data
    mu = y.mean()

    with pm.Model() as marginal_gp_model:
        # prior on length scale. change prior with future iterations of model
        ls = pm.Gamma('ls', 1, 0.5)

        # (Vinny) can also make the noise level and mean into parameters, if
        # you want:
        #sigma = pm.HalfCauchy('sigma', 1)
        #mu = pm.Normal('mu', y.mean(), sigma)

        # Specify the mean and covariance functions
        cov_func = pm.gp.cov.ExpQuad(1, ls=ls)
        mean_func = pm.gp.mean.Constant(mu)

        # Specify the GP.
        gp = pm.gp.Marginal(cov_func=cov_func, mean_func=mean_func)

        # set the marginal likelihood based on the training data
        # and give it the noise level
        y_ = gp.marginal_likelihood("y", X=X, y=y, noise=sigma)

    # perform hyperparameter sampling (this trains our model)
    with marginal_gp_model:
        traces = pm.sample()

    # plot posterior
    az.plot_trace(traces)
    plt.show()

    # predictions
    range_x = np.max(X) - np.min(X)
    Xnew = np.linspace(np.min(X) - 0.2*range_x, np.max(X) + 0.2*range_x, 100)[:, None]
    with marginal_gp_model:
        y_pred = gp.conditional('y_pred', Xnew)
        ppc = pm.sample_posterior_predictive(traces, var_names=['y_pred'], samples=num_post_pred_samples)

    # plotting
    plt.errorbar(X, y, yerr=sigma, c='red', fmt='o', label='True data')

    # plot posterior predictive samples
    if plot_post_pred_samples:
        plt.plot(Xnew, ppc['y_pred'].T, c='grey', alpha=0.1)

    # plot posterior predictive distribution
    else:
        # get mean and std of posterior predictive distribution
        mean_ppc = np.mean(ppc['y_pred'], axis=0)
        std_ppc = np.std(ppc['y_pred'], axis=0)

        # plotting
        plt.plot(Xnew, mean_ppc, c='grey', label='Posterior predictive mean')
        plt.fill_between(Xnew.flatten(), mean_ppc - std_ppc, mean_ppc + std_ppc, color='grey', alpha=0.1, label='1$\\sigma$ posterior predictive region')

    plt.xlabel('Time [years]')
    plt.ylabel('Temperature [$^\\ocirc C$]')
    plt.title('Temperature vs. time\nPosterior of Gaussian Process Regression')
    plt.legend()
    plt.show()

    return marginal_gp_model
