# model.py
# contains functions to infer parameters in polynomial fits to the data, calculate likelihoods (given parameters and data), and calculate odds ratios
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import pandas as pd
from scipy import stats

def calc_linear_likelihood(data, C_0, C_1):
    """
    Calculates the likelihood based on a linear model given the data and parameters (C_0, C_1)
    model: temp = C_1*depth + C_0

    Parameters
    ----------
    data : pandas DataFrame
        data and metadata contained in pandas DataFrame
        Format described in tutorial notebook
    C_0, C_1 : floats
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
    likelihood =  np.prod(1. / np.sqrt(2 * np.pi * temp_error ** 2) * np.exp(-(temp - C_1 * depth - C_0)**2 / (2 * temp_error ** 2) ) )
    return likelihood


def calc_quad_likelihood(data, C_0, C_1, C_2):
    """
    Calculates the likelihood based on a quadratic model given the data and parameters (m, b)
    model: temp = C_2*depth^2 + C_1*depth + C_0

    Parameters
    ----------
    data : pandas DataFrame
        data and metadata contained in pandas DataFrame
        Format described in tutorial notebook
    C_0, C_1, C_2 : floats
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
    likelihood =  np.prod(1. / np.sqrt(2 * np.pi * temp_error ** 2) * np.exp(-(temp - C_2*depth**2 - C_1 * depth - C_0)**2 / (2 * temp_error ** 2) ) )
    return likelihood

def fit_quad(data):
    """
    Fits the data to a quadratic function
    Only errors on temperature are considered in this model
    Based on Hogg, Bovy, and Lang section 1 (https://arxiv.org/abs/1008.4686)
    model: temp = C_2*depth^2 + C_1*depth + C_0

    Parameters
    ----------
    data : pandas DataFrame
        data and metadata contained in pandas DataFrame
        Format described in tutorial notebook

    Returns
    -------
    params, param_errors: 1-D numpy arrays of floats
        parameter values from the model
        standard deviations of each parameter
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
    cov_mat = np.linalg.inv(A.T @ C_inv @ A)
    params = cov_mat @ (A.T @ C_inv @ Y)

    #get stdev of parameters from covariance matrix
    param_errors = np.sqrt(np.diag(cov_mat))
    return params, cov_mat


def fit_quad_MCMC(data, init_guess, n_tuning_steps = 1500, n_draws = 2500, n_chains = 4, nosetest=False):
    """
    Fits the data to a quadratic function using pymc3
    Errors on temperature are considered in the model
    model: temp = C_2*depth^2 + C_1*depth + C_0 — flat prior on C_1 & C_2, uniform prior on C_0
    Plots the traces in the MCMC (if n_chains > 2)

    Parameters
    ----------
    data : pandas DataFrame
        data and metadata contained in pandas DataFrame
        Format described in tutorial notebook
    init_guess : dict
        dictionary containing initial values for each of the parameters in the model (C_0, C_1, C_2))
    n_tuning_steps : int (>= 0)
        number of tuning steps used in MCMC (default = 1500)
        NOTE: Number of tuning steps must be >= 0
        If < 0, n_tuning_steps will automatically be set to the default (1500)
    n_draws : int (> 0)
        number of draws used in MCMC (default = 2500)
        NOTE: n_draws must be >= 4 for convergence checks and > 0 in general
        If < 1, n_draws will automatically be set to the default (2500)
    n_chains : int (> 0)
        number of walkers used to sample posterior in MCMC (default = 5)
        NOTE: number of chains must be >= 2 to visualize traces and must be > 0 in general
        If < 1, n_chains will automatically be set to the default (4)
    nosetest : bool
        bool that specifies whether or not a test is being conducted
        if testing is being run, then sampling will not be performed

    Returns
    -------
    traces : pymc3 MultiTrace object
        Traces generated from MCMC sampling

    NOTE: when testing, None is returned, as no sampling/inference is performed

    """
    # error checking for MCMC-related parameters
    # if parameters outside allowed values, set them to the default
    if n_tuning_steps < 0:
        print("You have entered an invalid value for n_tuning_steps (must be >= 0). Reverting to default (1500)")
        n_tuning_steps = 1500
    if n_draws < 1:
        print("You have entered an invalid value for n_draws (must be >= 1). Reverting to default (2500)")
        n_draws = 2500
    if n_chains < 1:
        print("You have entered an invalid value for n_chains (must be >= 1). Reverting to default (4)")
        n_chains = 4


    # prepare data
    depth = data['Depth'].values
    temp = data['Temperature'].values
    sigma_y = data['temp_errors'].values

    # define degree of polynomial
    n = 2 # for quadratic

    with pm.Model() as quad_model:
        # define priors for each parameter in the quadratic fit
        C_0 = pm.Uniform('C_0', -55, -43) # constrain based on surface temps in austral summer
        C_n = [pm.Uniform('C_{}'.format(i), -60/800**i, 10/800**i) for i in range(1, n+1)]
        line = C_0 + np.sum([C_n[i] * depth**(i+1) for i in range(n)])

        # define (Gaussian) likelihood
        y_obs = pm.Normal("temp_pred", mu = line, sd = sigma_y, observed=temp)

    # if a test, do not sample, just return
    if not nosetest:
        with quad_model:
            # unleash the inference
            traces = pm.sample(start=init_guess, tune=n_tuning_steps, draws=n_draws, chains=n_chains) # need at least two chains to use following arviz function
            # plot traces if n_chains >= 2
            if n_chains >= 2:
                az.plot_trace(traces)

    return traces if not nosetest else None


def n_polyfit_MCMC(n, data, init_guess, n_tuning_steps = 1500, n_draws = 2500, n_chains = 4, nosetest=False, compute_traces = False):
    """
    Fits the data to a polynomial function of degree n  using pymc3
    Errors on temperature are considered in the model
    model: temp = C_0 + C_1 * depth + C_2 * depth ^2 + ... + C_n * depth^n — uniform priors on all parameters bounded by Antarctic ice temps
    Plots the traces in the MCMC (if n_chains > 2)

    Parameters
    ----------
    data : pandas DataFrame
        data and metadata contained in pandas DataFrame
        Format described in tutorial notebook
    init_guess : dict
        dictionary containing initial values for each of the parameters in the model (C_0, C_1, C_2))
    n_tuning_steps : int (>= 0)
        number of tuning steps used in MCMC (default = 1500)
        NOTE: Number of tuning steps must be >= 0
        If < 0, n_tuning_steps will automatically be set to the default (1500)
    n_draws : int (> 0)
        number of draws used in MCMC (default = 2500)
        NOTE: n_draws must be >= 4 for convergence checks and > 0 in general
        If < 1, n_draws will automatically be set to the default (2500)
    n_chains : int (> 0)
        number of walkers used to sample posterior in MCMC (default = 5)
        NOTE: number of chains must be >= 2 to visualize traces and must be > 0 in general
        If < 1, n_chains will automatically be set to the default (4)
    nosetest : bool
        bool that specifies whether or not a test is being conducted
        if testing is being run, then sampling will not be performed
    compute_traces : bool
        bool that indicates wheter or not to compute the traces

    Returns
    -------
    traces : pymc3 MultiTrace object
        Traces generated from MCMC sampling
    best_fit : dict
        dictionary containing best-fit parameters and covariance matrix

    NOTE: when testing, None is returned, as no sampling/inference is performed

    """
    # error checking for MCMC-related parameters
    # if parameters outside allowed values, set them to the default
    if n_tuning_steps < 0:
        print("You have entered an invalid value for n_tuning_steps (must be >= 0). Reverting to default (1500)")
        n_tuning_steps = 1500
    if n_draws < 1:
        print("You have entered an invalid value for n_draws (must be >= 1). Reverting to default (2500)")
        n_draws = 2500
    if n_chains < 1:
        print("You have entered an invalid value for n_chains (must be >= 1). Reverting to default (4)")
        n_chains = 4



    # prepare data
    depth = data['Depth'].values
    temp = data['Temperature'].values
    sigma_y = data['temp_errors'].values

    with pm.Model() as poly_model:
        # define priors for each parameter in the polynomial fit (e.g C_0 + C_1*x + C_2*x^2 + ...)
        C_0 = pm.Uniform('C_0',-55,-43) # not expected to change more than +/- 5 deg C according to base camp measurements
        C_n = [pm.Uniform('C_{}'.format(i), -60/800**i, 10/800**i) for i in range(1,n+1)]
        polynomial =  C_0 + np.sum([C_n[i] * depth**(i+1) for i in range(n)])

        # define likelihood
        y_obs = pm.Normal("temp_pred", mu = polynomial, sd = 1, observed=temp)

    if not nosetest:
        with poly_model:
            # unleash the inference
            if compute_traces == True:
                traces = pm.sample(init="adapt_diag", tune=n_tuning_steps, draws=n_draws, chains=n_chains) # need at least two chains to plot traces
                #az.plot_pair(traces, divergences=True)

                if n_chains >= 2:
                    az.plot_trace(traces)
            else:
                traces = 0

            best_fit, scipy_output = pm.find_MAP(start = init_guess, return_raw=True)
            covariance_matrix = np.flip(scipy_output.hess_inv.todense()/1)
            best_fit['covariance matrix'] = covariance_matrix

    return (traces, best_fit) if not nosetest else None


def get_params(n, input_params, input_traces = False):
    """
    Helper function to extract parameters from fit to polynomial of degree n using pymc3 traces

    Parameters
    ----------
    n: integer
        indicates the power of the polynomial fit
    input_params : can be either best_fit or traces

        best_fit : dict with 1-D numpy arrays of floats
            parameter values from the model and covariance matrix
        traces : pymc3 MultiTrace object
            Traces generated from MCMC sampling
    input_traces : bool
        indicates whether input is best_fit or traces

    Returns
    -------
    params, param_errors: 1-D numpy arrays of floats
        parameter values from the model
        standard deviations of each parameter

    """

    # extract parameters and uncertainty using arviz
    if input_traces == True:

        params_list = []
        params_uncert = []
        for parameter in ['C_{}'.format(i) for i in range(n+1)]:
            params_list.append(az.summary(input, round_to=9)['mean'][parameter])
            params_uncert.append(az.summary(input_params, round_to=9)['sd'][parameter])
        params = np.array(params_list)
        params_errors = np.array(params_uncert)


    else:
        best_fit_list = []
        best_fit_errors_list = []

        parameters = []
        errors = {}
        for i in range(n+1):
            parameters = parameters + ['C_{}'.format(i)]
            errors['C_{}'.format(i)]= np.sqrt(input_params['covariance matrix'][i][i])
        for parameter in parameters:
            best_fit_list.append(input_params[parameter])
            best_fit_errors_list.append(errors[parameter])

        params = np.array(best_fit_list)
        params_errors = np.array(best_fit_errors_list)


    return params, params_errors



def plot_polyfit(data, best_fit_list):
    """
    Helper function which fits the data to a polynomial function from pymc3 results

    Parameters
    ----------
    data : list with names for pandas DataFrame
        list of names of data and metadata contained in pandas DataFrame
    best_fit_list: list with 1-D numpy arrays of floats
        parameter values from the model

        NOTE:
        order in which lists of data, params_list and params_errors_list are constructed should be the same, i.e.
        data = [data_2002, data_2007]
        params_list = [params_2002, params_2007]
    """

    # range of depth locations
    x = np.linspace(800,2500)


    for year in range(len(data)):
        print("Paremters from MCMC for the year {}".format(data[year]['data_year'][0]))
        print(best_fit_list[year])

        n = len(best_fit_list[year])
        polynomial = np.sum([best_fit_list[year][i] * x**i for i in range(n)], axis = 0)
        data[year].plot(x='Depth', y='Temperature', kind='scatter', yerr=0.1,color='orange')
        plt.plot(x, polynomial, linestyle='dashed', color='blue')
        plt.title(r'Real data with polynomial [$x^{}$] fit (parameters from MCMC) for {}'.format(n-1, data[year]['data_year'][0]))


def get_timetable(data, best_fit_list, best_fit_errors_list):
    """
    Collects the ground level temperature from multiple datasets

    Parameters
    ----------
    data : list with names for pandas DataFrame
        list of names of data and metadata contained in pandas DataFrame
    best_fit_list, best_fit_errors_list: list with 1-D numpy arrays of floats
        parameter values from the model
        standard deviations of each parameter

        NOTE:
        order in which lists of data, params_list and params_errors_list are constructed should be the same, i.e.
        data = [data_2002, data_2007]
        best_fit_list = [params_2002, params_2007]

    Returns
    -------
    timetable: pandas DataFrame
        data and metadata contained in pandas DataFrame
        Format described in tutotial notebook

    """
    # error checking
    if not all(isinstance(entries, pd.DataFrame) for entries in data):
        print("data is not the right type!")
        raise TypeError


    # define lists used in calculation
    year_list = []
    temp_list = []
    pred_errs_list = []

    for year in range(len(data)):

        year_list.append(data[year]['data_year'][0])
        temp_list.append(best_fit_list[year][0])
        pred_errs_list.append(best_fit_errors_list[year][0])

    timetable = pd.DataFrame({'year': year_list, 'temperature': temp_list, 'prediction_errors': pred_errs_list})
    return timetable


def get_odds_ratio(n_M1, n_M2, data, best_fit1, best_fit2):
    """
    Computes the odds ratio between two models given the data, degrees of polynomial fits,
    best-fit parameters, and covariance matrices

    Parameters
    ----------
    n_M1, n_M2: integer
        describes the highest order (n) of the polynomial from each model
    data : list with names for pandas DataFrame
        list of names of data and metadata contained in pandas DataFrame
    best_fit1, best_fit2 : list with 1-D numpy arrays of floats
        parameter values from the model 1 and model 2, respectively

    Returns
    -------
    odds_ratio: float
        Odds ratio in Bayesian model comparison
    """
    odds_ratio_list = []

    for year in range(len(data)):

        # prepare data
        depth = data[year]['Depth'].values
        temp = data[year]['Temperature'].values
        temp_error = data[year]['temp_errors'].values # expected equal errors for any temperature

        # range of depth locations
        x = data[year]['Depth'].values
        x.sort()

        #makes sure the temp is sorted to get correct residuals
        temp.sort()

        coeff1 = [best_fit1['C_{}'.format(i)] for i in range(n_M1+1)]
        coeff2 = [best_fit2['C_{}'.format(i)] for i in range(n_M2+1)]

        mu1 = np.sum([coeff1[i] * x**i for i in range(n_M1+1)], axis = 0)
        mu2 = np.sum([coeff2[i] * x**i for i in range(n_M2+1)], axis = 0)

        chi_squared1 = np.sum((temp - mu1)**2 / ( temp_error[0] ** 2))
        chi_squared2 = np.sum((temp - mu2)**2 / ( temp_error[0] ** 2))

        max_loglikelihood1 =  np.log(1. / (2 * np.pi * temp_error[0] ** 2)** (len(data[year])/2)) - chi_squared1/2
        max_loglikelihood2 =  np.log(1. / (2 * np.pi * temp_error[0] ** 2)** (len(data[year])/2)) - chi_squared2/2

        curvature1 = np.sqrt(np.linalg.det(best_fit1['covariance matrix'])) * (2 * np.pi) **  (n_M1/2)
        curvature2 = np.sqrt(np.linalg.det(best_fit2['covariance matrix'])) * (2 * np.pi) **  (n_M2/2)

        prior_C0 = 1. / (-44 - -52)
        prior_Cn_M1 = np.prod([1. / (10/800**i - -60/800**i) for i in range(n_M1)])
        prior_Cn_M2 = np.prod([1. / (10/800**i - -60/800**i) for i in range(n_M2)])

        loglikelihood1 = max_loglikelihood1 * np.log(curvature1) * np.log(prior_C0 * prior_Cn_M1)
        loglikelihood2 = max_loglikelihood2 * np.log(curvature2) * np.log(prior_C0 * prior_Cn_M2)

        odds_ratio_list.append(np.exp(loglikelihood1 - loglikelihood2))

    return odds_ratio_list


def fit_GPR(timetable, num_forecast_years=0, plot_post_pred_samples=False, num_post_pred_samples=150, nosetest=False):
    '''
    Performs a Gaussian Process Regression to infer temperature v. time dependence

    Parameters
    ----------
    timetable: pandas DataFrame
        DataFrame of data and metadata for temperatures at a certain depth over a large period of time
        Incorporates the following columns: year, temperature, prediction_errors (error on regressions from above)
    num_forecast_years: int
        Number of years ahead of the last date to forecast the temperature
    pplot_post_pred_samples: bool
        If false, the posterior predictive distribution is plotted by visualizing the mean PPC values + 1-sigma std
        devs at each X-value. If true, posterior predictive distribution is plotted by visualizing samples from the
        posterio predictive distribution.
    num_post_pred_samples: int
        Number of poster predictive samples to take when visualizing posterior predictive distribution of fit
    nosetest: bool
        Whether or not to simply compile the model for testing.

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


    with pm.Model() as marginal_gp_model:
        # prior on length scale
        l = pm.Gamma('l', 1, 0.5)

        # prior on amplitude term
        a = pm.HalfNormal('a', sigma=1.0)

        # Specify the mean and covariance functions
        cov_func = pm.gp.cov.ExpQuad(1, ls=l)
        sig_mu = pm.HalfCauchy('sigma', 1)
        mu = pm.Normal('mu', y.mean(), sig_mu)
        mean_func = pm.gp.mean.Constant(mu)

        # Specify the GP.
        gp = pm.gp.Marginal(cov_func=a*cov_func, mean_func=mean_func)

        # set the marginal likelihood based on the training data
        # and give it the noise level
        y_ = gp.marginal_likelihood("y", X=X, y=y, noise=sigma)

    # not a test, proceed with sampling
    if not nosetest:

        # perform hyperparameter sampling (this trains our model)
        with marginal_gp_model:
            traces = pm.sample()

        # plot posterior
        az.plot_trace(traces)
        plt.show()

        # predictions
        range_x = np.max(X) + num_forecast_years - np.min(X)
        Xnew = np.linspace(np.min(X) - 0.2*range_x, np.max(X) + num_forecast_years + 0.2*range_x, 100)[:, None]
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
