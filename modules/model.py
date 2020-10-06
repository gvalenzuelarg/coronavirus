"""Tools to initialize, train, and tune models for COVID-19 forecasts.
"""

import numpy as np
import pandas as pd
import itertools
from babel.dates import format_date
from babel.numbers import format_decimal
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, explained_variance_score
from scipy.optimize import curve_fit, fsolve
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics

from modules import processing

def logistic_curve(X, a, b, c, l):
    """Logistic function with standard parameters.

    Parameters
    ----------
    X : ndarray
        A 1-dimensional array of x-values.
    a : float
        The logistic growth rate of the curve.
    b : float
        The x-value of the curve's midpoint.
    c : float
        The function's upper asymptote (saturation limit).
    l : float
        The function's lower asymptote (intial value).
    
    Returns
    -------
    y : ndarray
        A 1-dimensional array of y-values corresponding to X.
    """
    y = l + (c - l) / (1+np.exp(-a*(X-b)))
    return y

def growth_limit(series):
    """Estimates the upper limit of a series with logistic growth.

    Parameters
    ----------
    series : Series
        A time Series with logistic growth.

    Returns
    -------
    cap : float
        The upper limit for the series.
    """
    y = series.dropna()
    X = np.arange(len(y))
    parms = curve_fit(logistic_curve, X, y.values,
                     p0=[.3,1,y[-1]], maxfev=100000)
    cap = parms[0][2].astype(int)
    return cap

def init_fit(series, hyperparams, cap=None, outliers=[]):
    """Initializes and fits a Prophet model.

    Change points are set to be inferred from the full series.
    A cap must be provided for logistic growth. A list of timestapms
    for outliers is optional.

    Parameters
    ----------
    series : Series
        A Series with a datetime index.
    hyperparms : dict
        A dictionary containing Prophet hyperparameters.
    cap : float
        The growth limit for the logistic trend.
    outliers : list
        Datestamps of outliers to be ignored.

    Returns
    -------
    model : dict
        A dictionary with the trained Prophet model, the training
        DataFrame, and the logistic growth limit.
    """
    train = processing.to_prophet_input(series, cap=cap, outliers=outliers)
    m = Prophet(**hyperparams,
                daily_seasonality=False,
                yearly_seasonality=False,
                changepoint_range=1)
    m.fit(train)
    model = {'m' : m, 'train' : series, 'cap' : cap}
    return model

def predict(model, periods):
    """A custom predict function for a Prophet model.

    The Prophet predict method is modified to model 
    non-decreasing functions. It also returns the 
    95% confidence interval for each forecast.
    
    Paramaters
    ----------
    model : dict
        A trained Prophet model created with init_fit.
    periods : int
        The number of periods to forecast.
    
    Returns
    -------
    forecast : DataFrame
        A datetime index DataFrame with the confidence interval
        (yhat_lower, yhat_upper), the forcast (yhat), and the
        changes per period (yhat_delta).
    """
    present = model['train'].index[-1]
    future = model['m'].make_future_dataframe(periods=periods)
    future['cap'] = model['cap']
    prophet_output = model['m'].predict(future)
    prophet_output = prophet_output.set_index(prophet_output['ds'])
    forecast = prophet_output.loc[
        present : , ['yhat_lower', 'yhat_upper', 'yhat']].astype(int)
    forecast_delta=np.maximum(0,processing.time_series_delta(forecast))
    yhat_delta = forecast_delta['yhat'].astype(int)
    yhat_delta.rename('yhat_delta', inplace=True)
    error = model['train'][-1] - forecast.iloc[0]['yhat']
    forecast = processing.cummulative_continuation(
        forecast_delta, list(forecast.iloc[0] + error))
    forecast = pd.concat([forecast, yhat_delta], axis=1)
    return forecast

def predict_raw(model, periods):
    """Returns the raw output of a Prophet model.
    
    Paramaters
    ----------
    model : dict
        A trained Prophet model created with init_fit.
    periods : int
        The number of periods to forecast.
    cap : float
        An upper limit for the case of logistic growth.
    
    Returns
    -------
    prophet_output : DataFrame
        The output of m.predict() method of the Prophet class.
    """
    future = model['m'].make_future_dataframe(periods=periods)
    future['cap'] = model['cap']
    prophet_output = model['m'].predict(future)
    return prophet_output

def date_of_peak(prophet_output):
    """Estimates the date of peak logistic growth per period.

    Parameters
    ----------
    prophet_output : DataFrame
        The output DataFrame of a Prophet model.
    
    Returns
    -------
    peak : Timestamp
        A pandas timestamp with the date of the peak growth.
    """
    output_delta = processing.time_series_delta(prophet_output['trend'])
    peak = prophet_output['ds'][np.argmax(output_delta.values)]
    return peak

def date_of_saturation(prophet_output):
    """Estimates the date when the forecasted values of a Prophet
        model reach the saturation limit for a logistic growth.
    Parameters
    ----------
    prophet_output : DataFrame
        The output DataFrame of a Prophet model.
    
    Returns
    -------
    saturation : Timestamp
        A pandas timestamp with the date when saturations occur.
    """
    cap = prophet_output['cap'][0]
    index = np.argwhere(prophet_output['yhat'].values >= cap)[0]
    saturation = prophet_output['ds'][index[0]]
    return saturation

def mortality(cases, deaths, dates=None):
    """Calculates the mortality rate on a specified list dates.

    If no dates are provided, the mortality is calculated on every date.
    Note that the time scope of forecasts for cases and deaths must match.

    Parameters
    ----------
    cases : DataFrame
        A DataFrame of datetime indexed cases.
    deaths : DataFrame
        A DataFrame of datetime indexed deaths.
    dates : list
        A list of dates 'YYYY-MM-DD'.

    Returns
    -------
    mortality : Dataframe
        A Dataframe with the mortality on all or the specified dates.
    """
    if dates == None:   
        dates = cases.index
    mortality = deaths.loc[dates]/cases.loc[dates]*100
    return mortality

def rmse_in_sample(train, prophet_output):
    """Calculates the root square mean error of the in-sample predictions
    of a Prophet model.
    
    Parameters
    ----------
    train : DataFrame 
        A DataFrame used to train a Prophet model.
    prophet_output : DataFrame
        The output DataFrame of a Prophet model trained with trian.

    Returns
    rmse : float
        The RMSE of the in-sample predicted values.
    """
    y = train
    y_pred = prophet_output['yhat'][ : len(train)]
    rmse = mean_squared_error(y, y_pred)**(1/2)
    return rmse

def explained_variance_in_sample(train, prophet_output):
    """Calculates the explained variance of the in-sample predictions
    of a Prophet model.
    
    Parameters
    ----------
    train : DataFrame 
        A DataFrame used to train a Prophet model.
    prophet_output : DataFrame
        The output DataFrame of a Prophet model trained with trian.

    Returns
    explained_variance : float
        The explained variance of the in-sample predicted values.
    """
    y = train
    y_pred = prophet_output['yhat'][ : len(train)]
    explained_variance = explained_variance_score(y, y_pred)
    return explained_variance

def hyperparameter_tunning(train, growth='logistic', horizon='45 days'):
    """Returns optimal hyperparameters for a Prophet model.

    The hyperparameters are tunned using Prophet's cross_validation
    method. Only changepoint_prior_scale and seasonality_prior_scale
    are tunned using a fixed parameter grid and RMSE as performance
    metric.

    Parameters
    ----------
    train : DataFrame 
        A valid DataFrame to train a Prophet model.
    growth : str
        Prophet's model growth type. Logistic by default, can be set
        to linear.
    Returns
    -------
    best_params : dict
        A dictonary with the optimal value of the hyperparameters.
    """
    param_grid = {  
        'changepoint_prior_scale' : [0.05, 0.5, 5],
        'seasonality_prior_scale' : [0.1, 1, 10]}

    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []  # Store the RMSEs for each params here

    # Use cross validation to evaluate all parameters
    for params in all_params:
        m = Prophet(
            growth=growth, **params, changepoint_range=1,
            daily_seasonality=False, yearly_seasonality=False).fit(train)
        df_cv = cross_validation(m, horizon=horizon, parallel=None)
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].values[0])

    # Find the best parameters
    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses
    print(tuning_results)
    best_params = all_params[np.argmin(rmses)]
    print(best_params)

    #Visual inspection
    m = Prophet(
        growth=growth, **best_params, changepoint_range=1,
        daily_seasonality=False, yearly_seasonality=False).fit(train)
    future = m.make_future_dataframe(periods=70)
    future['cap'] = train['cap'][0]
    fcst = m.predict(future)
    fig = m.plot(fcst)
    plt.show()
    return best_params

def report(model_cases, model_deaths, country):
    """A summary of a countries current data, next day predictions
    and long term trends.

    Parameters
    ----------
    model_cases : dict
        A trained cases Prophet model created with init_fit.
    model_deaths : dict
        A trained deaths Prophet model created with init_fit.
    country : str
        The country's name.

    Returns
    -------
    text_list : list
        A list of sentences formatted to be printed as '\n.join(text_list)'
    """
    # For localized formatting
    locale = 'de_DE'

    # Dates
    today = pd.to_datetime('today').normalize()
    yesterday = today - pd.Timedelta(1,'D')

    #Long-term cases predictions
    prophet_output_cases = predict_raw(model_cases, 1460)

    # In-sample deaths predictions
    prophet_output_deaths = predict_raw(model_deaths, 0)

    # Variables numerical data
    cases_yesterday = model_cases['train'][-1]
    cases_today = predict(model_cases, 1).iloc[0]['yhat']
    deaths_yesterday = model_deaths['train'][-1]
    deaths_today = predict(model_deaths, 1).iloc[0]['yhat']
    daily_cases_yesterday = cases_yesterday-model_cases['train'][-2]
    daily_deaths_yesterday = deaths_yesterday-model_deaths['train'][-2]
    daily_cases_today = cases_today-cases_yesterday
    daily_deaths_today = deaths_today-deaths_yesterday
    mortality_yesterday = np.round(
        deaths_yesterday / cases_yesterday * 100, 1)

    # Only valid for logistic growth models
    if model_cases['cap'] != None:
        peak = date_of_peak(prophet_output_cases)
        end = date_of_saturation(prophet_output_cases)
        total_cases = model_cases['cap']
        total_deaths = model_deaths['cap']
        mortality_final = np.round(total_deaths / total_cases * 100, 1)
    
    rmse_cases = np.round(
        rmse_in_sample(model_cases['train'], prophet_output_cases), 2)
    explained_variance_cases = np.round(
        explained_variance_in_sample(model_cases['train'],
        prophet_output_cases), 3)
    rmse_deaths = np.round(
        rmse_in_sample(model_deaths['train'], prophet_output_deaths), 2)
    explained_variance_deaths = np.round(
        explained_variance_in_sample(model_deaths['train'],
        prophet_output_deaths), 3)
    
    # List of sentences to print
    text_list = []
    text_list.append('{}:'.format(country))
    text_list.append('')
    text_list.append('\tData on {}:'.format(format_date(yesterday, locale=locale)))
    text_list.append('')
    text_list.append('\t\tCases: {} ({}).'.format(
        format_decimal(cases_yesterday, locale=locale),
        format_decimal(daily_cases_yesterday, locale=locale, format='+#,###;-#')))
    text_list.append('\t\tDeaths: {} ({}).'.format(
        format_decimal(deaths_yesterday, locale=locale),
        format_decimal(daily_deaths_yesterday, locale=locale, format='+#,###;-#')))
    text_list.append('\t\tMortality rate: {}%.'.format(
        format_decimal(mortality_yesterday, locale=locale)))
    text_list.append('')
    text_list.append('\tToday\'s predictions:')
    text_list.append('')
    text_list.append('\t\tCases: {} ({}).'.format(
        format_decimal(cases_today, locale=locale),
        format_decimal(daily_cases_today, locale=locale, format='+#,###;-#')))
    text_list.append('\t\tDeaths: {} ({}).'.format(
        format_decimal(deaths_today, locale=locale),
        format_decimal(daily_deaths_today, locale=locale, format='+#,###;-#')))
    # Only valid for logistic growth models
    if model_cases['cap'] != None:
        text_list.append('')
        text_list.append('\tExpected parameters:')
        text_list.append('')
        text_list.append('\t\tMaximum daily infections on {}.'.format(
        format_date(peak, locale=locale)))
        text_list.append('\t\tCurrent pandemic\'s wave to end on {}.'.format(
            format_date(end, locale=locale)))
        text_list.append('\t\tTotal number of cases: {}.'.format(
            format_decimal(total_cases, locale=locale)))
        text_list.append('\t\tTotal number of deaths: {}.'.format(
            format_decimal(total_deaths, locale=locale)))
        text_list.append('\t\tFinal mortality rate: {}%.'.format(
            format_decimal(mortality_final, locale=locale)))
    text_list.append('')
    text_list.append('\tCases model\'s metrics:')
    text_list.append('')
    text_list.append('\t\tRoot Mean Squared Error: {}.'.format(
        format_decimal(rmse_cases, locale=locale)))
    text_list.append('\t\tExplained Variance: {}.'.format(
        format_decimal(explained_variance_cases, locale=locale)))
    text_list.append('')
    text_list.append('\tDeaths model\'s metrics:')
    text_list.append('')
    text_list.append('\t\tRoot Mean Squared Error: {}.'.format(
        format_decimal(rmse_deaths, locale=locale)))
    text_list.append('\t\tExplained Variance: {}.'.format(
        format_decimal(explained_variance_deaths, locale=locale)))
    text_list.append('')
    print('\n'.join(text_list))
    return text_list