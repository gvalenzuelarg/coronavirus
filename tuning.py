import numpy as np
import pandas as pd

from modules import processing, model, graph
from model_data.parameters import params_cases, start_cases, outliers_cases, params_deaths, start_deaths, outliers_deaths

# Data import and preprocessing

# JHU CSSE COVID-19 Data: https://github.com/CSSEGISandData/COVID-19
url_cases = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
url_deaths = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'

# Dataframe of confirmed cases, deaths
cases = processing.csse_covid_19_time_series_csv_to_df(url_cases)
deaths = processing.csse_covid_19_time_series_csv_to_df(url_deaths)

# Pandas series with world population by country in 2019
population_2019 = processing.population_2019()

# List consisting of the World, the top 6 countries by cases, plus Chile and Germany
countries = list(
    cases.iloc[-1].sort_values(
            ascending=False)[0 : 7].index) + ['Chile', 'Germany']

# Test if the parameters for the countries in the list are available
for country in countries:
    try:
        params_cases[country]
        print(country)
    except:
        print(
            '{}: Parameters missing. A model must be first tuned.'.format(
                country))

country = 'Chile'
print(country)

# Cases

_ = graph.cases(cases[country])
_ = graph.daily_cases(cases[country])

# Logistic 

cap = model.growth_limit(cases[country]['2020-10':])
print(cap)
print(country)
params = {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 10}
m = model.init_fit(cases[country], params, cap)
fsct = model.predict_raw(m, 365)
processing.time_series_delta(fsct['trend']).plot()

train = processing.to_prophet_input(cases[country], cap)
params = model.hyperparameter_tunning(train, horizon='60 days')

# Linear

train = processing.to_prophet_input(cases[country]['2020-08':])
params = model.hyperparameter_tunning(train, 'linear', '30 days')

# Deaths

_ = graph.deaths(deaths[country])
_ = graph.daily_deaths(deaths[country])

# Logistic 

cap = model.growth_limit(deaths[country]['2020-10-20':])
print(cap)
print(country)
params = {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 10}
m = model.init_fit(deaths[country], params, cap)
fsct = model.predict_raw(m, 365)
processing.time_series_delta(fsct['trend']).plot()

train = processing.to_prophet_input(deaths[country], cap)
params = model.hyperparameter_tunning(train, horizon='60 days')

# Linear

train = processing.to_prophet_input(deaths[country]['2020-08':])
params = model.hyperparameter_tunning(train, 'linear', '30 days')