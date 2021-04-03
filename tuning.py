import numpy as np
import pandas as pd

from modules import processing, model, graph
from model_data.parameters import cap_cases, cap_deaths, params_cases, start_cases, outliers_cases, params_deaths, start_deaths, outliers_deaths

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

country = 'Germany'
print(country)

# Cases

_ = graph.cases(cases[country])
_ = graph.daily_cases(cases[country])

# Logistic 

cap = model.growth_limit(cases[country]['2021-2':], lower_bound=True)
print(cap)
print(country)
params = {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 10}
m_cases = model.init_fit(cases[country]['2020-12':], params_cases[country], 6131960)
fsct = model.predict_raw(m_cases, 70)
processing.time_series_delta(fsct['trend']).plot()

train = processing.to_prophet_input(cases[country], cap_cases[country])
params = model.hyperparameter_tuning(train, horizon='60 days', parallel=None)
print(country)

# Linear

train = processing.to_prophet_input(cases[country]['2020-08':])
params = model.hyperparameter_tuning(train, 'linear', '30 days')

# Deaths

_ = graph.deaths(deaths[country])
_ = graph.daily_deaths(deaths[country])

# Logistic 

cap = model.growth_limit(deaths[country]['2021-2':], lower_bound=True)
print(cap)
print(country)
params = {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 10}
m_deaths = model.init_fit(deaths[country]['2020-12':], params_deaths[country], 97058)
fsct = model.predict_raw(m_deaths, 70)
processing.time_series_delta(fsct['trend']).plot()

train = processing.to_prophet_input(deaths[country], cap_deaths[country])
params = model.hyperparameter_tuning(train, horizon='60 days', parallel=None)
print(country)

# Linear

train = processing.to_prophet_input(deaths[country]['2020-08':])
params = model.hyperparameter_tuning(train, 'linear', '30 days')

# Situation overview and forecast per country
print('{}: Calculating 10 week forecast...'.format(country))
cases_forecast_country = model.predict(m_cases, 70)
deaths_forecast_country = model.predict(m_deaths, 70)

cases_country = cases.loc['2020-03':, country]
deaths_country = deaths.loc['2020-03':, country]
# Graph
fig, _ = graph.country_situation_with_forecast(
    cases_country, deaths_country,
    cases_forecast_country, deaths_forecast_country)
