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

# Models
    
# Dictionary of models per country
models_cases={}
models_deaths={}

# Models intiation and training
for country in countries:
    # Models for cases
    print('{}: Cases model training...'.format(country))
    if params_cases[country]['growth'] == 'logistic':
        models_cases[country] = model.init_train(
            series=cases[country].dropna(),
            hyperparams=params_cases[country],
            cap=model.growth_limit(
                cases[country][start_cases[country] : ].dropna()),
            outliers=outliers_cases[country]
        )
    else:
        models_cases[country] = model.init_train(
            series=cases[country][start_cases[country] : ].dropna(),
            hyperparams=params_cases[country],
            outliers=outliers_cases[country]
        )
    # Models for deaths
    print('{}: Deaths model training...'.format(country))
    if params_deaths[country]['growth'] == 'logistic':
        models_deaths[country] = model.init_train(
            series=deaths[country].dropna(),
            hyperparams=params_deaths[country],
            cap=model.growth_limit(
                deaths[country][start_deaths[country] : ].dropna()),
            outliers=outliers_deaths[country]
        )
    else:
        models_deaths[country] = model.init_train(
            series=deaths[country][start_deaths[country] : ].dropna(),
            hyperparams=params_deaths[country],
            outliers=outliers_deaths[country]
        )

# Print and saves today's report
print('Generating COVID-19 daily situation report...')
# List of text lines for saving to the report file
text_list = []
text_list.append('\t\tCOVID-19 daily situation report')
text_list.append('')
for country in countries:
    country_report = model.report(
        models_cases[country], models_deaths[country], country)
    text_list += country_report
#Report file
with open('output/report.txt', 'w') as report:
    report.write('\n'.join(text_list))

# Dataframes with 10 week predictions
cases_list = []
deaths_list = []
for country in countries:
    print('{}: Calculating 10 week forecast...'.format(country))
    df_cases = model.predict(models_cases[country], 70)
    cases_list.append(df_cases)
    df_deaths = model.predict(models_deaths[country], 70)
    deaths_list.append(df_deaths)
cases_forecast = pd.concat(cases_list, keys=countries, axis=1)
deaths_forecast = pd.concat(deaths_list, keys=countries, axis=1)

# Graphs

# Cases, deaths Dataframe for selected countries
cases_data = cases.loc['2020-03-01':, countries]
deaths_data = deaths.loc['2020-03-01':, countries]

# COVID-19 situation overview graph
fig, _ = graph.countries_situation_overview(cases_data, deaths_data)
fig.savefig('output/situation_overview.png', dpi=300, bbox_inches='tight')

# Population for all countries but the World
population = population_2019.reindex(countries[1 : ])

# Cases DataFrames for all countries but the World
data = cases.loc['2020-03-01':, countries[1 : ]]
forecast = cases_forecast.loc[:, (countries[1 : ], 'yhat')]
forecast.columns = forecast.columns.droplevel(1)

# Cases graph
fig, _ = graph.cases(data, forecast)
fig.savefig('output/cases.png', dpi=300, bbox_inches='tight')

# Cases per million graph
fig, _ = graph.cases_per_million(data, population, forecast)
fig.savefig('output/cases_per_million.png', dpi=300, bbox_inches='tight')

# Daily cases graph
fig, _ = graph.daily_cases(data)
fig.savefig('output/cases_daily.png', dpi=300, bbox_inches='tight')

# Daily cases per million graph
fig, _ = graph.daily_cases_per_million(data, population)
fig.savefig('output/cases_daily_per_million.png', dpi=300, bbox_inches='tight')

# Cases by days since first ocurrence graph
fig, _ = graph.cases_by_days(data)
fig.savefig('output/cases_by_days.png', dpi=300, bbox_inches='tight')

# Cases by days since first ocurrence per million graph
fig, _ = graph.cases_by_days_per_million(data, population)
fig.savefig('output/cases_by_days_per_million.png', dpi=300, bbox_inches='tight')

# Deaths DataFrames for all countries but the World
data = deaths.loc['2020-03-01':, countries[1 : ]]
forecast = deaths_forecast.loc[:, (countries[1 : ], 'yhat')]
forecast.columns = forecast.columns.droplevel(1)

# Deaths graph
fig, _ = graph.deaths(data, forecast)
fig.savefig('output/deaths.png', dpi=300, bbox_inches='tight')

# Deaths per million graph
fig, _ = graph.deaths_per_million(data, population, forecast)
fig.savefig('output/deaths_per_million.png', dpi=300, bbox_inches='tight')

# Daily deaths graph
fig, _ = graph.daily_deaths(data)
fig.savefig('output/deaths_daily.png', dpi=300, bbox_inches='tight')

# Daily deaths per million graph
fig, _ = graph.daily_deaths_per_million(data, population)
fig.savefig('output/deaths_daily_per_million.png', dpi=300, bbox_inches='tight')

# Deaths by days since first ocurrence graph
fig, _ = graph.deaths_by_days(data)
fig.savefig('output/deaths_by_days.png', dpi=300, bbox_inches='tight')

# Deaths by days since first ocurrence per million graph
fig, _ = graph.deaths_by_days_per_million(data, population)
fig.savefig('output/deaths_by_days_per_million.png', dpi=300, bbox_inches='tight')

# Situation overview and forecast per country
for country in countries:
    cases_country = cases.loc['2020-03-01':, country]
    deaths_country = deaths.loc['2020-03-01':, country]
    cases_forecast_country = cases_forecast[country]
    deaths_forecast_country = deaths_forecast[country]
    # Graph
    fig, _ = graph.country_situation_10_week_forecast(
        cases_country, deaths_country,
        cases_forecast_country, deaths_forecast_country)
    fig.savefig('output/{}.png'.format(str.lower(country).replace(' ','_')),
        dpi=300, bbox_inches='tight')