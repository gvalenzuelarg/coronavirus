"""Tools to visualize the JHU CSSE COVID-19 Data and the forecasts made
with it using the model module.
"""

import numpy as np
import pandas as pd
from babel.dates import format_date
from babel.numbers import format_decimal

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import seaborn as sns

from modules import processing

# Seaborn styling options
sns.set_style('darkgrid')
sns.set_context('paper')
sns.set_palette('muted')
palette = sns.xkcd_palette(['denim blue','pale red'])
blue, red = sns.xkcd_palette(['denim blue','pale red'])

# For localized formatting
locale = 'de_DE'

# Dates
today = pd.to_datetime('today').normalize()
yesterday = today - pd.Timedelta(1,'D')

footnote = 'Updated on {}. JHU CSSE COVID-19 Data: https://github.com/CSSEGISandData/COVID-19.'.format(format_date(today, locale=locale))

def cases(cases, cases_forecast=pd.DataFrame()):
    fig, ax = plt.subplots()
    if cases_forecast.empty == False:
        sns.lineplot(data=cases_forecast, dashes=False, legend=False)
        for i in np.arange(len(cases_forecast.columns)):
            ax.lines[i].set_linestyle('--')
    sns.lineplot(data=cases, dashes=False)
    ax.set_title('COVID-19 cases')
    ax.set_xlabel(None)
    ax.set_ylabel('Cases')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y,p: format_decimal(y, locale=locale)))
    plt.xticks(rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.67))
    plt.figtext(0.5, -0.03, footnote, fontsize=6, ha='center')
    plt.show()
    return fig, ax

def cases_per_million(cases,  population, cases_forecast=pd.DataFrame()):
    cases = cases / population * 1000000
    cases_forecast = cases_forecast / population * 1000000
    fig, ax = plt.subplots()
    if cases_forecast.empty == False:
        sns.lineplot(data=cases_forecast, dashes=False, legend=False)
        for i in np.arange(len(cases_forecast.columns)):
            ax.lines[i].set_linestyle('--')
    sns.lineplot(data=cases, dashes=False)
    ax.set_title('COVID-19 cases per million inhabitants')
    ax.set_xlabel(None)
    ax.set_ylabel('Cases per million inhabitants')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, p : format_decimal(y, locale=locale)))
    plt.xticks(rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.67))
    plt.figtext(0.5, -0.03, footnote, fontsize=6, ha='center')
    plt.show()
    return fig, ax

def deaths(deaths, deaths_forecast=pd.DataFrame()):
    fig, ax = plt.subplots()
    if deaths_forecast.empty == False:
        sns.lineplot(data=deaths_forecast, dashes=False, legend=False)
        for i in np.arange(len(deaths_forecast.columns)):
            ax.lines[i].set_linestyle('--')
    sns.lineplot(data=deaths, dashes=False)
    ax.set_title('COVID-19 deaths')
    ax.set_xlabel(None)
    ax.set_ylabel('Deaths')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, p : format_decimal(y, locale=locale)))
    plt.xticks(rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.67))
    plt.figtext(0.5, -0.03, footnote, fontsize=6, ha='center')
    plt.show()
    return fig, ax

def deaths_per_million(deaths, population, deaths_forecast=pd.DataFrame()):
    deaths = deaths / population * 1000000
    deaths_forecast = deaths_forecast / population * 1000000
    fig, ax = plt.subplots()
    if deaths_forecast.empty == False: 
        sns.lineplot(data=deaths_forecast, dashes=False, legend=False)
        for i in np.arange(len(deaths_forecast.columns)):
            ax.lines[i].set_linestyle('--')
    sns.lineplot(data=deaths, dashes=False)
    ax.set_title('COVID-19 deaths per million inhabitants')
    ax.set_xlabel(None)
    ax.set_ylabel('Deaths per million inhabitants')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, p : format_decimal(y, locale=locale)))
    plt.xticks(rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.67))
    plt.figtext(0.5, -0.03, footnote, fontsize=6, ha='center')
    plt.show()
    return fig, ax

def daily_cases(cases):
    cases = processing.time_series_delta(cases).rolling(7).mean()
    fig, ax = plt.subplots()
    sns.lineplot(data=cases, dashes=False)
    ax.set_title('Daily COVID-19 cases (7 day rolling average)')
    ax.set_xlabel(None)
    ax.set_ylabel('Daily cases')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, p : format_decimal(y, locale=locale)))
    plt.xticks(rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.67))
    plt.figtext(0.5, -0.03, footnote, fontsize=6, ha='center')
    plt.show()
    return fig, ax

def daily_deaths(deaths):
    deaths = processing.time_series_delta(deaths).rolling(7).mean()
    fig, ax = plt.subplots()
    sns.lineplot(data=deaths, dashes=False)
    ax.set_title('Daily COVID-19 deaths (7 day rolling average)')
    ax.set_xlabel(None)
    ax.set_ylabel('Daily deaths')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, p : format_decimal(y, locale=locale)))
    plt.xticks(rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.67))
    plt.figtext(0.5, -0.03, footnote, fontsize=6, ha='center')
    plt.show()
    return fig, ax

def daily_cases_per_million(cases, population):
    cases = processing.time_series_delta(cases).rolling(7).mean() / population * 1000000
    fig, ax = plt.subplots()
    sns.lineplot(data=cases, dashes=False)
    ax.set_title('Daily COVID-19 cases per million inhabitants (7 day rolling average)')
    ax.set_xlabel(None)
    ax.set_ylabel('Daily cases per million inhabitants')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, p : format_decimal(y, locale=locale)))
    plt.xticks(rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.67))
    plt.figtext(0.5, -0.03, footnote, fontsize=6, ha='center')
    plt.show()
    return fig, ax

def daily_deaths_per_million(deaths, population):
    deaths = processing.time_series_delta(deaths).rolling(7).mean() / population * 1000000
    fig, ax = plt.subplots()
    sns.lineplot(data=deaths, dashes=False)
    ax.set_title('Daily COVID-19 deaths per million inhabitants (7 day rolling average)')
    ax.set_xlabel(None)
    ax.set_ylabel('Daily deaths per million inhabitants')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, p : format_decimal(y, locale=locale)))
    plt.xticks(rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.67))
    plt.figtext(0.5, -0.03, footnote, fontsize=6, ha='center')
    plt.show()
    return fig, ax

def cases_by_days(cases):
    cases = processing.align_from_first_ocurrence(cases)
    fig, ax = plt.subplots()
    sns.lineplot(data=cases, dashes=False)
    ax.set_title('COVID-19 cases by days since the first ocurrence')
    ax.set_xlabel('Days since the first case')
    ax.set_ylabel('Cases')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, p : format_decimal(y, locale=locale)))
    ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.67))
    plt.figtext(0.5, -0.01, footnote, fontsize=6, ha='center')
    plt.show()
    return fig, ax

def deaths_by_days(deaths):
    deaths = processing.align_from_first_ocurrence(deaths)
    fig, ax = plt.subplots()
    sns.lineplot(data=deaths, dashes=False)
    ax.set_title('COVID-19 deaths by days since the first ocurrence')
    ax.set_xlabel('Days since the first death')
    ax.set_ylabel('Deaths')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, p : format_decimal(y, locale=locale)))
    ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.67))
    plt.figtext(0.5, -0.01, footnote, fontsize=6, ha='center')
    plt.show()
    return fig, ax

def cases_by_days_per_million(cases, population):
    cases = processing.align_from_first_ocurrence(cases)
    cases = cases / population * 1000000
    fig, ax = plt.subplots()
    sns.lineplot(data=cases, dashes=False)
    ax.set_title('COVID-19 cases by days since the first ocurrence per million inhabitants')
    ax.set_xlabel('Days since the first case')
    ax.set_ylabel('Cases per million inhabitants')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, p : format_decimal(y, locale=locale)))
    ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.67))
    plt.figtext(0.5, -0.01, footnote, fontsize=6, ha='center')
    plt.show()
    return fig, ax

def deaths_by_days_per_million(deaths, population):
    deaths = processing.align_from_first_ocurrence(deaths)
    deaths = deaths / population * 1000000
    fig, ax = plt.subplots()
    sns.lineplot(data=deaths, dashes=False)
    ax.set_title('COVID-19 deaths by days since the first ocurrence per million inhabitants')
    ax.set_xlabel('Days since the first death')
    ax.set_ylabel('Deaths per million inhabitants')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, p : format_decimal(y, locale=locale)))
    ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.67))
    plt.figtext(0.5, -0.01, footnote, fontsize=6, ha='center')
    plt.show()
    return fig, ax

def country_situation_with_forecast(cases, deaths, cases_forecast, deaths_forecast):
    country = cases.name
    days = cases_forecast.shape[0]
    fig, axs = plt.subplots(2, 2 , figsize=(12, 6), sharex='col')
    fig.subplots_adjust(hspace=0.05)
    # Cases
    sns.lineplot(
        data=cases, color=blue, dashes=False,
        ax=axs[0,0], label='Confirmed cases')
    sns.lineplot(
        data=cases_forecast['yhat'], color=blue,
        ax=axs[0,0], label='Projected cases')
    axs[0,0].lines[1].set_linestyle('--')
    axs[0,0].fill_between(
        cases_forecast.index, cases_forecast['yhat_lower'],
        cases_forecast['yhat_upper'], color=blue, alpha=0.1,
        label='95% confidence interval')
    axs[0,0].legend(loc='upper left')
    axs[0,0].yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, p : format_decimal(y,locale=locale)))
    axs[0,0].set_ylabel('Cases')
    # Daily cases
    sns.lineplot(
        data=processing.time_series_delta(cases).rolling(7).mean(),
        color=blue, dashes=False, ax=axs[0,1])
    axs[0,1].yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, p : format_decimal(y,locale=locale)))
    axs[0,1].set_ylabel('Daily cases')
    # Deaths
    sns.lineplot(
        data=deaths, color=red, dashes=False,
        ax=axs[1,0], label='Confirmed deaths')
    sns.lineplot(
        data=deaths_forecast['yhat'], color=red,
        ax=axs[1,0], label='Projected deaths')
    axs[1,0].lines[1].set_linestyle('--')
    axs[1,0].fill_between(
        deaths_forecast.index, deaths_forecast['yhat_lower'],
        deaths_forecast['yhat_upper'], color=red, alpha=0.1,
        label='95% confidence interval')
    axs[1,0].legend(loc='upper left')
    axs[1,0].yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, p : format_decimal(y,locale=locale)))
    axs[1,0].set_ylabel('Deaths')
    axs[1,0].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.setp(axs[1,0].get_xticklabels(), rotation=45)
    axs[1,0].set_xlabel(None)
    # Daily deaths
    sns.lineplot(
        data=processing.time_series_delta(deaths).rolling(7).mean(),
        color=red, dashes=False, ax=axs[1,1])
    axs[1,1].yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, p : format_decimal(y,locale=locale)))
    axs[1,1].set_ylabel('Daily deaths')
    axs[1,1].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.setp(axs[1,1].get_xticklabels(), rotation=45)
    axs[1,1].set_xlabel(None)
    fig.suptitle('{}: COVID-19 situation and {} day forecast'.format(country, days))
    plt.figtext(0.5, -0.01, footnote, fontsize=6, ha='center')
    plt.show()
    return fig, axs

def countries_situation_overview(cases, deaths):
    countries = list(cases.columns)
    fig, axs = plt.subplots(3, 3 ,figsize=(13, 8))
    fig.subplots_adjust(hspace=0.6, wspace=.3)
    axs = axs.flatten()
    #Creates subplots in a loop
    for i in range(len(countries)):
        #Create country's dataframe of cases + deaths
        df = pd.concat([cases[countries[i]],deaths[countries[i]]],axis=1)
        df.columns = ['Cases', 'Deaths']
        #Calculates country's mortality rate
        mortality= np.round(
            df.iloc[-1,:]['Deaths'] / df.iloc[-1,:]['Cases'] * 100, 1)
        #Country's subplot
        sns.lineplot(
            data=df, palette=palette, dashes=False, legend=False, ax=axs[i])
        axs[i].set(yscale='log')
        axs[i].set_title(
            '{}. Mortality rate: {}%'.format(
                countries[i],format_decimal(mortality,locale=locale)))
        axs[i].set_xlabel(None)
        axs[i].yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda y, p : format_decimal(y, locale=locale)))
        axs[i].set_xticklabels(cases.index, rotation=45)
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    fig.suptitle('COVID-19 situation overview')
    fig.legend(df.columns, bbox_to_anchor=(0.9, 0.98), loc='upper right')
    plt.figtext(0.5, 0.03, footnote, fontsize=6, ha='center')
    plt.show()
    return fig, axs

