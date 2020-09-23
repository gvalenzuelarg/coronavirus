"""Tools for preprocessing the data from JHU CSSE COVID-19 Data
and the population data from the World Bank.
"""

import numpy as np
import pandas as pd

def csse_covid_19_time_series_csv_to_df(url):
    """From a global csse_covid_19_time_series CSV, creates a pandas 
    DataFrame with a daily datetime index, first column World, and 
    remaining columns countries/regions in alphabetical order.

    Parameters
    ----------
    url: str
        URL to a global csse_covid_19_time_series CSV file.
        Example: 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'

    Returns
    -------
    df: DataFrame
        A DataFrame indexed by days with countries/regions columns.
    """
    df = pd.read_csv(url)
    df = df.groupby('Country/Region').sum()
    df = df.transpose()
    df = df.drop(index=['Lat','Long'])
    df.index = pd.to_datetime(df.index)
    df.index.name ='Date'
    df.columns.name = None
    # Create column 'World'
    world = pd.DataFrame(df.sum(axis=1), columns=['World'])
    df = pd.concat([world, df], axis=1)
    df.rename(columns={
        'Czechia' : 'Czech Republic',
        'Taiwan*' : 'Taiwan',
        'US' : 'USA',
        'Korea, South' : 'South Korea',
        'United Kingdom' : 'UK'}, inplace=True)
    return df

def align_from_first_ocurrence(df):
    """Aligns and reindex the columns of a DataFrame created with csse_covid_19_time_series_csv_to_df
    from their first nonnegative value.

    Parameters
    ----------
    df : DataFrame
        A DataFrame created with csse_covid_19_time_series_csv_to_df.

    Returns
    -------
    df_aligned : DataFrame
        A DataFrame with its columns aligned and reindex by the first nonnegative entry.
    """
    df_aligned = pd.DataFrame()
    countries = list(df.columns.values)
    for country in countries:
        series = df.loc[:, country][df.loc[:, country] > 0]
        series = series.reset_index()
        series = series.drop(columns='Date')
        df_aligned = pd.concat([df_aligned,series], axis=1)
    return df_aligned

def time_series_delta(df):
    """From a DataFrame with columns consisting of time series,
    returns a DataFrame with the changes per unit of time
    in each column.

    Parameters
    ----------
    df : DataFrame
        A DataFrame with a datetime index.

    Returns
    -------
    df_delta : DataFrame
        A Dataframe of differences per unit of time.
    """
    df_delta = df - df.shift(1)
    df_delta = df_delta.dropna()
    return df_delta

def cummulative_continuation(df_delta,initial_value):
    """From a DataFrame with columns consisting of time series with a
    common datetime index, returns column-wise cummulative sums
    starting from an initial value(s).

    Parameters
    ----------
    df_delta : DataFrame
        A DataFrame with time series as columns.
    intial_value : float or list 
        An initial value or a list of initial values (one per column).
    Returns
    -------
    df : DataFrame
        A DataFrame of cummulative values starting from the intial value(s).
    """
    df = df_delta.cumsum(axis=0)
    df = df + initial_value
    return df

def population_2019():
    """Extracts the world population in 2019 from the World Bank
    population CSV file. Country names are matched to the
    JHU CSSE COVID-19 Data designations. 

    Returns
    -------
    population_2019 : DataFrame
        A DataFrame with the world's 2019 population.
    """
    population = pd.read_csv('data/population.csv')
    population_2019 = population.set_index('Country Name')['2019']
    population_2019.rename(index={
        'Bahamas, The' : 'Bahamas',
        'Brunei Darussalam' : 'Brunei',
        'Myanmar' : 'Burma',
        'Congo, Dem. Rep.' : 'Congo (Kinshasa)',
        'Congo, Rep.' : 'Congo (Brazzaville)',
        'Egypt, Arab Rep.' : 'Egypt',
        'Gambia, The' : 'Gambia',
        'Iran, Islamic Rep.' : 'Iran',
        'Kyrgyz Republic' : 'Kyrgyzstan',
        'Lao PDR' : 'Laos',
        'St. Kitts and Nevis' : 'Saint Kitts and Nevis',
        'St. Lucia' : 'Saint Lucia',
        'St. Vincent and the Grenadines' : 'Saint Vincent and the Grenadines',
        'Slovak Republic' : 'Slovakia',
        'Korea, Rep.' : 'South Korea',
        'Syrian Arab Republic' : 'Syria',
        'Venezuela, RB' : 'Venezuela',
        'Yemen, Rep.' : 'Yemen',
        'United States' : 'USA',
        'United Kingdom' : 'UK',
        'Russian Federation' : 'Russia'}, inplace=True)
    population_2019['Diamond Princess'] = 3711
    population_2019['Holy See'] = 801
    population_2019['MS Zaandam'] = 1829
    population_2019['Taiwan'] = 23780452
    population_2019['Western Sahara'] = 567402
    return population_2019

def to_prophet_input(series, cap=None, outliers=[]):
    """Formats a time series as an input to Prophet.

    The input to Prophet is always a dataframe with
    columns ds, y, and cap, a datestamp, the measurements
    to forecast, and a growth limit, respectively. A list
    of outliers to drop is optional.
    
    Parameters
    ----------
    series : Series
        A Series with a datetime index.
    cap : float
        The growth limit for the logistic trend.
    outliers : list
        Datestamps of outliers to be ignored.

    Returns
    -------
    train : DataFrame
        A DataFrame with a datestamp column (ds) 
        and a numeric column (y) and a constant column (cap).
    """
    train = pd.DataFrame({'ds' : series.index,'y' : series.values})
    train.loc[train['ds'].isin(outliers), 'y'] = None
    train['cap'] = cap
    return train