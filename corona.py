import numpy as np
import pandas as pd
from babel.dates import format_date
from babel.numbers import format_decimal

from modules import processing

# Variables with localized dates
locale = 'de_DE'
today = pd.to_datetime('today').normalize()
today_str = format_date(today, locale=locale)
yesterday = today - pd.Timedelta(1,'D')
yesterday_str = format_date(yesterday, locale=locale)

# JHU CSSE COVID-19 Data: https://github.com/CSSEGISandData/COVID-19
url_cases = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
url_deaths = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'

# Dataframe of confirmed cases
cases = processing.csse_covid_19_time_series_csv_to_df(url_cases)

# Dataframe of daily cases
cases_daily = processing.time_series_delta(cases)

# Dataframe of deaths
deaths = processing.csse_covid_19_time_series_csv_to_df(url_deaths)

# Dataframe of daily deaths
deaths_daily = processing.time_series_delta(deaths)

# Dataframe of confirmed cases reindexed to days since the first instance
cases_from_first_ocurrence = processing.align_from_first_ocurrence(cases)

# Dataframe of deaths reindexed to days since the first instance
deaths_from_first_ocurrence = processing.align_from_first_ocurrence(deaths)

# Pandas series with world population by country in 2019
population_2019 = processing.population_2019()

# Creates list consisting of the World, the top 6 countries by confirmed cases, plus Chile and Germany
countries = list(cases.iloc[-1].sort_values(ascending=False)[0:7].index)+['Chile','Germany']

#Projection of cases for selected countries
from sklearn.metrics import mean_squared_error, explained_variance_score
from fbprophet import Prophet
from scipy.optimize import curve_fit
from scipy.optimize import fsolve

#Logistic Curve model
def logistic_curve(X,a,b,c):
    return c/(1+np.exp(-a*(X-b)))
    
#list of dataframes with confirmed cases per country
dfs_c=[]
#list of dataframes with deaths per country
dfs_d=[]
#Dictionary of automodels confirmed cases
models_c={}
#Dictionary of automodels deaths
models_d={}
#Logistic model parameters confirmed cases
parm_c={}
#Logistic model parameters deaths
parm_d={}
#Countries confirmed cases trend points
t_c={'World':None,
    'USA':'2020-05',
    'Brazil':'2020-06-15',
    'India':'2020-06',
    'Russia':'2020-06-21',
    'South Africa':None,
    'Peru':'2020-04-10',
    'Colombia':None,
    'Mexico':'2020-07-09',
    'Chile':'2020-07-15',
    'Germany':'2020-05-15'}
#Countries deaths trend points
t_d={'World':'2020-04-15',
    'USA':'2020-05',
    'Brazil':'2020-07-15',
    'India':'2020-06-20',
    'Russia':'2020-06-15',
    'South Africa':None,
    'Peru':'2020-05-29',
    'Colombia':None,
    'Mexico':'2020-07-10',
    'Chile':'2020-07-18',
    'Germany':'2020-05-15'}
#Countries cases hyperparameters
hyper_c={'World':{'growth':'logistic',
                'changepoint_prior_scale':5,
                'seasonality_prior_scale':0.1},
        'USA':{'growth':'logistic',
                'changepoint_prior_scale':0.5,
                'seasonality_prior_scale':1},
        'Brazil':{'growth':'logistic',
                'changepoint_prior_scale':0.05,
                'seasonality_prior_scale':1},
        'India':{'growth':'logistic',
                'changepoint_prior_scale':5,
                'seasonality_prior_scale':1},
        'Russia':{'growth':'logistic',
                'changepoint_prior_scale':0.5,
                'seasonality_prior_scale':0.1},
        'South Africa':{'growth':'logistic',
                'changepoint_prior_scale':5,
                'seasonality_prior_scale':0.1},
        'Peru':{'growth':'logistic',
                'changepoint_prior_scale':5,
                'seasonality_prior_scale':0.1},
        'Colombia':{'growth':'logistic',
                'changepoint_prior_scale':0.05,
                'seasonality_prior_scale':10},
        'Mexico':{'growth':'logistic',
                'changepoint_prior_scale':5,
                'seasonality_prior_scale':1},
        'Chile':{'growth':'logistic',
                'changepoint_prior_scale':5,
                'seasonality_prior_scale':1},
        'Germany':{'growth':'linear',
                'changepoint_prior_scale':0.05,
                'seasonality_prior_scale':10},
        }
#Countries deaths hyperparameters
hyper_d={'World':{'growth':'logistic',
                'changepoint_prior_scale':5,
                'seasonality_prior_scale':10},
        'USA':{'growth':'logistic',
                'changepoint_prior_scale':0.5,
                'seasonality_prior_scale':10},
        'Brazil':{'growth':'logistic',
                'changepoint_prior_scale':5,
                'seasonality_prior_scale':0.1},
        'India':{'growth':'logistic',
                'changepoint_prior_scale':5,
                'seasonality_prior_scale':1},
        'Russia':{'growth':'logistic',
                'changepoint_prior_scale':5,
                'seasonality_prior_scale':0.1},
        'South Africa':{'growth':'logistic',
                'changepoint_prior_scale':5,
                'seasonality_prior_scale':10},
        'Peru':{'growth':'logistic',
                'changepoint_prior_scale':0.5,
                'seasonality_prior_scale':10},
        'Colombia':{'growth':'logistic',
                'changepoint_prior_scale':0.05,
                'seasonality_prior_scale':1},
        'Mexico':{'growth':'logistic',
                'changepoint_prior_scale':5,
                'seasonality_prior_scale':10},
        'Chile':{'growth':'logistic',
                'changepoint_prior_scale':5,
                'seasonality_prior_scale':0.1},
        'Germany':{'growth':'linear',
                'changepoint_prior_scale':0.05,
                'seasonality_prior_scale':10}
                }
#Countries cases outliers
outliers_c={'World':[],
            'USA':[],
            'Brazil':['2020-06-19'],
            'India':[],
            'Russia':[],
            'South Africa':[],
            'Peru':['2020-06-03','2020-06-12','2020-07-25','2020-07-26','2020-07-27','2020-07-30','2020-08-01','2020-08-02','2020-08-08','2020-08-09','2020-08-12','2020-08-14','2020-08-15','2020-09-03','2020-09-04'],
            'Colombia':[],
            'Mexico':[],
            'Chile':['2020-05-30','2020-06-06'],
            'Germany':[]}
#Countries deaths outliers
outliers_d={'World':['2020-08-14'],
            'USA':[],
            'Brazil':[],
            'India':['2020-06-16'],
            'Russia':[],
            'South Africa':['2020-07-22'],
            'Peru':['2020-07-23','2020-08-14'],
            'Colombia':[],
            'Mexico':['2020-06-04'],
            'Chile':['2020-06-08','2020-07-17'],
            'Germany':[]}
#Countries pandemic status
status={'World':'second wave',
        'USA':'second wave',
        'Brazil':'first wave',
        'India':'first wave',
        'Russia':'first wave',
        'South Africa':'first wave',
        'Peru':'first wave',
        'Colombia':'first wave',
        'Mexico':'first wave',
        'Chile':'first wave',
        'Germany':'saturation point'}
#list of text lines to be printed/saved to the report file
textlist=[]
#Title of the report
textlist.append('\t\tDaily Report on COVID-19')
textlist.append('')
#Model calculations
for country in countries:
    #Confirmed cases prediction
    #Cap estimation
    y=cases[country][t_c[country]:].dropna()
    X=np.arange(len(y))
    parm_c[country] = curve_fit(logistic_curve,X,y.values,p0=[.3,1,y[-1]],maxfev = 100000)
    a, b, c = parm_c[country][0]
    cap=c
    #Metrics logistic model
    RMSE_c_log=mean_squared_error(y,logistic_curve(X,a,b,c))**(1/2)
    explained_variance_c_log=explained_variance_score(y,logistic_curve(X,a,b,c))
    #Prapare data
    y=cases[country].dropna()
    X=np.arange(len(y))
    #Calculate date of pick daily infections
    #y_year=logistic_curve(np.arange(365),a,b,c)
    #dy=np.diff(y_year)
    #arg_max=np.argmax(dy)
    first_day=y.index[0]
    max_daily_date=first_day+pd.Timedelta(int(b),'D')
    #Calculate end of pandemic
    sol = int(fsolve(lambda x : logistic_curve(x,a,b,c) - int(c),b))
    ending_date=first_day+pd.Timedelta(sol,'D')
    #Prophet model
    #Choice of trainset depending on pandemic status
    if status[country]=='saturation point':
        train=pd.DataFrame({'ds':y[t_c[country]:].index,'y':y[t_c[country]:].values})
    else:
        train=pd.DataFrame({'ds':y.index,'y':y.values})
    train.loc[train['ds'].isin(outliers_c[country]), 'y'] = None
    train['cap'] = cap
    print('Cases model for {}...'.format(country))
    m = Prophet(**hyper_c[country], daily_seasonality=False, yearly_seasonality=False)
    m.fit(train)
    future = m.make_future_dataframe(periods=1460)
    future['cap'] = cap
    models_c[country] = m.predict(future)
    #5 weeks predictions
    forecast_daily=np.maximum(0,processing.time_series_delta(models_c[country][['yhat']][-1461:-(1460-35)]))
    forecast_daily=pd.DataFrame(forecast_daily.values,index=pd.date_range(today, periods=(35), freq='D'))
    forecast=processing.cummulative_continuation(forecast_daily,cases[country][-1])
    forecast=pd.DataFrame(forecast,index=pd.date_range(today, periods=(35), freq='D'))
    predictions_in_sample=pd.DataFrame(models_c[country]['yhat'][:len(train)].values,columns=['Prediction'],index=models_c[country]['ds'][:len(train)])
    #country's confirmed cases dataframe
    df_c=pd.concat([cases[country][cases[country]>0],forecast,forecast_daily],axis=1)
    df_c.columns=['Data','Prediction','Daily']
    df_c.index.name='Date'
    #Saves dataframe to list
    dfs_c.append(df_c)
    #Metrics Prophet model
    if status[country]=='saturation point':
        train=pd.DataFrame({'ds':y[t_c[country]:].index,'y':y[t_c[country]:].values})
    else:
        train=pd.DataFrame({'ds':y.index,'y':y.values})
    RMSE_c_arima=np.round(mean_squared_error(train['y'],predictions_in_sample)**(1/2),2)
    explained_variance_c_arima=np.round(explained_variance_score(train['y'],predictions_in_sample),3)
    #Total confirmed cases
    total_cases=c
    #Calculates country's mortality rate
    mortality= round(deaths.iloc[-1,:][country]/cases.iloc[-1,:][country]*100,1)
    #Metrics avg.
    RMSE_c=np.round(np.mean([RMSE_c_arima,RMSE_c_log]),2)
    explained_variance_c=np.round(np.mean([explained_variance_c_arima,explained_variance_c_log]),3)

    #Deaths prediction
    #Cap estimation
    y=deaths[country][t_d[country]:].dropna()
    X=np.arange(len(y))
    parm_d[country] = curve_fit(logistic_curve,X,y.values,p0=[.3,1,y[-1]],maxfev = 100000)
    a, b, c = parm_d[country][0]
    cap=c
    #Metrics logistic model
    RMSE_d_log=mean_squared_error(y,logistic_curve(X,a,b,c))**(1/2)
    explained_variance_d_log=explained_variance_score(y,logistic_curve(X,a,b,c))
    #Prapare data
    y=deaths[country].dropna()
    X=np.arange(len(y))
    #Prophet model
    #Choice of trainset depending on pandemic status
    if status[country]=='saturation point':
        train=pd.DataFrame({'ds':y[t_d[country]:].index,'y':y[t_d[country]:].values})
    else:
        train=pd.DataFrame({'ds':y.index,'y':y.values})
    train.loc[train['ds'].isin(outliers_d[country]), 'y'] = None
    train['cap'] = cap
    print('Deaths model for {}...'.format(country))
    m = Prophet(**hyper_d[country], daily_seasonality=False, yearly_seasonality=False)
    m.fit(train)
    future = m.make_future_dataframe(periods=1460)
    future['cap'] = cap
    models_d[country] = m.predict(future)
    #5 weeks predictions
    forecast_daily=np.maximum(0,processing.time_series_delta(models_d[country][['yhat']][-1461:-(1460-35)]))
    forecast_daily=pd.DataFrame(forecast_daily.values,index=pd.date_range(today, periods=(35), freq='D'))
    forecast=processing.cummulative_continuation(forecast_daily,deaths[country][-1])
    forecast=pd.DataFrame(forecast,index=pd.date_range(today, periods=(35), freq='D'))
    predictions_in_sample=pd.DataFrame(models_d[country]['yhat'][:len(train)].values,columns=['Prediction'],index=models_d[country]['ds'][:len(train)])
    #country's deaths dataframe
    df_d=pd.concat([deaths[country][deaths[country]>0],forecast,forecast_daily],axis=1)
    df_d.columns=['Data','Prediction','Daily']
    df_d.index.name='Date'
    #Saves dataframe to list
    dfs_d.append(df_d)
    #Metrics Prophet model
    if status[country]=='saturation point':
        train=pd.DataFrame({'ds':y[t_d[country]:].index,'y':y[t_d[country]:].values})
    else:
        train=pd.DataFrame({'ds':y.index,'y':y.values})
    RMSE_d_arima=mean_squared_error(train['y'],predictions_in_sample)**(1/2)
    explained_variance_d_arima=explained_variance_score(train['y'],predictions_in_sample)
    #Total deaths
    total_deaths=c
    #Metrics avg.
    RMSE_d=np.round(np.mean([RMSE_d_arima,RMSE_d_log]),2)
    explained_variance_d=np.round(np.mean([explained_variance_d_arima,explained_variance_d_log]),3)

    #Create a list of lines for the report
    textlist.append('{}:'.format(country))
    textlist.append('')
    textlist.append('\tData on {}:'.format(yesterday_str))
    textlist.append('')
    textlist.append('\t\tConfirmed cases: {} ({}).'.format(format_decimal(cases.iloc[-1,:][country],locale=locale),format_decimal((cases.iloc[-1,:][country]-cases.iloc[-2,:][country]).astype(int),locale=locale,format='+#,###;-#')))
    textlist.append('\t\tDeaths: {} ({}).'.format(format_decimal(deaths.iloc[-1,:][country],locale=locale),format_decimal((deaths.iloc[-1,:][country]-deaths.iloc[-2,:][country]).astype(int),locale=locale,format='+#,###;-#')))
    textlist.append('\t\tMortality rate: {}%.'.format(format_decimal(mortality,locale=locale)))
    textlist.append('')
    textlist.append('\tToday\'s predictions:')
    textlist.append('')
    textlist.append('\t\tConfirmed cases: {} ({}).'.format(format_decimal(df_c.loc[today,'Prediction'].astype(int),locale=locale),format_decimal((df_c.loc[today,'Prediction']-df_c.loc[yesterday,'Data']).astype(int),locale=locale,format='+#,###;-#')))
    textlist.append('\t\tDeaths: {} ({}).'.format(format_decimal(df_d.loc[today,'Prediction'].astype(int),locale=locale),format_decimal((df_d.loc[today,'Prediction']-df_d.loc[yesterday,'Data']).astype(int),locale=locale,format='+#,###;-#')))
    textlist.append('')
    textlist.append('\tExpected parameters:')
    textlist.append('')
    textlist.append('\t\tMaximum daily infections on {}.'.format(format_date(max_daily_date,locale=locale)))
    textlist.append('\t\tThe pandemic is expected to end on {}.'.format(format_date(ending_date,locale=locale)))
    textlist.append('\t\tTotal number of confirmed cases: {}.'.format(format_decimal(int(total_cases),locale=locale)))
    textlist.append('\t\tTotal number of deaths: {}.'.format(format_decimal(int(total_deaths),locale=locale)))
    textlist.append('\t\tFinal mortality rate: {}%.'.format(format_decimal(round(total_deaths/total_cases*100,1),locale=locale)))
    textlist.append('')
    textlist.append('\tConfirmed cases model\'s metrics:')
    textlist.append('')
    textlist.append('\t\tRoot Mean Squared Error: {}.'.format(format_decimal(RMSE_c,locale=locale)))
    textlist.append('\t\tExplained Variance: {}.'.format(format_decimal(explained_variance_c,locale=locale)))
    textlist.append('')
    textlist.append('\tDeaths model\'s metrics:')
    textlist.append('')
    textlist.append('\t\tRoot Mean Squared Error: {}.'.format(format_decimal(RMSE_d,locale=locale)))
    textlist.append('\t\tExplained Variance: {}.'.format(format_decimal(explained_variance_d,locale=locale)))
    textlist.append('')
#Dataframe with all confirmed cases
model_data_c=pd.concat(dfs_c,keys=countries,axis=1)
#Dataframe with all deaths
model_data_d=pd.concat(dfs_d,keys=countries,axis=1)
#Saves dataframes as cvs
model_data_c.to_csv('output/cases.csv', float_format='%.f')
model_data_d.to_csv('output/deaths.csv', float_format='%.f')
#Creates report file
with open("output/report.txt", "w") as report:
    report.write('\n'.join(textlist))
#Prints report
with open("output/report.txt", "r") as report:
    print(report.read())

#Graphs
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import seaborn as sns
sns.set_style('darkgrid')
sns.set_context('paper')
sns.set_palette('muted')

#Sublist of countries to graph
countries=list(cases.iloc[-1].sort_values(ascending=False)[1:7].index)+['Germany','Chile']

#Confirmed cases graph
#Prepare data
df_c=model_data_c.loc[:,(countries,'Prediction')].dropna().iloc[:35]
df_c.columns=df_c.columns.droplevel(1)
df_c=cases.loc[yesterday:yesterday][countries].append(df_c)
#Graph
fig, ax =plt.subplots()
sns.lineplot(data=df_c,alpha=1,dashes=False,legend=False)
for i in np.arange(8):
    ax.lines[i].set_linestyle('--')
sns.lineplot(data=cases.loc['2020-03-01':,countries],dashes=False)
#ax.set(yscale='log')
ax.set_title('COVID-19 cases')
ax.set_xlabel(None)
ax.set_ylabel('Cases')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,p: format_decimal(y,locale=locale)))
#ax.set_ylim(-10000,700000)
#ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=45)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.legend(loc='center right', bbox_to_anchor=(1.25,0.67))
#plt.savefig('confirmed.pdf', bbox_inches='tight')
footnote='Updated on {}. JHU CSSE COVID-19 Data: https://github.com/CSSEGISandData/COVID-19.'.format(yesterday_str)
plt.figtext(0.5,-0.03,footnote, fontsize=6, ha='center')
fig.savefig('output/cases.png', dpi=300, bbox_inches='tight')
plt.show()

#Confirmed cases graph per 1M people
#Prepare data
df_c=model_data_c.loc[:,(countries,'Prediction')].dropna().iloc[:35]
df_c.columns=df_c.columns.droplevel(1)
df_c=cases.loc[yesterday:yesterday][countries].append(df_c)
df_c=df_c/population_2019.reindex(countries)*1000000
#Graph
fig, ax =plt.subplots()
sns.lineplot(data=df_c,alpha=1,dashes=False,legend=False)
for i in np.arange(8):
    ax.lines[i].set_linestyle('--')
sns.lineplot(data=cases.loc['2020-03-01':,countries]/population_2019.reindex(countries)*1000000,dashes=False)
#ax.set(yscale='log')
ax.set_title('COVID-19 cases per million inhabitants')
ax.set_xlabel(None)
ax.set_ylabel('Cases per million inhabitants')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,p: format_decimal(y,locale=locale)))
#ax.set_ylim(-10000,700000)
#ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=45)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.legend(loc='center right', bbox_to_anchor=(1.25,0.67))
#plt.savefig('confirmed.pdf', bbox_inches='tight')
footnote='Updated on {}. JHU CSSE COVID-19 Data: https://github.com/CSSEGISandData/COVID-19.'.format(yesterday_str)
plt.figtext(0.5,-0.03,footnote, fontsize=6, ha='center')
fig.savefig('output/cases_per_M.png', dpi=300, bbox_inches='tight')
plt.show()

#Daily cases graph
#Prepare data
#df_c=model_data_c.loc[:,(countries,'Daily')].iloc[-35:]
#df_c.columns=df_c.columns.droplevel(1)
#df_c=confirmed_daily.loc['2020-03-01':,countries][-7:].append(df_c).rolling(7).mean().dropna()
#Graph
fig, ax =plt.subplots()
#sns.lineplot(data=df_c,dashes=False, legend=False)
#for i in np.arange(8):
#    ax.lines[i].set_linestyle('--')
sns.lineplot(data=cases_daily.loc['2020-03-01':,countries].rolling(7).mean(),dashes=False)
#ax.set(yscale='log')
ax.set_title('Daily COVID-19 cases (7 day rolling average)')
ax.set_xlabel(None)
ax.set_ylabel('Daily cases')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,p: format_decimal(y,locale=locale)))
#ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=45)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.legend(loc='center right', bbox_to_anchor=(1.25,0.67))
footnote='Updated on {}. JHU CSSE COVID-19 Data: https://github.com/CSSEGISandData/COVID-19.'.format(yesterday_str)
plt.figtext(0.5,-0.03,footnote, fontsize=6, ha='center')
fig.savefig('output/cases_daily.png', dpi=300, bbox_inches='tight')
plt.show()

#Daily cases graph per million people 
#Prepare data
#df_c=model_data_c.loc[:,(countries,'Daily')].iloc[-35:]
#df_c.columns=df_c.columns.droplevel(1)
#df_c=cases_daily.loc['2020-03-01':,countries][-7:].append(df_c).rolling(7).mean().dropna()
#Graph
fig, ax =plt.subplots()
#sns.lineplot(data=df_c,dashes=False, legend=False)
#for i in np.arange(8):
#    ax.lines[i].set_linestyle('--')
df_c=cases_daily.loc['2020-03-01':,countries].rolling(7).mean()/population_2019.reindex(countries)*1000000
sns.lineplot(data=df_c,dashes=False)
#ax.set(yscale='log')
ax.set_title('Daily COVID-19 cases per million inhabitants (7 day rolling average)')
ax.set_xlabel(None)
ax.set_ylabel('Daily cases per million inhabitants')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,p: format_decimal(y,locale=locale)))
#ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=45)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.legend(loc='center right', bbox_to_anchor=(1.25,0.67))
footnote='Updated on {}. JHU CSSE COVID-19 Data: https://github.com/CSSEGISandData/COVID-19.'.format(yesterday_str)
plt.figtext(0.5,-0.03,footnote, fontsize=6, ha='center')
fig.savefig('output/cases_daily_per_M.png', dpi=300, bbox_inches='tight')
plt.show()

#Deaths graph 
#Prepare data
df_d=model_data_d.loc[:,(countries,'Prediction')].dropna().iloc[:35]
df_d.columns=df_d.columns.droplevel(1)
df_d=deaths.loc[yesterday:yesterday][countries].append(df_d)
#Graph
fig, ax =plt.subplots()
#ax.set_ylim(-250,20000)
sns.lineplot(data=df_d,alpha=1,dashes=False,legend=False)
for i in np.arange(8):
    ax.lines[i].set_linestyle('--')
sns.lineplot(data=deaths.loc['2020-03-01':,countries],dashes=False)
#ax.set(yscale='log')
ax.set_title('COVID-19 deaths')
ax.set_xlabel(None)
ax.set_ylabel('Deaths')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,p: format_decimal(y,locale=locale)))
#ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=45)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.legend(loc='center right', bbox_to_anchor=(1.25,0.67))
#plt.savefig('deaths.pdf',bbox_inches='tight')
footnote='Updated on {}. JHU CSSE COVID-19 Data: https://github.com/CSSEGISandData/COVID-19.'.format(yesterday_str)
plt.figtext(0.5,-0.03,footnote, fontsize=6, ha='center')
fig.savefig('output/deaths.png',dpi=300,bbox_inches='tight')
plt.show()

#Deaths graph per million people 
#Prepare data
df_d=model_data_d.loc[:,(countries,'Prediction')].dropna().iloc[:35]
df_d.columns=df_d.columns.droplevel(1)
df_d=deaths.loc[yesterday:yesterday][countries].append(df_d)/population_2019.reindex(countries)*1000000
#Graph
fig, ax =plt.subplots()
#ax.set_ylim(-250,20000)
sns.lineplot(data=df_d,alpha=1,dashes=False,legend=False)
for i in np.arange(8):
    ax.lines[i].set_linestyle('--')
sns.lineplot(data=deaths.loc['2020-03-01':,countries]/population_2019.reindex(countries)*1000000,dashes=False)
#ax.set(yscale='log')
ax.set_title('COVID-19 deaths per million inhabitants')
ax.set_xlabel(None)
ax.set_ylabel('Deaths per million inhabitants')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,p: format_decimal(y,locale=locale)))
#ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=45)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.legend(loc='center right', bbox_to_anchor=(1.25,0.67))
#plt.savefig('deaths.pdf',bbox_inches='tight')
footnote='Updated on {}. JHU CSSE COVID-19 Data: https://github.com/CSSEGISandData/COVID-19.'.format(yesterday_str)
plt.figtext(0.5,-0.03,footnote, fontsize=6, ha='center')
fig.savefig('output/deaths_per_M.png',dpi=300,bbox_inches='tight')
plt.show()

#Daily deaths graph
#Prepare data
#df_d=model_data_d.loc[:,(countries,'Daily')].iloc[-35:]
#df_d.columns=df_d.columns.droplevel(1)
#df_d=deaths_daily.loc['2020-03-01':,countries][-7:].append(df_d).rolling(7).mean().dropna()
#Graph
fig, ax =plt.subplots()
#sns.lineplot(data=df_d,dashes=False, legend=False)
#for i in np.arange(8):
#    ax.lines[i].set_linestyle('--')
sns.lineplot(data=deaths_daily.loc['2020-03-01':,countries].rolling(7).mean(),dashes=False)
#ax.set(yscale='log')
ax.set_title('Daily COVID-19 deaths (7 day rolling average)')
ax.set_xlabel(None)
ax.set_ylabel('Daily deaths')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,p: format_decimal(y,locale=locale)))
#ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=45)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.legend(loc='center right', bbox_to_anchor=(1.25,0.67))
footnote='Updated on {}. JHU CSSE COVID-19 Data: https://github.com/CSSEGISandData/COVID-19.'.format(yesterday_str)
plt.figtext(0.5,-0.03,footnote, fontsize=6, ha='center')
fig.savefig('output/deaths_daily.png', dpi=300, bbox_inches='tight')
plt.show()

#Daily deaths graph per million people
#Prepare data
#df_d=model_data_d.loc[:,(countries,'Daily')].iloc[-35:]
#df_d.columns=df_d.columns.droplevel(1)
#df_d=deaths_daily.loc['2020-03-01':,countries][-7:].append(df_d).rolling(7).mean().dropna()
#Graph
fig, ax =plt.subplots()
#sns.lineplot(data=df_d,dashes=False, legend=False)
#for i in np.arange(8):
#    ax.lines[i].set_linestyle('--')
df_d=deaths_daily.loc['2020-03-01':,countries].rolling(7).mean()/population_2019.reindex(countries)*1000000
sns.lineplot(data=df_d,dashes=False)
#ax.set(yscale='log')
ax.set_title('Daily COVID-19 deaths per million inhabitants (7 day rolling average)')
ax.set_xlabel(None)
ax.set_ylabel('Daily deaths per million inhabitants')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,p: format_decimal(y,locale=locale)))
#ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=45)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.legend(loc='center right', bbox_to_anchor=(1.25,0.67))
footnote='Updated on {}. JHU CSSE COVID-19 Data: https://github.com/CSSEGISandData/COVID-19.'.format(yesterday_str)
plt.figtext(0.5,-0.03,footnote, fontsize=6, ha='center')
fig.savefig('output/deaths_daily_per_M.png', dpi=300, bbox_inches='tight')
plt.show()

#COVID-19 cases by days since the first ocurrence
#Prepare data
train=cases['Chile'][cases['Chile']>0].astype(float)
y_fc,ci=processing.time_series_delta(models_c['Chile'][['yhat']][-1461:-(1460-40)]),processing.time_series_delta(models_c['Chile'][['yhat_lower','yhat_upper']][-1461:-(1460-40)]).dropna()
y_fc,ci=np.maximum(0,y_fc)['yhat'].values,np.maximum(0,ci)[['yhat_lower','yhat_upper']].values
y_fc,ci=processing.cummulative_continuation(y_fc,cases['Chile'][-1]),processing.cummulative_continuation(ci,cases['Chile'][-1])
y_fc=np.concatenate((np.array([train.iloc[-1]]),y_fc))
X_fc=np.arange(len(train)-1,len(train)+40)
#Graph
fig, ax =plt.subplots()
#ax.set_ylim([-10000,800000])
mgrey=(0.4745098039215686, 0.4745098039215686, 0.4745098039215686)
#sns.lineplot(x=X_fc,y=y_fc,color=mgrey,alpha=1)
#ax.fill_between(X_fc[1:], ci[:,0], ci[:,1], color=mgrey, alpha=0.1,label='95% CI')
sns.lineplot(data=cases_from_first_ocurrence.loc[30:,countries]/population_2019.reindex(countries)*1000000,dashes=False)
ax.set_title('COVID-19 cases by days since the first ocurrence')
#ax.set(yscale='log')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,p: format_decimal(y,locale=locale)))
#ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.set_xlabel('Days since the first case')
ax.set_ylabel('Cases')
ax.legend(loc='center right', bbox_to_anchor=(1.25,0.67))
#ax.lines[0].set_linestyle('--')
footnote='Updated on {}. JHU CSSE COVID-19 Data: https://github.com/CSSEGISandData/COVID-19.'.format(yesterday_str)
plt.figtext(0.5,-0.01,footnote, fontsize=6, ha='center')
fig.savefig('output/cases_since_1d.png',dpi=300, bbox_inches='tight')
plt.show()

#COVID-19 deaths by days since the first ocurrence
#Prepare data
train=cases['Chile'][cases['Chile']>0].astype(float)
y_fc,ci=processing.time_series_delta(models_c['Chile'][['yhat']][-1461:-(1460-40)]).dropna(),processing.time_series_delta(models_c['Chile'][['yhat_lower','yhat_upper']][-1461:-(1460-40)])
y_fc,ci=np.maximum(0,y_fc)['yhat'].values,np.maximum(0,ci)[['yhat_lower','yhat_upper']].values
y_fc,ci=processing.cummulative_continuation(y_fc,cases['Chile'][-1]),processing.cummulative_continuation(ci,cases['Chile'][-1])
y_fc=np.concatenate((np.array([train.iloc[-1]]),y_fc))
X_fc=np.arange(len(train)-1,len(train)+40)
#Graph
fig, ax =plt.subplots()
#ax.set_ylim([-10000,800000])
mgrey=(0.4745098039215686, 0.4745098039215686, 0.4745098039215686)
#sns.lineplot(x=X_fc,y=y_fc,color=mgrey,alpha=1)
#ax.fill_between(X_fc[1:], ci[:,0], ci[:,1], color=mgrey, alpha=0.1,label='95% CI')
sns.lineplot(data=deaths_from_first_ocurrence.loc[30:,countries]/population_2019.reindex(countries)*1000000,dashes=False)
ax.set_title('COVID-19 deaths by days since the first ocurrence')
#ax.set(yscale='log')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,p: format_decimal(y,locale=locale)))
#ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.set_xlabel('Days since the first death')
ax.set_ylabel('Deaths')
ax.legend(loc='center right', bbox_to_anchor=(1.25,0.67))
#ax.lines[0].set_linestyle('--')
footnote='Updated on {}. JHU CSSE COVID-19 Data: https://github.com/CSSEGISandData/COVID-19.'.format(yesterday_str)
plt.figtext(0.5,-0.01,footnote, fontsize=6, ha='center')
fig.savefig('output/deaths_since_1d.png',dpi=300, bbox_inches='tight')
plt.show()

#COVID-19 overview and 10 week projection
countries=list(cases.iloc[-1].sort_values(ascending=False)[0:7].index)+['Chile','Germany']
dblue,pred=sns.xkcd_palette(['denim blue','pale red'])
for country in countries:
    fig, axs=plt.subplots(2,2,figsize=(12,6),sharex='col')
    fig.subplots_adjust(hspace=0.05)
    #Prepare data confirmed cases
    train=cases[country][cases[country]>0].astype(float)
    y_fc,ci=processing.time_series_delta(models_c[country][['yhat']][-1461:-(1460-70)]),processing.time_series_delta(models_c[country][['yhat_lower','yhat_upper']][-1461:-(1460-70)])
    y_fc,ci=np.maximum(0,y_fc)['yhat'].values,np.maximum(0,ci)[['yhat_lower','yhat_upper']].values
    y_fc,ci=processing.cummulative_continuation(y_fc,cases[country][-1]),processing.cummulative_continuation(ci,cases[country][-1])
    y_fc=np.concatenate((np.array([train.iloc[-1]]),y_fc))
    X_fc=pd.date_range(train.index[-1], periods=71, freq='D')
    #Graph confirmed cases
    sns.lineplot(data=train.loc['2020-03-10':],color=dblue,dashes=False,ax=axs[0,0],label='Confirmed cases')
    sns.lineplot(x=X_fc,y=y_fc,color=dblue,alpha=1,ax=axs[0,0],label='Projected cases')
    axs[0,0].lines[1].set_linestyle('--')
    axs[0,0].fill_between(X_fc[1:], ci[:,0], ci[:,1], color=dblue, alpha=0.1,label='95% confidence interval')
    axs[0,0].legend(loc='upper left')
    axs[0,0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,p: format_decimal(y,locale=locale)))
    #axs[0].yaxis.set_major_locator(ticker.MultipleLocator(10000))
    axs[0,0].set_ylabel('Cases')
    #Prepare data daily cases
    train=cases_daily[country].astype(float)
    #y_fc,ci=processing.time_series_delta(models_c[country][['yhat']][-1461:-(1460-70)]),processing.time_series_delta(models_c[country][['yhat_lower','yhat_upper']][-1461:-(1460-70)])
    #y_fc,ci=np.concatenate([train[-7:].values,y_fc['yhat'].values]),np.concatenate([pd.concat([train[-6:],train[-6:]],axis=1).values,ci.values])
    #y_fc,ci=pd.DataFrame(np.maximum(0,y_fc),columns=['yhat']).rolling(7).mean().dropna(),pd.DataFrame(np.maximum(0,ci),columns=[['min','max']]).rolling(7).mean().dropna()
    #y_fc=y_fc['yhat'].values
    #train=train[train>0].rolling(7).mean().dropna()
    train=train.rolling(7).mean().dropna()
    #X_fc=pd.date_range(train.index[-1], periods=71, freq='D')
    #Graph daily cases
    sns.lineplot(data=train.loc['2020-03-10':],color=dblue,dashes=False,ax=axs[0,1],label=None)
    #sns.lineplot(x=X_fc,y=y_fc,color=dblue,alpha=1,ax=axs[0,1],label='Projected daily cases')
    #axs[0,1].lines[1].set_linestyle('--')
    #axs[0,1].fill_between(X_fc[1:], ci[['min','max']].values[:,0], ci[['min','max']].values[:,1], color=dblue, alpha=0.1,label='95% confidence interval')
    #axs[0,1].legend(loc='upper left')
    axs[0,1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,p: format_decimal(y,locale=locale)))
    #axs[0].yaxis.set_major_locator(ticker.MultipleLocator(10000))
    axs[0,1].set_ylabel('Daily cases')
    #Prepare data deaths
    train=deaths[country][deaths[country]>0].astype(float)
    y_fc,ci=y_fc,ci=processing.time_series_delta(models_d[country][['yhat']][-1461:-(1460-70)]).dropna(),processing.time_series_delta(models_d[country][['yhat_lower','yhat_upper']][-1461:-(1460-70)]).dropna()
    y_fc,ci=np.maximum(0,y_fc)['yhat'].values,np.maximum(0,ci)[['yhat_lower','yhat_upper']].values
    y_fc,ci=processing.cummulative_continuation(y_fc,deaths[country][-1]),processing.cummulative_continuation(ci,deaths[country][-1])
    y_fc=np.concatenate((np.array([train.iloc[-1]]),y_fc))
    X_fc=pd.date_range(train.index[-1], periods=71, freq='D')
    #Graph deaths
    sns.lineplot(data=train.loc['2020-03-10':],color=pred,dashes=False,ax=axs[1,0],label='Confirmed deaths')
    sns.lineplot(x=X_fc,y=y_fc,color=pred,alpha=1,ax=axs[1,0],label='Projected deaths')
    axs[1,0].lines[1].set_linestyle('--')
    axs[1,0].fill_between(X_fc[1:], ci[:,0], ci[:,1], color=pred, alpha=0.1,label='95% confidence interval')
    axs[1,0].legend(loc='upper left')
    axs[1,0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,p: format_decimal(y,locale=locale)))
    #axs[1].yaxis.set_major_locator(ticker.MultipleLocator(100))
    plt.setp(axs[1,0].get_xticklabels(), rotation=45)
    axs[1,0].set_xlabel(None)
    axs[1,0].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    #axs[1,0].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    axs[1,0].set_ylabel('Deaths')
    #Prepare data daily deaths
    train=deaths_daily[country].astype(float)
    #y_fc,ci=y_fc,ci=processing.time_series_delta(models_d[country][['yhat']][-1461:-(1460-70)]),processing.time_series_delta(models_d[country][['yhat_lower','yhat_upper']][-1461:-(1460-70)])
    #y_fc,ci=np.concatenate([train[-7:].values,y_fc['yhat'].values]),np.concatenate([pd.concat([train[-6:],train[-6:]],axis=1).values,ci.values])
    #y_fc,ci=pd.DataFrame(np.maximum(0,y_fc),columns=['yhat']).rolling(7).mean().dropna(),pd.DataFrame(np.maximum(0,ci),columns=[['min','max']]).rolling(7).mean().dropna()
    #y_fc=y_fc['yhat'].values
    #train=train[train>0].rolling(7).mean().dropna()
    train=train.rolling(7).mean().dropna()
    X_fc=pd.date_range(train.index[-1], periods=71, freq='D')
    #Graph daily deaths
    sns.lineplot(data=train.loc['2020-03-10':],color=pred,dashes=False,ax=axs[1,1],label=None)
    #sns.lineplot(x=X_fc,y=y_fc,color=pred,alpha=1,ax=axs[1,1],label='Projected daily deaths')
    #axs[1,1].lines[1].set_linestyle('--')
    #axs[1,1].fill_between(X_fc[1:], ci[['min','max']].values[:,0], ci[['min','max']].values[:,1], color=pred, alpha=0.1,label='95% confidence interval')
    #axs[1,1].legend(loc='upper left')
    axs[1,1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,p: format_decimal(y,locale=locale)))
    #axs[0].yaxis.set_major_locator(ticker.MultipleLocator(10000))
    axs[1,1].set_ylabel('Daily deaths')
    axs[1,1].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    #axs[1,1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(axs[1,1].get_xticklabels(), rotation=45)
    axs[1,1].set_xlabel(None)
    fig.suptitle('{}: COVID-19 situation and 10 week forecast'.format(country))
    footnote='Updated on {}. JHU CSSE COVID-19 Data: https://github.com/CSSEGISandData/COVID-19.'.format(yesterday_str)
    plt.figtext(0.5,-0.01,footnote, fontsize=6, ha='center')
    fig.savefig('output/{}.png'.format(str.lower(country)),dpi=300, bbox_inches='tight')
    plt.show()

#Countries COVID-19 evolution
#Initialize subplots
fig, axs=plt.subplots(3,3,figsize=(13,8))
fig.subplots_adjust(hspace=0.6, wspace=.3)
#Create an array of subplots
axs=axs.flatten()
#Creates subplots in a loop
for i in range(len(countries)):
    #Create country's dataframe of cases + deaths
    df=pd.concat([cases[countries[i]],deaths[countries[i]]],axis=1)
    df.columns=['Cases','Deaths']
    #Calculates country's mortality rate
    mortality= round(df.iloc[-1,:]['Deaths']/df.iloc[-1,:]['Cases']*100,1)
    #Country's subplot
    colors=['denim blue','pale red']
    sns.lineplot(data=df,palette=sns.xkcd_palette(colors),dashes=False,legend=False,ax=axs[i])
    axs[i].set(yscale='log')
    axs[i].set_title('{}. Mortality rate: {}%'.format(countries[i],format_decimal(mortality,locale=locale)))
    axs[i].set_xlabel(None)
    axs[i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,p: format_decimal(y,locale=locale)))
    #axs[i].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    axs[i].set_xticklabels(cases.index,rotation=45)
    axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
fig.suptitle('COVID-19 situation overview')
fig.legend(df.columns, bbox_to_anchor=(0.9,0.98), loc='upper right')
footnote='Updated on {}. JHU CSSE COVID-19 Data: https://github.com/CSSEGISandData/COVID-19.'.format(yesterday_str)
plt.figtext(0.5,0.03,footnote, fontsize=6, ha='center')
fig.savefig('output/overview.png',dpi=300, bbox_inches='tight')
plt.show()