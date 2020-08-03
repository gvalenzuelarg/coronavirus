import numpy as np
import pandas as pd
import datetime
from babel.dates import format_date, format_datetime, format_time
from babel.numbers import format_number, format_decimal, format_percent

#Variables with dates
locale='de_DE'
today=datetime.date.today()
today_str=format_date(today,locale=locale)
yesterday=today-datetime.timedelta(1)
yesterday_str=format_date(yesterday,locale=locale)

def cum_to_daily(cum_data):
    #Returns daily changes dataframe provided with cummulative data dataframe
    daily_data=cum_data-cum_data.shift(+1)
    return daily_data

def daily_to_cum(cum_data,daily_forcast):
    #Returns cummulative forcast from daily change forcast, provided with current cummulative data
    cum_forcast=daily_forcast.cumsum(axis=0)
    cum_forcast=cum_forcast+cum_data[-1]
    return cum_forcast

#Data source from JHU CSSE: https://github.com/CSSEGISandData/COVID-19
url_c='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
url_d='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'

#Dataframe of confirmed cases
confirmed=pd.read_csv(url_c)
confirmed=confirmed.groupby('Country/Region').sum().astype(int)
confirmed=confirmed.transpose()
confirmed=confirmed.drop(index=['Lat','Long'])
confirmed.index=pd.to_datetime(confirmed.index)
confirmed.index.name='Date'
confirmed.columns.name=None
#Add a first column 'World'
world_c=pd.DataFrame(confirmed.sum(axis=1),columns=['World'])
confirmed=pd.concat([world_c,confirmed],axis=1)
confirmed.rename(columns={'US': 'USA', 'Korea, South': 'South Korea','United Kingdom':'UK'}, inplace=True)

#Dataframe of daily cases
confirmed_daily=cum_to_daily(confirmed)
confirmed_daily=confirmed_daily.fillna(0)

#Chile: On 2020-06-17 there was a retroactive report of 36179.0 cases. In order for the model to work better I will replace this data point with the average of the surrounding dates
confirmed_daily['Chile']['2020-06-17']=np.mean([confirmed_daily['Chile']['2020-06-16'],confirmed_daily['Chile']['2020-06-18']])

#Dataframe of deaths
deaths=pd.read_csv(url_d)
deaths=deaths.groupby('Country/Region').sum().astype(int)
deaths=deaths.transpose()
deaths=deaths.drop(index=['Lat','Long'])
deaths.index=pd.to_datetime(deaths.index)
deaths.index.name='Date'
deaths.columns.name=None
#Add a first column 'World'
world_d=pd.DataFrame(deaths.sum(axis=1),columns=['World'])
deaths=pd.concat([world_d,deaths],axis=1)
deaths.rename(columns={'US': 'USA', 'Korea, South': 'South Korea','United Kingdom':'UK'}, inplace=True)

#Dataframe of daily deaths
deaths_daily=cum_to_daily(deaths)
deaths_daily=deaths_daily.fillna(0)

#Chile: On 2020-06-08 there was a retroactive report of 627.0 deaths. In order for the model to work better I will replace this data point with the average of the surrounding dates
deaths_daily['Chile']['2020-06-08']=np.mean([deaths_daily['Chile']['2020-06-07'],deaths_daily['Chile']['2020-06-09']])
#Chile: On 2020-07-17 there was a retroactive report of 1057.0 deaths. In order for the model to work better I will replace this data point with the average of the surrounding dates
deaths_daily['Chile']['2020-07-17']=np.mean([deaths_daily['Chile']['2020-07-16'],deaths_daily['Chile']['2020-07-18']])

#Dataframe of confirmed cases reindexed to days since the first instance
confirmed_comp=pd.DataFrame()
all_countries=list(confirmed.columns.values)
for country in all_countries:
    df=confirmed.loc[:,country][confirmed.loc[:,country]>0]
    df=df.reset_index()
    df=df.drop(columns='Date')
    confirmed_comp=pd.concat([confirmed_comp,df], axis=1)

#Dataframe of deaths reindexed to days since the first instance
deaths_comp=pd.DataFrame()
all_countries=list(deaths.columns.values)
for country in all_countries:
    df=deaths.loc[:,country][deaths.loc[:,country]>0]
    df=df.reset_index()
    df=df.drop(columns='Date')
    deaths_comp=pd.concat([deaths_comp,df], axis=1)

#Creates list with the World plus the top 6 countries by confirmed cases plus Chile and Germany
countries=list(confirmed.iloc[-1].sort_values(ascending=False)[0:7].index)+['Chile','Germany']

#Projection of cases for selected countries
from sklearn.metrics import mean_squared_error, explained_variance_score
from fbprophet import Prophet
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
#Logistic Curve model
def logistic_model(X,a,b,c):
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
t_c={'World':'2020-03',
 'USA':'2020-03',
 'Brazil':'2020-06-15',
 'India':None,
 'Russia':'2020-06-21',
 'South Africa':None,
 'Mexico':'2020-07-09',
 'Chile':'2020-07-15',
 'Germany':'2020-05-15'}
#Countries deaths trend points
t_d={'World':'2020-05',
 'USA':'2020-05-15',
 'Brazil':'2020-07-15',
 'India':'2020-06-10',
 'Russia':'2020-06-15',
 'South Africa':'2020-03',
 'Mexico':'2020-07-15',
 'Chile':'2020-06-15',
 'Germany':'2020-05-15'}
 #Countries pandemic status
status={'World':'second wave',
 'USA':'second wave',
 'Brazil':'first wave',
 'India':'first wave',
 'Russia':'first wave',
 'South Africa':'first wave',
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
    y=confirmed[country][t_c[country]:].dropna()
    X=np.arange(len(y))
    parm = curve_fit(logistic_model,X,y.values,p0=[.3,1,y[-1]],maxfev = 100000)
    cap=parm[0][2]
    #Prapare data
    y=confirmed[country].dropna()
    X=np.arange(len(y))
    #Prophet model
    train=pd.DataFrame({'ds':y.index,'y':y.values})
    train['cap'] = cap
    #Choice of model depending saturation status
    if status[country]=='saturation point':
        train=pd.DataFrame({'ds':y[t_c[country]:].index,'y':y[t_c[country]:].values})
        m = Prophet(changepoint_prior_scale=0.5,changepoint_range=1)
    else:
        m = Prophet(growth='logistic',changepoint_prior_scale=0.5,changepoint_range=1)
    m.fit(train)
    future = m.make_future_dataframe(periods=1460)
    future['cap'] = cap
    models_c[country] = m.predict(future)
    #5 weeks predictions
    forecast_daily=np.maximum(0,cum_to_daily(models_c[country][['yhat']][-1461:-(1460-35)])).dropna()
    forecast_daily=pd.DataFrame(forecast_daily.values,index=pd.date_range(today, periods=(35), freq='D'))
    forecast=daily_to_cum(confirmed[country],forecast_daily)
    forecast=pd.DataFrame(forecast,index=pd.date_range(today, periods=(35), freq='D'))
    predictions_in_sample=pd.DataFrame(models_c[country]['yhat'][:len(train)].values,columns=['Prediction'],index=models_c[country]['ds'][:len(train)])
    #country's confirmed cases dataframe
    df_c=pd.concat([confirmed[country][confirmed[country]>0],forecast,forecast_daily],axis=1)
    df_c.columns=['Data','Prediction','Daily']
    df_c.index.name='Date'
    #Saves dataframe to list
    dfs_c.append(df_c)
    #Metrics Prophet model
    RMSE_c_arima=np.round(mean_squared_error(train['y'],predictions_in_sample)**(1/2),2)
    explained_variance_c_arima=np.round(explained_variance_score(train['y'],predictions_in_sample),3)
    #Fitting a logistic model for long term predictions
    parm_c[country] = curve_fit(logistic_model,X,y.values,p0=[0.3,50,y[-1]],maxfev = 100000)
    a,b,c=parm_c[country][0]
    #Calculate date of pick daily infections
    #y_year=logistic_model(np.arange(365),a,b,c)
    #dy=np.diff(y_year)
    #arg_max=np.argmax(dy)
    first_day=y.index[0]
    max_daily_date=first_day+datetime.timedelta(int(b))
    #Calculate end of pandemic
    sol = int(fsolve(lambda x : logistic_model(x,a,b,c) - int(c),b))
    ending_date=first_day+datetime.timedelta(sol)
    #Total confirmed cases
    forecast_daily=cum_to_daily(models_c[country][['yhat']][-1461:-(1461-(sol-len(train)))]).dropna()
    forecast=daily_to_cum(confirmed[country],forecast_daily)
    total_cases=forecast['yhat'].values[-1]
    #Calculates country's mortality rate
    mortality= round(deaths.iloc[-1,:][country]/confirmed.iloc[-1,:][country]*100,1)
    #Metrics logistic model
    RMSE_c_log=mean_squared_error(y,logistic_model(X,a,b,c))**(1/2)
    explained_variance_c_log=explained_variance_score(y,logistic_model(X,a,b,c))
    #Metrics avg.
    RMSE_c=np.round(np.mean([RMSE_c_arima,RMSE_c_log]),2)
    explained_variance_c=np.round(np.mean([explained_variance_c_arima,explained_variance_c_log]),3)

    #Deaths prediction
    #Cap estimation
    y=deaths[country][t_d[country]:].dropna()
    X=np.arange(len(y))
    parm = curve_fit(logistic_model,X,y.values,p0=[.3,1,y[-1]],maxfev = 100000)
    cap=parm[0][2]
    #Prapare data
    y=deaths[country].dropna()
    X=np.arange(len(y))
    #Prophet model
    train=pd.DataFrame({'ds':y.index,'y':y.values})
    train['cap'] = cap
    #Choice of model depending saturation status
    if status[country]=='saturation point':
        train=pd.DataFrame({'ds':y[t_d[country]:].index,'y':y[t_d[country]:].values})
        m = Prophet(changepoint_prior_scale=0.5,changepoint_range=1)
    else:
        m = Prophet(growth='logistic',changepoint_prior_scale=0.5,changepoint_range=1)
    m.fit(train)
    future = m.make_future_dataframe(periods=1460)
    future['cap'] = cap
    models_d[country] = m.predict(future)
    #5 weeks predictions
    forecast_daily=np.maximum(0,cum_to_daily(models_d[country][['yhat']][-1461:-(1460-35)])).dropna()
    forecast_daily=pd.DataFrame(forecast_daily.values,index=pd.date_range(today, periods=(35), freq='D'))
    forecast=daily_to_cum(deaths[country],forecast_daily.values)
    forecast=pd.DataFrame(forecast,index=pd.date_range(today, periods=(35), freq='D'))
    predictions_in_sample=pd.DataFrame(models_d[country]['yhat'][:len(train)].values,columns=['Prediction'],index=models_d[country]['ds'][:len(train)])
    #country's deaths dataframe
    df_d=pd.concat([deaths[country][deaths[country]>0],forecast,forecast_daily],axis=1)
    df_d.columns=['Data','Prediction','Daily']
    df_d.index.name='Date'
    #Saves dataframe to list
    dfs_d.append(df_d)
    #Metrics Prophet model
    RMSE_d_arima=mean_squared_error(train['y'],predictions_in_sample)**(1/2)
    explained_variance_d_arima=explained_variance_score(train['y'],predictions_in_sample)
    #Fitting a logistic model for long term predictions
    parm_d[country] = curve_fit(logistic_model,X,y.values,p0=[0.3,50,y[-1]],maxfev = 10000)
    a,b,c=parm_d[country][0]
    #Total deaths
    forecast_daily=cum_to_daily(models_d[country][['yhat']][-1461:-(1460-(sol-len(train)))]).dropna()
    forecast=daily_to_cum(deaths[country],forecast_daily)
    total_deaths=forecast['yhat'].values[-1]
    #Metrics logistic model
    RMSE_d_log=mean_squared_error(y,logistic_model(X,a,b,c))**(1/2)
    explained_variance_d_log=explained_variance_score(y,logistic_model(X,a,b,c))
    #Metrics avg.
    RMSE_d=np.round(np.mean([RMSE_d_arima,RMSE_d_log]),2)
    explained_variance_d=np.round(np.mean([explained_variance_d_arima,explained_variance_d_log]),3)

    #Create a list of lines for the report
    textlist.append('{}:'.format(country))
    textlist.append('')
    textlist.append('\tData on {}:'.format(yesterday_str))
    textlist.append('')
    textlist.append('\t\tConfirmed cases: {} ({}).'.format(format_decimal(confirmed.iloc[-1,:][country],locale=locale),format_decimal((confirmed.iloc[-1,:][country]-confirmed.iloc[-2,:][country]).astype(int),locale=locale,format='+#,###;-#')))
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
model_data_c.to_csv('output/confirmed_cases.csv', float_format='%.f')
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
countries=list(confirmed.iloc[-1].sort_values(ascending=False)[1:7].index)+['Germany','Chile']

#Confirmed cases graph
#Prepare data
df_c=model_data_c.loc[:,(countries,'Prediction')].dropna().iloc[:35]
df_c.columns=df_c.columns.droplevel(1)
df_c=confirmed.loc[yesterday:yesterday][countries].append(df_c)
#Graph
fig, ax =plt.subplots()
sns.lineplot(data=df_c,alpha=1,dashes=False,legend=False)
for i in np.arange(8):
    ax.lines[i].set_linestyle('--')
sns.lineplot(data=confirmed.loc['2020-03-01':,countries],dashes=False)
#ax.set(yscale='log')
ax.set_title('Confirmed Cases')
ax.set_xlabel(None)
ax.set_ylabel('Number of confirmed cases')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,p: format_decimal(y,locale=locale)))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=30)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.set_ylim(-10000,700000)
ax.legend(loc='center right', bbox_to_anchor=(1.25,0.6))
#plt.savefig('confirmed.pdf', bbox_inches='tight')
fig.savefig('output/confirmed.png', dpi=300, bbox_inches='tight')
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
sns.lineplot(data=confirmed_daily.loc['2020-03-01':,countries].rolling(7).mean(),dashes=False)
#ax.set(yscale='log')
ax.set_title('Daily Cases')
ax.set_xlabel(None)
ax.set_ylabel('Number of daily cases (7 day rolling average)')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,p: format_decimal(y,locale=locale)))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=30)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.legend(loc='center right', bbox_to_anchor=(1.25,0.6))
fig.savefig('output/confirmed_daily.png', dpi=300, bbox_inches='tight')
plt.show()

#Deaths graph
#Prepare data
df_d=model_data_d.loc[:,(countries,'Prediction')].dropna().iloc[:35]
df_d.columns=df_d.columns.droplevel(1)
df_d=deaths.loc[yesterday:yesterday][countries].append(df_d)
#Graph
fig, ax =plt.subplots()
ax.set_ylim(-250,20000)
sns.lineplot(data=df_d,alpha=1,dashes=False,legend=False)
for i in np.arange(8):
    ax.lines[i].set_linestyle('--')
sns.lineplot(data=deaths.loc['2020-03-01':,countries],dashes=False)
#ax.set(yscale='log')
ax.set_title('Deaths')
ax.set_xlabel(None)
ax.set_ylabel('Number of deaths')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,p: format_decimal(y,locale=locale)))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=30)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.legend(loc='center right', bbox_to_anchor=(1.25,0.6))
#plt.savefig('deaths.pdf',bbox_inches='tight')
fig.savefig('output/deaths.png',dpi=300,bbox_inches='tight')
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
ax.set_title('Daily Deaths')
ax.set_xlabel(None)
ax.set_ylabel('Number of daily deaths (7 day rolling average)')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,p: format_decimal(y,locale=locale)))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=30)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.legend(loc='center right', bbox_to_anchor=(1.25,0.6))
fig.savefig('output/deaths_daily.png', dpi=300, bbox_inches='tight')
plt.show()

#Projection of cases in Chile graph
#Prepare data
train=confirmed['Chile'][confirmed['Chile']>0].astype(float)
y_fc,ci=cum_to_daily(models_c['Chile'][['yhat']][-1461:-(1460-40)]).dropna(),cum_to_daily(models_c['Chile'][['yhat_lower','yhat_upper']][-1461:-(1460-40)]).dropna()
y_fc,ci=np.maximum(0,y_fc)['yhat'].values,np.maximum(0,ci)[['yhat_lower','yhat_upper']].values
y_fc,ci=daily_to_cum(confirmed['Chile'],y_fc),daily_to_cum(confirmed['Chile'],ci)
y_fc=np.concatenate((np.array([train.iloc[-1]]),y_fc))
X_fc=np.arange(len(train)-1,len(train)+40)
#Graph
fig, ax =plt.subplots()
ax.set_ylim([-10000,800000])
mgrey=(0.4745098039215686, 0.4745098039215686, 0.4745098039215686)
sns.lineplot(x=X_fc,y=y_fc,color=mgrey,alpha=1)
ax.fill_between(X_fc[1:], ci[:,0], ci[:,1], color=mgrey, alpha=0.1,label='95% CI')
sns.lineplot(data=confirmed_comp.loc[30:,countries],dashes=False)
ax.set_title('Projection of COVID-19 cases in Chile')
#ax.set(yscale='log')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,p: format_decimal(y,locale=locale)))
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.set_xlabel('Days since the first case')
ax.set_ylabel('Number of cases')
ax.legend(loc='center right', bbox_to_anchor=(1.25,0.6))
ax.lines[0].set_linestyle('--')
footnote='Updated on {}. Source: Johns Hopkins CSSE. https://github.com/CSSEGISandData/COVID-19'.format(yesterday_str)
plt.figtext(0.5,-0.01,footnote, fontsize=6, ha='center')
fig.savefig('output/projection_chile.png',dpi=300, bbox_inches='tight')
plt.show()

#Graph of projections for countries
countries=list(confirmed.iloc[-1].sort_values(ascending=False)[0:7].index)+['Chile','Germany']
for country in countries:
    dblue,pred=sns.xkcd_palette(['denim blue','pale red'])
    fig, axs=plt.subplots(2,2,figsize=(12,6),sharex='col')
    fig.subplots_adjust(hspace=0.05)
    #Prepare data confirmed cases
    train=confirmed[country][confirmed[country]>0].astype(float)
    y_fc,ci=cum_to_daily(models_c[country][['yhat']][-1461:-(1460-70)]).dropna(),cum_to_daily(models_c[country][['yhat_lower','yhat_upper']][-1461:-(1460-70)]).dropna()
    y_fc,ci=np.maximum(0,y_fc)['yhat'].values,np.maximum(0,ci)[['yhat_lower','yhat_upper']].values
    y_fc,ci=daily_to_cum(confirmed[country],y_fc),daily_to_cum(confirmed[country],ci)
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
    axs[0,0].set_ylabel('Number of cases')
    #Prepare data daily cases
    train=confirmed_daily[country].astype(float)
    #y_fc,ci=cum_to_daily(models_c[country][['yhat']][-1461:-(1460-70)]).dropna(),cum_to_daily(models_c[country][['yhat_lower','yhat_upper']][-1461:-(1460-70)]).dropna()
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
    axs[0,1].set_ylabel('Number of daily cases')
    #Prepare data deaths
    train=deaths[country][deaths[country]>0].astype(float)
    y_fc,ci=y_fc,ci=cum_to_daily(models_d[country][['yhat']][-1461:-(1460-70)]).dropna(),cum_to_daily(models_d[country][['yhat_lower','yhat_upper']][-1461:-(1460-70)]).dropna()
    y_fc,ci=np.maximum(0,y_fc)['yhat'].values,np.maximum(0,ci)[['yhat_lower','yhat_upper']].values
    y_fc,ci=daily_to_cum(deaths[country],y_fc),daily_to_cum(deaths[country],ci)
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
    axs[1,0].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    axs[1,0].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    axs[1,0].set_ylabel('Number of deaths')
    #Prepare data daily deaths
    train=deaths_daily[country].astype(float)
    #y_fc,ci=y_fc,ci=cum_to_daily(models_d[country][['yhat']][-1461:-(1460-70)]).dropna(),cum_to_daily(models_d[country][['yhat_lower','yhat_upper']][-1461:-(1460-70)]).dropna()
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
    axs[1,1].set_ylabel('Number of daily deaths')
    axs[1,1].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    axs[1,1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(axs[1,1].get_xticklabels(), rotation=45)
    fig.suptitle('Projections for COVID-19 in {}'.format(country))
    footnote='Updated on {}. Source: Johns Hopkins CSSE. https://github.com/CSSEGISandData/COVID-19'.format(yesterday_str)
    plt.figtext(0.5,-0.01,footnote, fontsize=6, ha='center')
    fig.savefig('output/{}_proj.png'.format(str.lower(country)),dpi=300, bbox_inches='tight')
    plt.show()

#Countries performance comparison
#Initialize subplots
fig, axs=plt.subplots(3,3,figsize=(13,8))
fig.subplots_adjust(hspace=0.6, wspace=.3)
#Create an array of subplots
axs=axs.flatten()
#Creates subplots in a loop
for i in range(len(countries)):
    #Create country's dataframe of cases + deaths
    df=pd.concat([confirmed[countries[i]],deaths[countries[i]]],axis=1)
    df.columns=['Confirmed Cases','Deaths']
    #Calculates country's mortality rate
    mortality= round(df.iloc[-1,:]['Deaths']/df.iloc[-1,:]['Confirmed Cases']*100,1)
    #Country's subplot
    colors=['denim blue','pale red']
    sns.lineplot(data=df,palette=sns.xkcd_palette(colors),dashes=False,legend=False,ax=axs[i])
    axs[i].set(yscale='log')
    axs[i].set_title('{}. Mortality rate on {}: {}'.format(countries[i],yesterday_str,str(mortality)+'%'))
    axs[i].set_xlabel(None)
    axs[i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,p: format_decimal(y,locale=locale)))
    axs[i].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    axs[i].set_xticklabels(confirmed.index,rotation=45)
    axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
fig.suptitle('Comparison of COVID-19 performance')
fig.legend(df.columns,loc='center right', bbox_to_anchor=(0.82,0.89))
fig.savefig('output/comparison.png',dpi=300, bbox_inches='tight')
plt.show()