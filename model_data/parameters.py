"""Parameters for trained models.
"""

# Countries start date for cases model traning
start_cases = {
    'World' : None,
    'USA' : '2020-05',
    'Brazil' : '2020-06-15',
    'India' : '2020-06',
    'Russia' : '2020-07-20',
    'South Africa' : '2020-09',
    'Peru' : '2020-04-10',
    'Colombia' : None,
    'Mexico' : '2020-07-09',
    'Chile' : '2020-07-15',
    'Germany' : '2020-05-15'
    }
# Countries start date for deaths model traning
start_deaths = {
    'World' : '2020-04-15',
    'USA' : '2020-05',
    'Brazil' : '2020-07-15',
    'India' : '2020-06-20',
    'Russia' : '2020-08',
    'South Africa' : '2020-05',
    'Peru' : '2020-05-29',
    'Colombia' : None,
    'Mexico' : '2020-07-10',
    'Chile' : '2020-07-18',
    'Germany' : '2020-05-15'
    }
# Countries hyperparameters for cases models
params_cases = {
    'World' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 5,
        'seasonality_prior_scale' : 0.1},
    'USA' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 1},
    'Brazil' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.05,
        'seasonality_prior_scale' : 1},
    'India' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 5,
        'seasonality_prior_scale' : 1},
    'Russia' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 0.1},
    'South Africa' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 5,
        'seasonality_prior_scale' : 0.1},
    'Peru' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 5,
        'seasonality_prior_scale' : 0.1},
    'Colombia' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.05,
        'seasonality_prior_scale' : 10},
    'Mexico' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 5,
        'seasonality_prior_scale' : 1},
    'Chile' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 5,
        'seasonality_prior_scale' : 1},
    'Germany' : {
        'growth' : 'linear',
        'changepoint_prior_scale' : 0.05,
        'seasonality_prior_scale' : 10}
    }
# Countries hyperparameters for deaths models
params_deaths = {
    'World':{
        'growth' : 'logistic',
        'changepoint_prior_scale' : 5,
        'seasonality_prior_scale' : 10},
    'USA' : {
            'growth' : 'logistic',
            'changepoint_prior_scale' : 0.5,
            'seasonality_prior_scale' : 10},
    'Brazil' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 5,
        'seasonality_prior_scale' : 0.1},
    'India' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 5,
        'seasonality_prior_scale' : 1},
    'Russia' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 5,
        'seasonality_prior_scale' : 0.1},
    'South Africa' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 5,
        'seasonality_prior_scale' : 10},
    'Peru' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 10},
    'Colombia' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.05,
        'seasonality_prior_scale' : 1},
    'Mexico' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 5,
        'seasonality_prior_scale' : 10},
    'Chile' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 5,
        'seasonality_prior_scale' : 0.1},
    'Germany' : {
        'growth' : 'linear',
        'changepoint_prior_scale' : 0.05,
        'seasonality_prior_scale' : 10}
    }
#Countries cases outliers
outliers_cases = {
    'World' : [],
    'USA' : [],
    'Brazil' : ['2020-06-19'],
    'India' : [],
    'Russia' : [],
    'South Africa' : [],
    'Peru' : ['2020-06-03','2020-06-12','2020-07-25','2020-07-26','2020-07-27','2020-07-30','2020-08-01','2020-08-02','2020-08-08','2020-08-09','2020-08-12','2020-08-14','2020-08-15','2020-09-03','2020-09-04'],
    'Colombia' : [],
    'Mexico' : [],
    'Chile' : ['2020-05-30','2020-06-06'],
    'Germany' : []
    }
#Countries deaths outliers
outliers_deaths = {
    'World' : ['2020-08-14'],
    'USA' : [],
    'Brazil' : [],
    'India' : ['2020-06-16'],
    'Russia' : [],
    'South Africa' : ['2020-07-22'],
    'Peru' : ['2020-07-23','2020-08-14'],
    'Colombia' : [],
    'Mexico' : ['2020-06-04'],
    'Chile' : ['2020-06-08','2020-07-17'],
    'Germany' : []
    }