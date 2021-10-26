"""Parameters for trained models.
"""

# Cases cap
cap_cases = {
    'World' : 434881776,
    'USA' : 50547158,
    'Brazil' : 23760170,
    'India' : 40050958,
    'Russia' : 18712999,
    'UK' : 10717540,
    'France' : 12990394,
    'Turkey' : 13185717,
    'Chile' : 1883451,
    'Germany' : 5087404
}
# Deaths cap
cap_deaths = {
    'World' : 8912188,
    'USA' : 823165,
    'Brazil' : 675804,
    'India' : 555068,
    'Russia' : 483474,
    'UK' : 173013,
    'France' : 178560,
    'Turkey' : 103421,
    'Chile' : 38704,
    'Germany' : 105700
}
# Start date for cases model training
start_cases = {
    'World' : '2021-06',
    'USA' : '2021-06',
    'Brazil' : '2021-06',
    'India' : '2021-06',
    'Russia' : '2021-06',
    'South Africa' : '2020-09',
    'Peru' : '2020-08',
    'Colombia' : '2020-08-15',
    'Mexico' : '2020-07',
    'Spain' : '2020-06',
    'Argentina' : '2020-09-15',
    'France' : '2021-06',
    'UK' : '2021-06',
    'Italy' : '2020-11',
    'Turkey' : '2021-06', 
    'Chile' : '2021-06',
    'Germany' : '2021-06'
    }
# Start date for deaths model training
start_deaths = {
    'World' : '2021-06',
    'USA' : '2021-06',
    'Brazil' : '2021-06',
    'India' : '2021-06',
    'Russia' : '2021-06',
    'South Africa' : '2020-09',
    'Peru' : '2020-09',
    'Colombia' : '2020-09',
    'Mexico' : '2020-08',
    'Spain' : '2020-08',
    'Argentina' : '2020-07',
    'France' : '2021-06',
    'UK': '2021-06',
    'Italy' : '2020-11',
    'Turkey' : '2021-05',
    'Chile' : '2021-06',
    'Germany' : '2021-06'
    }
# Hyperparameters for cases model
params_cases = {
    'World' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 0.1},
    'USA' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 10},
    'Brazil' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 10},
    'India' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 1},
    'Russia' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 0.1},
    'South Africa' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 10},
    'Peru' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 10},
    'Colombia' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 0.1},
    'Mexico' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 0.1},
    'Spain' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 0.1},
    'Argentina' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 10},
    'France' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 5,
        'seasonality_prior_scale' : 1},
    'UK' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 0.1},
    'Italy' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 1},
    'Turkey' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 10},
    'Chile' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 0.1},
    'Germany' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 0.1}
    }
# Hyperparameters for deaths model
params_deaths = {
    'World':{
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 1},
    'USA' : {
            'growth' : 'logistic',
            'changepoint_prior_scale' : 5,
            'seasonality_prior_scale' : 1},
    'Brazil' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 5,
        'seasonality_prior_scale' : 10},
    'India' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 5,
        'seasonality_prior_scale' : 0.1},
    'Russia' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 10},
    'South Africa' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 1},
    'Peru' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 1},
    'Colombia' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 10},
    'Mexico' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 5,
        'seasonality_prior_scale' : 0.1},
    'Spain' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 5,
        'seasonality_prior_scale' : 1},
    'Argentina' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 10},
    'France' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 10},
    'UK' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 1},
    'Italy' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 0.1},
    'Turkey' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 10},
    'Chile' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 5,
        'seasonality_prior_scale' : 0.1},
    'Germany' : {
        'growth' : 'logistic',
        'changepoint_prior_scale' : 0.5,
        'seasonality_prior_scale' : 10}
    }
#Case outliers
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
    'Spain' : [],
    'Argentina' : [],
    'France' : ['2021-05-20'],
    'UK' : [],
    'Italy': [],
    'Turkey': [],
    'Chile' : ['2020-05-30','2020-06-06'],
    'Germany' : []
    }
#Death outliers
outliers_deaths = {
    'World' : ['2020-08-14', '2021-06-03'],
    'USA' : [],
    'Brazil' : [],
    'India' : ['2020-06-16'],
    'Russia' : [],
    'South Africa' : ['2020-07-22'],
    'Peru' : ['2020-07-23','2020-08-14'],
    'Colombia' : [],
    'Mexico' : ['2020-06-04'],
    'Spain' : [],
    'Argentina' : ['2020-10-01'],
    'France' : [],
    'UK' : [],
    'Italy' : [],
    'Turkey': [],
    'Chile' : ['2020-06-08','2020-07-17'],
    'Germany' : ['2020-10-08']
    }