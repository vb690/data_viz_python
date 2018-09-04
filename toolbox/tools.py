import random
import string

import numpy as np

import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit

########################### utility functions ##################################

def generate_string(lenght):
    '''
    Generate a random string
    '''
    return ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase, k=lenght))

def generate_mapping(df, column):
    '''
    Generate a dictionary mapping categorical variables into numerical given a dataframe and a column
    '''
    return {label : code for code, label in enumerate(df[column].unique())}

########################### data generation functions ###########################

def categorical_noise(column, ammount):
    '''
    Given a dataframe column of categorical values add categorical noise equal to ammount
    '''
    ammount = int(len(column) * ammount)
    sample = np.random.choice(column.index, ammount)
    string = generate_string(5)
    column.loc[sample] = column.loc[sample].apply(lambda x: string)
    return column

def scalar_noise(column, ammount):
    '''
    Given a dataframe column of scalar values add noise equal to ammount
    '''
    ammount = int(len(column) * ammount)
    sample = np.random.choice(column.index, ammount)
    std = column.std()
    column.loc[sample] = column.loc[sample].apply(lambda x: x + std*4)
    return column
        
def add_noise (df, n_columns, ammount):
    '''
    Given a dataframe add noise to n_columns
    '''
    columns = list(df)
    for column in np.random.choice(columns, n_columns):
        
        if df[column].dtype == 'object':
            df[column] = categorical_noise(df[column], ammount)
        elif df[column].dtype in ['int64', 'int32', 'float64', 'float32']:
            df[column] = scalar_noise(df[column], ammount)
    
    return df

def add_spurious_columns (df, n_columns):
    '''
    Given a dataframe add n spurious column made of random noramlly distributed values
    '''
    for column in range(n_columns):
        
        df['spurious_'+str(column)] = np.random.normal(0, 5, len(df))
    
    return df

def generate_synthetic(n_categorical, n_scalar, n_rows):
    '''
    Generate a synthetic dataframe with n_scalar columns, n_categorical columns and n_rows
    '''
    df = pd.DataFrame()
    for column in range(n_categorical):
        
        categories = [generate_string(5) for category in range(3)]
        df['categorical_'+str(column)] = np.random.choice(categories, n_rows)
    
    for column in range(n_scalar):
        
        df['scalar_'+str(column)] = np.random.normal(0, 1, n_rows)

    return pd.DataFrame(df)

def generate_time_series(len_series, n_conditions, n_individuals):
    '''
    Generate time series in the ammount  conditions X n_individuals
    '''
    t_minus_1 = 0
    t_step = []
    condition = []
    individual = []
    value = []
    for i in range(n_individuals):
        
        for c in range(n_conditions):

            condition_mean = np.random.randint(0, 50)
            condition_std = np.random.randint(1, 5)

            for t in range(len_series):

                if t == 0:
                    t_minus_1 = np.random.normal(condition_mean, condition_std)
                    t_step.append(t)
                    condition.append(c)
                    individual.append(i)
                    value.append(t_minus_1)
                else:
                    fluctuation = np.random.choice(['up', 'down'])
                    ammount = np.random.normal(condition_mean, condition_std)
                    if fluctuation == 'up':
                        t_minus_1 += ammount
                    else:
                        t_minus_1 -= ammount
                    t_step.append(t)
                    condition.append(c)
                    individual.append(i)
                    value.append(t_minus_1)

    df = pd.DataFrame()
    df['time'] = t_step
    df['condition'] = condition
    df['individual'] = individual
    df['value'] = value
    return df

def generate_linear(n_individuals, n_conditions):
    '''
    Generate a dataset having for each condition y = w*x
    '''
    conditions = []
    X = []
    y = []
    for condition in range(n_conditions):

        x = np.linspace(-3, 3, n_individuals)
        weights = np.random.uniform(-6, 6, 1)

        conditions.extend([condition for individual in range(n_individuals)])
        X.extend(x)
        y.extend(sum([x* weight for weight in weights]))

    df = pd.DataFrame()
    df['condition'] = conditions
    df['x'] = X
    df['y'] = y
    return df

def generate_poly(n_individuals, n_conditions, order):
    '''
    Generate a dataset having for each condition y = w*(x**order)
    '''
    conditions = []
    X = []
    y = []
    for condition in range(n_conditions):

        x = np.linspace(-3, 3, n_individuals)
        weights = np.random.uniform(-6, 6, order+1)
        
        conditions.extend([condition for individual in range(n_individuals)])
        X.extend(x)
        y.extend(sum([weight*(x**order) for order, weight in enumerate(weights)]))

    df = pd.DataFrame()
    df['condition'] = conditions
    df['x'] = X
    df['y'] = y
    return df
                      
########################### data generation functions ###########################

def generate_X_y(df, X_columns, y_column, normalize = False):
    '''
    Given a dataframe returns numpy array of feature columns and target columns, used for preparing data for modelling, allows to normalize the data
    '''
    X = np.array(df[X_columns]) 
    y = np.array(df[y_column])
    if normalize:
        X = Normalizer().fit_transform(X)
    return X, y

def generating_validation_test(X, y, drop_size=0.5):
    '''
    Given feature and target arrays generate validation and test sets
    '''
    for validation, test in StratifiedShuffleSplit(n_splits = 1, test_size = 0.3).split(X, y):
        
        X_validation, y_validation = X[validation], y[validation]
        X_test, y_test = X[test], y[test]

    return X_validation, y_validation, X_test, y_test
