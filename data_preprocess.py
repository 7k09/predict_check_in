import time
import numpy as np
import seaborn as sns
import numpy as np
import seaborn as sns
import pandas as pd
import statsmodels.api as sm

from sklearn import preprocessing
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import preprocessing


def load_data(file,column_names):
    df = pd.read_csv(file,sep=',', header=None, names = column_names)
#     df.columns = column_names
    return df

def format_data(df, y_name,X_names,intercept_flag = True):

    y, X = dmatrices(y_name + ' ~ '+ ' + '.join(X_names),
                  df, return_type="dataframe")
    print X.columns
    y = np.ravel(y)
    if intercept_flag == False:
        X = X.ix[:,1:]
    # fix column names of X
    #X_scaled = min_max_scaling(X)
    return X,y
def stats_summary(data):
    med = np.median(data)
    avg = np.mean(data)
    minimum = np.min(data)
    maximum = np.max(data)
    first_quantile = np.percentile(data,25)
    third_quantile = np.percentile(data,75)
    print " min:{mi} \n 25_percentile: {first} \n median: {m} \n avg: {a} \n 75_percentile: {third} \n max:{ma}"\
    .format(m=med,a= avg,ma = maximum,mi=minimum,first = first_quantile, third = third_quantile)

def plot_dist(X):
    X = X.ix[:,1:]

    for i in range(len(X.columns)):
        col = X.ix[:,i]
#         print X.columns[i]
        fig = plt.figure()
        plt.hist(list(col))


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def preprocess(df):
    newdf =df
    newdf['xy'] = df.x.values * df.y.values
    newdf['x_y'] = df.x.values / (df.y.values + 1e-7)
    newdf['y_x'] = df.y.values / (df.x.values + 1e-7)
    init_time = np.datetime64('2016-01-01T00:00', dtype='datetime64')
    times = pd.DatetimeIndex(init_time + np.timedelta64(int(t), 'm') for t in df.time.values)
    newdf['year'] = times.year
    newdf['month'] = times.month
    newdf['hour'] = times.hour
    newdf['weekday'] = times.weekday
    newdf['minute'] = times.minute
    return newdf