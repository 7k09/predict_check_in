import json
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
import numpy as np

def preprocess(df):
    newdf =df
    newdf['xy'] = df.x.values * df.y.values
    newdf['x_y'] = df.x.values / (df.y.values + 1e-7)
    newdf['y_x'] = df.y.values / (df.x.values + 1e-7)
    init_time = np.datetime64('2015-01-01T00:00', dtype='datetime64')
    times = pd.DatetimeIndex(init_time + np.timedelta64(int(t), 'm') for t in df.time.values)
    newdf['year'] = times.year
    newdf['month'] = times.month
    newdf['hour'] = times.hour
    newdf['weekday'] = times.weekday
    newdf['minute'] = times.minute
    return newdf

def save_model_result(row_ids, y_pred_class, y_preds, weights, name):
    output_dict = {}
    for (row_id, y_pred, w) in zip(row_ids, y_preds, weights):
        curr_dict = {}
        for place, prob in zip(y_pred_class, y_pred):
            if prob > 0:
                curr_dict[place] = prob * w
        output_dict[row_id] = curr_dict

    with open(name+'.json', 'w') as fp:
        json.dump(output_dict, fp)



# for each grids train and save models
def train_save_all_models(train_grids_path,test_grids_path, path, start=''):
    if not os.path.exists(path):
        os.makedirs(path)
    for (train_grid_path,test_grid_path) in zip(train_grids_path,test_grids_path):
        train_save_model(train_grid_path, test_grid_path, path, start)
    return
    
def train_save_model(train_grid_path, test_grid_path, path, start=''):
    train_grid_name = train_grid_path.split('.')[0].split('/')[-1]
    if train_grid_name <= start:
        return
    print train_grid_name
    data_type = {'row_id': np.int, 'x':np.float32, 'y':np.float32, 'accuracy':np.float32, 'time':np.int, 'place_id':np.int}
    data_type_test = {'row_id': np.int, 'x':np.float32, 'y':np.float32, 'accuracy':np.float32, 'time':np.int}
    train_grid = pd.read_csv(train_grid_path, sep=',', dtype=data_type, header=0, index_col=0)
    test_grid = pd.read_csv(test_grid_path, sep=',', dtype=data_type_test, header=0, index_col=0)
    train_grid = preprocess(train_grid)
    test_grid = preprocess(test_grid)
    X_names = ['x','y', 'accuracy', 'time', 'xy', 'x_y', 'y_x', 'year', 'month', 'hour', 'weekday', 'minute','weight']
    X_train = train_grid[X_names]
    y_train = train_grid.place_id
    X_test =test_grid[X_names]
    #train model and output probabilities
    train_model = RandomForestClassifier(n_estimators=500, random_state=10).fit(X_train,y_train,X_train.weight.values)
    y_preds, i = [], 0
    block_size = 8000
    while i < len(X_test):
        y_preds += train_model.predict_proba(X_test[i : i+block_size]).tolist()
        i += block_size
    y_pred_class = train_model.classes_
    name = path + train_grid_name
    save_model_result(test_grid.row_id.values, y_pred_class, y_preds, test_grid.weight.values, name)
    return
    
def train_save_model_wrapper(args):
    train_save_model(*args)