import operator
from sklearn.externals import joblib# import cPickle
import glob
from sklearn.cross_validation import train_test_split
from math import *
from model_training_evaluation import *
from data_preprocess import *
from gridding import *
import json
from prediction import *

def save_model_result(row_ids,y_pred_class, y_preds,name):
    output_dict = {}
    # print y_preds
    for row_id, y_pred in zip(row_ids,y_preds):
        output_dict[row_id] = [[i,j] for i,j in zip(y_pred_class,y_pred) if j>0]

    with open(name+'.json', 'w') as fp:
        json.dump(output_dict, fp)



# for each grids train and save models
def train_save_all_models(train_grids_path,test_grids_path,start, path ='/Users/yetu/kaggle/model_outputs/'):
    if not os.path.exists(path):
        os.makedirs(path)
    for (train_grid_path,test_grid_path) in zip(train_grids_path,test_grids_path):
        train_grid_name = train_grid_path.split('.')[0].split('/')[-1]

        if train_grid_name>= start:
            print train_grid_name
            train_grid = pd.read_csv( train_grid_path,sep=',', header= 0)
            test_grid = pd.read_csv( test_grid_path,sep=',', header= 0)
            train_grid = preprocess(train_grid)
            test_grid = preprocess(test_grid)

            X_names = ['x','y', 'time', 'xy', 'x_y', 'y_x', 'year', 'month', 'hour', 'weekday', 'minute','weight']

            X_train = train_grid[X_names]
            y_train = train_grid.place_id
            X_test =test_grid[X_names]
            #train model and output probabilities
            train_model = RandomForestClassifier(n_estimators=500, random_state=2).fit(X_train,y_train)
            y_preds = train_model.predict_proba(X_test)
            y_pred_class = train_model.classes_
            name = path + train_grid_name
            save_model_result(test_grid.row_id,y_pred_class,y_preds,name)
    return

 
sample_rate = 0.01
path = '/Users/yetu/kaggle/'
grid_size = 0.2

train_columns = ["row_id","x","y","accuracy","time","place_id"]
test_columns = ["row_id","x","y","accuracy","time"]
train_df = pd.read_csv(path+'train.csv', sep = ',', header= 0, names = train_columns)
# test_df = pd.read_csv(path+'test.csv', sep = ',', header= 0, names = test_columns)

sampled_df = train_df.sample(frac = sample_rate)

train_df, test_df = train_test_split(sampled_df, test_size = 0.5)
# # test_df = train_df.sample(frac = sample_rate)
test_df.to_csv(path + 'test_df.csv', sep=',')
#  # test_df.sample(frac = sample_rate)

# partition training data into n*n files
gridding(train_df, grid_size, path + 'train_grids/', train_columns)
gridding(test_df, grid_size, path + 'test_grids/', test_columns)

train_grids = glob.glob('/Users/yetu/kaggle/train_grids/*.csv')
test_grids = glob.glob('/Users/yetu/kaggle/test_grids/*.csv')

train_save_all_models(train_grids,test_grids,'')

test_df = pd.read_csv(path+'test_df.csv', sep = ',', header= 0, names = train_columns)
test_df = preprocess(test_df)

result = predict(test_df, grid_size, path+'model_outputs/')
# for i, j in zip(test_df.place_id,result):
#     print i,j
print accuracy_score(test_df.place_id,result)

