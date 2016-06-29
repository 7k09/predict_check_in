import operator
import glob
from sklearn.cross_validation import train_test_split
from math import *
# from model_training_evaluation import *
# from data_preprocess import *
from gridding import *
from prediction import *
from sklearn.metrics import accuracy_score
from training import *
from multiprocessing import Pool

# constants
data_type = {'row_id': object, 'x':np.float32, 'y':np.float32, 'accuracy':np.float32, 'time':np.int, 'place_id':np.int}
data_type_test = {'row_id': object, 'x':np.float32, 'y':np.float32, 'accuracy':np.float32, 'time':np.int}
train_columns = ["row_id","x","y","accuracy","time","place_id"]
test_columns = ["row_id","x","y","accuracy","time"]

# parameters
sample_rate = 1
path = '/Users/yetu/GitHub/predict_check_in/'
grid_size = 0.2



print 'loading data'
# raw_df = pd.read_csv(path+'train.csv', dtype=data_type)
# train_df, test_df = train_test_split(raw_df.sample(frac=sample_rate), test_size = 0.25)
# test_df.to_csv(path + 'test_df.csv', sep=',')
################
train_df = pd.read_csv(path+'train.csv', dtype=data_type)
train_df = train_df.sample(frac=sample_rate)
# test_df = pd.read_csv(path+'test.csv', dtype=data_type_test)
# test_df.to_csv(path + 'test_frac/test_df.csv', sep=',')


print 'griding'
gridding(train_df, grid_size, path + 'train_grids/', train_columns)
# gridding(test_df, grid_size, path + 'test_grids/', test_columns, False)

# print 'training and predicting by grid'
# #
# start = ''
# #
# train_grids = glob.glob(path+'train_grids/*.csv')
# test_grids = glob.glob(path+'test_grids/*.csv')
# # train_save_all_models(train_grids,test_grids, path+'model_outputs/', start)
#
# def remove_existing_files(train_grids, test_grids, path, start='', end=None):
#     i = 0
#     while i < len(train_grids):
#         train_file = train_grids[i]
#         file_name = train_file.split('.')[0].split('/')[-1]
#         if os.path.isfile(path+file_name+'.json') or file_name < start or (end != None and file_name > end):
#             del train_grids[i], test_grids[i]
#         else:
#             i += 1
# if __name__=="__main__":
#     pool = Pool(processes=3)
#     remove_existing_files(train_grids, test_grids, path+'model_outputs/')
#     n = len(train_grids)
#     args = zip(train_grids, test_grids, [path+'model_outputs/' for i in range(n)])
#     pool.map(train_save_model_wrapper, args)
#     pool.close()
#     pool.join()
#
#
#
# print 'predicting and outputing'
# output_paths = glob.glob(path+'model_outputs/*.json')
# # test_df = pd.read_csv(path+'test_df.csv', sep = ',', header= 0, index_col=0, dtype=data_type_test)
# result = predict(test_df, output_paths)
# # print accuracy_score(test_df.place_id.values, result)
# output_df = pd.DataFrame(result, columns=['place_id'])
# output_df.to_csv(path+'submision.csv', index_label='row_id')
#
#
