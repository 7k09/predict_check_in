from gridding import *
import json
"""
aggregate predictions from different model
using the weighted sum of probabilities
"""
def prediction_agg(y_tests_array, y_probs_array, model_weights):
    # aggregate result in a dict: place_id -> probability

    if y_tests_array == []:
        return 0

    result_dict = {}
    for y_tests, y_probs, weight in zip(y_tests_array, y_probs_array,model_weights):
        for y_test, y_prob in zip(y_tests,y_probs):
            if y_prob > 0:
                if y_test not in result_dict:
                        result_dict[y_test] =  y_prob * weight
                else:
                        result_dict[y_test] += y_prob * weight
    # find maximal probability in result_dict
    max_proba, max_place_id = 0, 0
    for item in result_dict:
        if result_dict[item] > max_proba:
            max_proba = result_dict[item]
            max_place_id = item
            
    return max_place_id

"""
for each record, output all the predictions
each prediction is made by a model corresponding to a cell the record intersects with
y_tests_array
"""
def predict_record(record, grid_size, outputs_path):
    y_tests_array, y_probs_array, model_weights= [], [], []
    X, Y, weights = record_cell_intersection(record, grid_size)
    # print record.row_id
    for x, y, w in zip(X, Y, weights):
        model_weights.append(w)
        file_name = outputs_path + 'grid_' + str(x) + '_' + str(y) + '.json'
        with open(file_name, 'r') as fid:
            output_dict = json.load(fid)


        # print [row[0] for row in output_dict[unicode(str(record.row_id), "utf-8")]]
        try:
            y_tests_array.append([row[0] for row in output_dict[unicode(str(record.row_id), "utf-8")]])
            y_probs_array.append([row[1] for row in output_dict[unicode(str(record.row_id), "utf-8")]])
        except:
            print('no record for: ' + str(record.row_id))
        
    return y_tests_array, y_probs_array, model_weights


def predict(test_df, grid_size, outputs_path):
    result = []
    for i in range(len(test_df)):
        curr = test_df.iloc[i]
        y_tests_array, y_probs_array, model_weights = predict_record(curr, grid_size, outputs_path)
        curr_predict = prediction_agg(y_tests_array, y_probs_array, model_weights)
        result.append(curr_predict)
    return result
