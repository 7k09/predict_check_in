from gridding import record_cell_intersection
import json
import operator


def combine_prob(full_prob, curr_prob):
    """
    prob: place_id -> prob
    """
    for place_id in curr_prob:
        if not full_prob.has_key(place_id):
            full_prob[place_id] = 0
        full_prob[place_id] += curr_prob[place_id]
    return full_prob

def add_dicts(full_dict, curr_dict):
    """
    dict: row_id -> place_id -> prob    
    """
    for row_id in curr_dict:
        if not full_dict.has_key(row_id):
            full_dict[row_id] = {}
        full_dict[row_id] = combine_prob(full_dict[row_id], curr_dict[row_id])
    return

def load_results(output_paths):
    full_dict = {}
    for file_name in output_paths:
        with open(file_name, 'r') as fid:
            curr_dict = json.load(fid)
        add_dicts(full_dict, curr_dict)
            
    return full_dict


def predict(test_df, output_paths):
    count = 0
    full_dict = load_results(output_paths)
    output = []
    for row_id in test_df.row_id.values:
        row_id_str = unicode(row_id, "utf-8")
        if not full_dict.has_key(row_id_str):
            print 'no records: ' + row_id_str
            output.append('')
        sorted_probs = sorted(full_dict[row_id_str].items(), key=operator.itemgetter(1), reverse=True)
        string = ''
        for i in range(min(len(sorted_probs), 3)):
            string += ' ' + sorted_probs[i][0]
        output.append(string[1:])
    return output
