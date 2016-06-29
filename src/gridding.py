import numpy as np
import pandas as pd
from math import ceil, floor

def intersection_area(bnd1, bnd2):
    """
    intersection area of two rectangular
    bnd = [x_l, x_u, y_l, y_u] specifies a rectangular
    """
    x_overlap = max(0, min(bnd1[1], bnd2[1]) - max(bnd1[0], bnd2[0]))
    y_overlap = max(0, min(bnd1[3], bnd2[3]) - max(bnd1[2], bnd2[2]))
    return x_overlap * y_overlap


def record_cell_intersection(record, grid_size, error=True):
    """
    for each record, compute the cells and weight it intersects with
    a cell is represented by the index of its top-left corner
    X, Y -- array of cells
    weights -- array of weights computed by the portion of intersection in the error box
    """
    X, Y, weights = [], [], []
    if error:
        err = record.accuracy / 10000 # location error
    else:
        err = 1e-8
    x_l, x_u = max(1e-8, record.x - err), min(10-1e-8, record.x + err) 
    y_l, y_u = max(1e-8, record.y - err), min(10-1e-8, record.y + err)
    bnd_err = [x_l, x_u, y_l, y_u]
    bnd_whole = [0, 10, 0, 10]
    for x in range(int(floor(x_l/grid_size)), int(ceil(x_u/grid_size))):
        for y in range(int(floor(y_l/grid_size)), int(ceil(y_u/grid_size))):
            bnd_cell = [x * grid_size, (x+1) * grid_size, y * grid_size, (y+1) * grid_size]
            wt = intersection_area(bnd_err, bnd_cell) / intersection_area(bnd_err, bnd_whole)
            if wt > 0.1:
                weights.append(wt)
                X.append(x)
                Y.append(y)
    return X, Y, weights

"""
associate each record to cells it intersect with 
save .csv for each cell
attributes are x, y, accuracy, time, place_id, weight
"""
import os
def gridding(df, grid_size, path, columns, error=True):
    n = int(ceil(10 / grid_size))
    # 4d array, grid[i][j] contains data for cell (i,j) 
    grid = [[ [] for i in range(n)] for j in range(n)]
    for i in range(0, len(df)):
        if i % 1000000 == 0:
            print str(i / 1000000) + 'M records gridded'
        curr = df.iloc[i]
        X, Y, weights = record_cell_intersection(curr, grid_size, error)
        for x, y, w in zip(X, Y, weights):
            grid[x][y].append(curr[columns].values.tolist() + [w])
            
    
    if not os.path.exists(path):
        os.makedirs(path)
    for x in range(n):
        for y in range(n):
            temp_df = pd.DataFrame(grid[x][y], columns = columns + ['weight'])
            temp_df.to_csv(path + 'grid_' + str(x) + '_' + str(y)+ '.csv', sep=',')
    return

