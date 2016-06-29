import numpy as np
import pandas as pd
from math import ceil, floor


"""
for each record, compute the cells and weight it intersects with
a cell is represented by the index of its top-left corner
X, Y -- array of cells
weights -- array of weights computed by the portion of intersection in the error box
"""
def record_cell_intersection(record, grid_size):
    X, Y, weights = [], [], []
    err = record.accuracy / 1000 # location error
    x_l, x_u = max(1e-8, record.x - err), min(10-1e-8, record.x + err) 
    y_l, y_u = max(1e-8, record.y - err), min(10-1e-8, record.y + err)
    for x in range(int(floor(x_l/grid_size)), int(ceil(x_u/grid_size))):
        for y in range(int(floor(y_l/grid_size)), int(ceil(y_u/grid_size))):
            X.append(x)
            Y.append(y)
            x_overlap = max(0, min(x_u, (x + 1) * grid_size) - max(x_l, x * grid_size))
            y_overlap = max(0, min(y_u, (y + 1) * grid_size) - max(y_l, y * grid_size))
            weights.append(x_overlap / err * y_overlap / err / 4)
    return X, Y, weights

"""
associate each record to cells it intersect with 
save .csv for each cell
attributes are x, y, accuracy, time, place_id, weight
"""
import os
def gridding(df, grid_size, path, columns):
    n = int(ceil(10 / grid_size))
    # 4d array, grid[i][j] contains data for cell (i,j) 
    grid = [[ [] for i in range(n)] for j in range(n)]
    for i in range(0, len(df)):
        curr = df.iloc[i]
        X, Y, weights = record_cell_intersection(curr, grid_size)
        for x, y, w in zip(X, Y, weights):
            grid[x][y].append(curr[columns].values.tolist() + [w])
            
    
    if not os.path.exists(path):
        os.makedirs(path)
    for x in range(n):
        for y in range(n):
            temp_df = pd.DataFrame(grid[x][y], columns = columns + ['weight'])
            temp_df.to_csv(path + 'grid_' + str(x) + '_' + str(y)+ '.csv', sep=',')
    return

