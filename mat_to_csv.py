import pandas as pd
import numpy as np
import scipy.io

path_to_mat_file = 'Saved_Results/all_data-7-13-19.mat'
path_to_csv_file = 'Saved_Results/all_data-7-13-19.csv'

mat = scipy.io.loadmat(path_to_mat_file)

k = mat['k'][0][0]

pts_neg = []
pts_pos = []
pts_nuc = []
pts_pos_o = []

for i in range(11400):
    for pos in mat['pts_neg'][0][i][0]:
        pts_neg.append([i + 1, 1, pos[0], pos[1]])
    for pos in mat['pts_pos'][0][i][0]:
        pts_pos.append([i + 1, 0, pos[0], pos[1]])
    for pos in mat['pts_nuc'][0][i][0]:
        pts_nuc.append([i + 1, 3, pos[0], pos[1]])
    for pos in mat['pts_pos_o'][0][i][0]:
        pts_pos_o.append([i + 1, 2, pos[0], pos[1]])

pts_neg = np.array(pts_neg)
pts_pos = np.array(pts_pos)
pts_nuc = np.array(pts_nuc)
pts_pos_o = np.array(pts_pos_o)

data = np.vstack((pts_pos,pts_neg, pts_pos_o, pts_nuc))
df = pd.DataFrame(data, columns=['image_index', 'class', 'x', 'y'])
df = df.astype(dtype={'image_index': int, 'class': int})

# make sure you do need to convert the position of y axis!!!
df['y'] = 1 - df['y']
# ##################
df['y'] = np.minimum(1, df['y'])
df['x'] = np.minimum(1, df['x'])

test_range = np.concatenate((np.arange(7001,8001),np.arange(8234,8301),np.arange(8834,8901)))
valid_range = np.concatenate((np.arange(6501,7001),np.arange(8201,8234),np.arange(8801,8834)))

# df_test = df.loc[df['image_index'].isin(test_range)].copy()
# df_test.to_csv('Saved_Results/new_test.csv', sep=' ', index=False)
# df_train = df.loc[~df['image_index'].isin(np.concatenate(test_range, valid_range))]
# df_train = df.loc[np.logical_not(df['image_index'].isin(np.concatenate(test_range, valid_range)))].copy()
# df_train.to_csv('Saved_Results/new_train.csv', sep=' ', index=False)
df.to_csv(path_to_csv_file, sep=' ', index=False)