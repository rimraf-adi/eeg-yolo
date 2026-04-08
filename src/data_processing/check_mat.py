import scipy.io as sio
import numpy as np

mat_path = "/Volumes/WORKSPACE/opensource-dataset/MAT_Files/DA00100S.mat"
mat = sio.loadmat(mat_path)

events = mat['events']
print(events)
