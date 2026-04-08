import scipy.io as sio
import scipy.io

# Only targets P001 by default, but keeping absolute limit dynamically updated
from src.config import PATHS
import os

mat_path = os.path.join(PATHS["mat_dir"], "P001.mat")
mat = sio.loadmat(mat_path)

events = mat['events']
print(events)
