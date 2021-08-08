import mat73
import scipy.io
from scipy.io import loadmat
import numpy as np
import h5py

from PIL import Image

# img = Image.open(r'C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Videos\260520\BF\Experiment1_w2Brightfield_s1_all.tif')
#
# for i in range(10):#img.n_frames):
#     try:
#         img.seek(i)
#         img.save(r'C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Videos\260520\BF\sequence_1\page_%s.tif'%(i,))
#     except EOFError:
#         break

import numpy as np
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import matplotlib.pyplot as plt
from datetime import datetime, date, time
import pandas as pd
# mat_path  = r'C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Videos\260520\track10408.mat'
mat_path = r"C:\Users\Amit\Desktop\Example trackfile.mat"
# <KeysViewHDF5 ['#refs#', '#subsystem#', 'CellProperties', 'ImagePath', 'TimeStamps', 'tbl']>

data_dict = mat73.loadmat(mat_path, use_attrdict=True)
# data_dict = scipy.io.loadmat(mat_path)




# struct = data_dict['CellProperties'] # assuming a structure was saved in the .mat
mdata = data_dict  # variable in mat file
mdtype = mdata.dtype  # dtypes of structures are "unsized objects"
# * SciPy reads in structures as structured NumPy arrays of dtype object
# * The size of the array is the size of the structure array, not the number
#   elements in any particular field. The shape defaults to 2-dimensional.
# * For convenience make a dictionary of the data using the names from dtypes
# * Since the structure has only one element, but is 2-D, index it at [0, 0]
ndata = {n: mdata[n][0, 0] for n in mdtype.names}
# Reconstruct the columns of the data table from just the time series
# Use the number of intervals to test if a field is a column or metadata
columns = [n for n, v in ndata.iteritems() if v.size == ndata['numIntervals']]
# now make a data frame, setting the time stamps as the index
df = pd.DataFrame(np.concatenate([ndata[c] for c in columns], axis=1),
                  index=[datetime(*ts) for ts in ndata['timestamps']],
                  columns=columns)

import h5py
import numpy as np
arrays = {}
f = h5py.File(mat_path)
for k, v in f.items():
    arrays[k] = np.array(v)

print()