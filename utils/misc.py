import numpy as np

def normalize(ds, axis = None):
    min_ds = np.min(ds, axis=axis)
    max_ds = np.max(ds, axis=axis)
    assert(max_ds != min_ds)
    ds_norm = (ds - min_ds)/(max_ds - min_ds)
    return ds_norm
    
