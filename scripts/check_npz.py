import numpy as np
import os

try:
    path = 'dataset_APIS/dataset/train'
    study = os.listdir(path)[0]
    slice_dir = os.listdir(os.path.join(path, study))[0]
    full_path_img = os.path.join(path, study, slice_dir, 'image.npz')
    full_path_mask = os.path.join(path, study, slice_dir, 'mask.npz')
    
    print(f"Checking {full_path_img}")
    with np.load(full_path_img) as data:
        print(f"Image keys: {data.files}")
        
    print(f"Checking {full_path_mask}")
    if os.path.exists(full_path_mask):
        with np.load(full_path_mask) as data:
            print(f"Mask keys: {data.files}")
except Exception as e:
    print(e)
