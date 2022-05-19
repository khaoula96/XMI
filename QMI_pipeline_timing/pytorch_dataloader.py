# -*- coding: utf-8 -*-
"""
Created on Thu May 12 17:26:59 2022

@author: kkadri
"""

import h5py
import time
import numpy as np
from torch.utils.data import DataLoader, Dataset



class H5Dataset(Dataset):
    def __init__(self, h5_path, chunk_size, nb_samples):
        self.h5_file = h5py.File(h5_path, "r")
        self.chunk_size = chunk_size
        self.len = nb_samples // self.chunk_size
        

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        
        dataset = self.h5_file
        c_start = self.chunk_size * index
        c_end = self.chunk_size * (index + 1) -1
        inp=[]

        keys = [
        "unhot_X_product_item",
        "as_is_X_price",
        "unhot_X_location_store",
        "unhot_X_calendar_month_of_year",
        "unhot_X_calendar_day_of_week",
        "unhot_X_product_color_group_code",
        "unhot_X_calendar_holiday",
        "unhot_X_doy",
        "unhot_X_velocity_quantile",
        "as_is_X_avr_price",
        "as_is_X_avr_discount",
        "unhot_X_category",
        "unhot_X_style",
        "unhot_X_season",
        "as_is_X_a",
        "unhot_X_b",
        "as_is_X_c",
        "as_is_X_d",
        "as_is_X_f",
        "as_is_X_g",
        "unhot_X_h",
        "unhot_X_s",
        "unhot_X_j",
        "as_is_X_k",
        "as_is_X_l",
        "unhot_X_m",
        "unhot_X_o",
        "as_is_X_p",
        "unhot_X_q",
        "unhot_X_r",
       ]
            
        for s in keys:
            r=np.array(dataset[s]) 
            data=r[c_start:c_end,:]
            inp.append(data)
        return inp

    
if __name__ == '__main__':
    
    print("start")
    start= time.time()
    loader = DataLoader(H5Dataset("train_tf.hdf5", chunk_size=5000, nb_samples=50000000))
    for step, x in enumerate(loader): 
        print(x)
    print("finish")
    delta=time.time()- start
    print("delta", delta)
   