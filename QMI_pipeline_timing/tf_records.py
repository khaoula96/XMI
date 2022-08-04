# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 21:28:22 2022

@author: kkadri
"""

import tensorflow as tf
import time
import h5py
import numpy as np
import os

def read_hdf5(path):

    sets_to_read = [
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
    
    hdf5 = h5py.File(path, "r")
    r = {s: np.array(hdf5[s]) for s in sets_to_read}
    hdf5.close()
    return r

# Convert each feature into a tensorflow feature object

def _float_feature(array):
    """ Returns a float list from a float / double. """
    return tf.train.Feature(float_list=tf.train.FloatList(value=array))

def _int_feature(array):
    """ Returns an int64 list from a bool / enum / int/uint."""

    return tf.train.Feature(int64_list=tf.train.Int64List(value=array))

def convert_to(directory, dataset_name):

    files = sorted([os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.hdf5')])

    filename = os.path.join(directory, dataset_name + '.tfrecords')
    print('Writing', filename)

    with tf.io.TFRecordWriter(filename) as writer:

        
        for file in files:
            try:
                data = read_hdf5(file)
            except OSError:
                print("Could not read {}. Skipping.".format(file))
                continue

            
            item = data['unhot_X_product_item']
            price=data['as_is_X_price']
            location= data['unhot_X_location_store']
            doy=data["unhot_X_doy"]
            dow=data["unhot_X_calendar_day_of_week"]
            moy=data['unhot_X_calendar_month_of_year']
            color=data["unhot_X_product_color_group_code"]
            avr_price=data["as_is_X_avr_price"]
            avr_discount=data["as_is_X_avr_discount"]
            vel=data["unhot_X_velocity_quantile"]
            hol=data["unhot_X_calendar_holiday"]
            cat=data["unhot_X_category"]
            a=data["as_is_X_a"]
            c= data["as_is_X_c"]
            d= data["as_is_X_d"]
            f=data["as_is_X_f"]
            g=data["as_is_X_g"]
            
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        
                        "item": _int_feature(item.flatten()),
                        "price": _float_feature(price.flatten()),
                        "location": _int_feature (location.flatten()),
                        "doy": _int_feature(doy.flatten()),
                        "dow": _int_feature(dow.flatten()),
                        "moy": _int_feature(moy.flatten()),
                        'color': _int_feature(color.flatten()),
                        "avr_price": _float_feature(avr_price.flatten()),
                        "avr_disc": _float_feature(avr_discount.flatten()),
                        "vel": _int_feature(vel.flatten()),
                        "hol": _int_feature(hol.flatten()),
                        "cat": _int_feature(cat.flatten()),
                        "a": _float_feature(a.flatten()),
                        "c": _float_feature(c.flatten()),
                        "d": _float_feature(d.flatten()),
                        "f": _float_feature(f.flatten()),
                        "g":_float_feature(g.flatten()),
                       
                    }
                )
            )
            writer.write(example.SerializeToString())


if __name__ == '__main__':
    train_tf=convert_to(directory=r"path" , dataset_name= 'train_tf')
    BATCH_SIZE=5000
    print("start")
    start= time.time()
    dataset= tf.data.TFRecordDataset(["train_tf.tfrecords"]).batch(BATCH_SIZE, drop_remainder=False).prefetch(buffer_size=tf.data.AUTOTUNE)
    for element in dataset:
        print(element)
    print("finish")
    delta=time.time()- start
    print("delta", delta)
   
    