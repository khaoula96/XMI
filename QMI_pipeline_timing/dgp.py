# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 13:53:38 2022

@author: kkadri
"""

import os
import numpy as np
import h5py


def create_syn_data(size=100000):
    np.random.seed(0)
    """y =  (a*sku_id + b store_id
    + c sku&store + ..."""
    N = size
    sku_ncat = 20
    store_ncat = 10
    color_ncat = 3
    sku_id = np.random.randint(0, sku_ncat, size=(N,))
    price = np.random.normal(1000, 2, size=(N,))
    store_id = np.random.randint(0, store_ncat, size=(N,))
    color = np.random.randint(0, color_ncat, size=(N,))
    dow = np.random.randint(0, 7, size=(N,))
    month = np.random.randint(0, 12, size=(N,))
    holiday = np.random.randint(0, 41, size=(N,))
    doy = np.random.randint(0, 366, size=(N,))
    velocity = np.random.randint(0, 4, size=(N,))
    
    avr_price = np.random.normal(2000, 3, size=(N,))
    avr_discount = np.random.normal(10, 3, size=(N,))
   
    category = np.random.randint(0, 1000, size=(N,))
    style = np.random.randint(0, 100, size=(N,))
    season = np.random.randint(0, 4, size=(N,))
    
    a= np.random.normal(10,5, size=(N,))
    b= np.random.randint(0, 500, size=(N,))
    c= np.random.normal(5000, 2, size=(N,))
    d= np.random.normal(9, 3, size=(N,))
    f= np.random.normal(5, 0.1, size=(N,))
    g= np.random.normal(90, 4, size=(N,))
    h= np.random.randint(0, 76, size=(N,))
    s= np.random.randint(0, 30, size=(N,))
    j= np.random.randint(0, 40, size=(N,))

    k= np.random.normal(50, 1, size=(N,))
    l= np.random.normal(10, 0.2, size=(N,))
    m= np.random.randint(0, 44, size=(N,))
    o= np.random.randint(0, 8, size=(N,))
    p= np.random.normal(0, 4, size=(N,))
    q= np.random.randint(0, 7, size=(N,))
    r= np.random.randint(0, 10, size=(N,))

    sku_store_weights = np.random.rand(
        sku_ncat, store_ncat).astype(np.float32) * 10
    sku_weights = (
        np.random.rand(
            sku_ncat,
        )
        * 10
    )

    store_weights = (
        np.random.rand(
            store_ncat,
        )
        * 10
    )

    color_weights = (
        np.random.rand(
            color_ncat,
        )
        * 2
    )

    doy_weights = (
        np.random.rand(
            366,
        )
        * 2
    )

    dow_weights = (
        np.random.rand(
            7,
        )
        * 120
    )
    month_weights = (
        np.random.rand(
            12,
        )
        * 10
    )
    holiday_weights = (
        np.random.rand(
            41,
        )
        * 100
    )

    y = np.zeros((N,))
    for i in range(N):
        y[i] = (
            sku_weights[sku_id[i]]
            + color_weights[color[i]]
            + dow_weights[dow[i]]
            + month_weights[month[i]]
            + store_weights[store_id[i]]
            + sku_store_weights[sku_id[i]][store_id[i]]
            + holiday_weights[holiday[i]]
            + doy_weights[doy[i]]
        )
    data = {}
    data["y_out"] = y.reshape((-1, 1))
    data["unhot_X_product_item"] = sku_id.reshape((-1, 1))
    data["as_is_X_price"] = price.reshape((-1, 1))
    data["unhot_X_location_store"] = store_id.reshape((-1, 1))
    data["unhot_X_product_color_group_code"] = color.reshape((-1, 1))
    data["unhot_X_calendar_day_of_week"] = dow.reshape((-1, 1))
    data["unhot_X_calendar_month_of_year"] = month.reshape((-1, 1))
    data["unhot_X_calendar_holiday"] = holiday.reshape((-1, 1))
    data["unhot_X_doy"] = doy.reshape((-1, 1))
    data["unhot_X_velocity_quantile"] = velocity.reshape((-1, 1))
    data["as_is_X_avr_price"]= avr_price.reshape((-1, 1))
    data["as_is_X_avr_discount"]= avr_discount.reshape((-1, 1))
    data["unhot_X_category"]= category.reshape((-1, 1))
    data["unhot_X_style"]= style.reshape((-1, 1))
    data["unhot_X_season"]= season.reshape((-1, 1))
    
    data["as_is_X_a"]= a.reshape((-1, 1))
    data["unhot_X_b"]= b.reshape((-1, 1))
    data["as_is_X_c"]= c.reshape((-1, 1))
    data["as_is_X_d"]= d.reshape((-1, 1))
    data["as_is_X_f"]= f.reshape((-1, 1))
    data["as_is_X_g"]= g.reshape((-1, 1))
    data["unhot_X_h"]= h.reshape((-1, 1))
    data["unhot_X_s"]= s.reshape((-1, 1))
    data["unhot_X_j"]= j.reshape((-1, 1))
    data["as_is_X_k"]= k.reshape((-1, 1))
    data["as_is_X_l"]= l.reshape((-1, 1))
    data["unhot_X_m"]= m.reshape((-1, 1))
    data["unhot_X_o"]= o.reshape((-1, 1))
    data["as_is_X_p"]= p.reshape((-1, 1))
    data["unhot_X_q"]= q.reshape((-1, 1))
    data["unhot_X_r"]= r.reshape((-1, 1))

    data["data_names"] = [
        "y_out",
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
    
    data["data_shapes"] = [
        data["y_out"].shape[1],
        sku_ncat,
        data["as_is_X_price"].shape[1],
        store_ncat,
        12,
        7,
        color_ncat,
        41,
        366,
        4,
        data["as_is_X_avr_price"].shape[1],
        data["as_is_X_avr_discount"].shape[1],
        1000,
        100,
        4,
        data["as_is_X_a"].shape[1],
        500,
        data["as_is_X_c"].shape[1],
        data["as_is_X_d"].shape[1],
        data["as_is_X_f"].shape[1],
        data["as_is_X_g"].shape[1],
        76,
        30,
        40,
        data["as_is_X_k"].shape[1],
        data["as_is_X_l"].shape[1],
        44,
        8,
        data["as_is_X_p"].shape[1],
        7,
        10
       
    ]

    return data
 
def create_hdf5(inputs, filename):

    with h5py.File(filename, "w") as f:
        
        f.create_dataset("y_out", data=inputs["y_out"], dtype=np.float32)
                
        f.create_dataset("data_names", data=inputs["data_names"])      
        
        f.create_dataset("data_shapes", data=inputs["data_shapes"])    
        
        f.create_dataset("unhot_X_product_item",data=inputs["unhot_X_product_item"])
        
        f.create_dataset("as_is_X_price", data=inputs["as_is_X_price"], dtype=np.float32) 

        
        f.create_dataset("unhot_X_location_store", data=inputs["unhot_X_location_store"] )
        
        f.create_dataset("unhot_X_calendar_month_of_year", data=inputs["unhot_X_calendar_month_of_year"])
        
        f.create_dataset("unhot_X_product_color_group_code",data=inputs["unhot_X_product_color_group_code"])
        
        f.create_dataset("unhot_X_calendar_day_of_week", data=inputs["unhot_X_calendar_day_of_week"])
        
        f.create_dataset(
            "unhot_X_calendar_holiday", data=inputs["unhot_X_calendar_holiday"]
        )
        f.create_dataset("unhot_X_doy", data=inputs["unhot_X_doy"])
        f.create_dataset(
            "unhot_X_velocity_quantile", data=inputs["unhot_X_velocity_quantile"]
        )
        f.create_dataset(
            "as_is_X_avr_price", data=inputs["as_is_X_avr_price"], dtype=np.float32
        )
        f.create_dataset(
            "as_is_X_avr_discount", data=inputs["as_is_X_avr_discount"], dtype=np.float32
        )
        f.create_dataset(
            "unhot_X_category", data=inputs["unhot_X_category"]
        )
        f.create_dataset(
            "unhot_X_style", data=inputs["unhot_X_style"]
        )
        f.create_dataset(
            "unhot_X_season", data=inputs["unhot_X_season"]
        )
        
        f.create_dataset(
            "as_is_X_a", data=inputs["as_is_X_a"], dtype=np.float32
        )
        f.create_dataset(
            "unhot_X_b", data=inputs["unhot_X_b"]
        )
        f.create_dataset(
            "as_is_X_c", data=inputs[ "as_is_X_c"], dtype=np.float32
        )
        f.create_dataset(
            "as_is_X_d", data=inputs["as_is_X_d"], dtype=np.float32
        )
        f.create_dataset(
            "as_is_X_f", data=inputs["as_is_X_f"], dtype=np.float32
        )
        f.create_dataset(
            "as_is_X_g", data=inputs["as_is_X_g"], dtype=np.float32
        )
        f.create_dataset(
            "unhot_X_h", data=inputs["unhot_X_h"]
        )
        f.create_dataset(
            "unhot_X_s", data=inputs["unhot_X_s"]
        )
        f.create_dataset(
            "unhot_X_j", data=inputs["unhot_X_j"]
        )
        f.create_dataset(
            "as_is_X_k", data=inputs["as_is_X_k"], dtype=np.float32
        )
        f.create_dataset(
           "as_is_X_l", data=inputs["as_is_X_l"], dtype=np.float32
        )
        f.create_dataset(
            "unhot_X_m", data=inputs["unhot_X_m"]
        )
        f.create_dataset(
            "unhot_X_o", data=inputs["unhot_X_o"]
        )
        f.create_dataset(
            "as_is_X_p", data=inputs["as_is_X_p"], dtype=np.float32
        )
        f.create_dataset(
            "unhot_X_q", data=inputs["unhot_X_q"]
        )
        f.create_dataset(
            "unhot_X_r", data=inputs["unhot_X_r"]
        )
        
        
        f.close()


def generate_data(path, train_size=50000000):

    dt = create_syn_data(size=train_size)
    create_hdf5(
        dt,
        os.path.join(path, "train_tf.hdf5"),
    )


if __name__ == "__main__":
    #generate_data("path")
    pass
