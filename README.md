# XMI

### This repository contains the different experimenets we did for the purpose of the CIKM paper (Building Modular Network models through the XMI). 
### We do not include the XMI itself as this is a proprietary.

#### We Mainly conducted three experimets:
##### 1st experiment: Data pipeline timing (This experiment will be found under "/QMI_pipeline_timing" directory):
###### We experienced with several testing files consisting of 50M rows with different combination of categorical and continuouis features (dgp.py) and compared the QMI (Queue Machine Interface), the Tensorflows One Shot Iterator (tf_records.py) and the Pytorchs Dataloader (pytorch.py) timings as we cycl through the data.
##### 2nd experiment: Electrical demand model vs prophet:
###### This experiemnt will be found under the directory "/Elect_model_vs_Prophet", which contains the XMI input and output for the models trained including the model coefficients, structure, output of each node of the model and the predictions and forecasts and so the models can be recreated exactly. In addition, the Prophet model and timings code used , the electrical demand data and the python script used to generate the 10 models can be found here. 

