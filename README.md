# XMI

### This repository contains the different experimenets we did for the purpose of the CIKM paper (Building Modular Network models through the XMI). 
### We do not include the XMI itself as this is a proprietary.

#### We Mainly conducted three experimets:
##### 1st experiment: Data pipeline timing (This experiment will be found under "QMI_pipeline_timing" directory):
###### We experienced with several testing files consisting of 50M rows with different combination of categorical and continuouis features (dgp.py) and compared the QMI (Queue Machine Interface), the Tensorflows One Shot Iterator (tf_records.py) and the Pytorchs Dataloader (pytorch.py) timings as we cycl through the data.

##### 2nd experiment: Electrical demand model vs prophet:
###### the electrical demand data and the python script used to generate the 10 models can be found under the directory "Elect_model_vs_Prophet", which they contain the XMI input located under "Elect_model_vs_Prophet/elect_data" and output for the 10 trained models (stlf1 for model 1, stlf2 for model 2, etc...) including the model coefficients (for example stlf1/coeffs/narx1_W.csv contains the weight matrix for the narx1 which is node 1), structure (the structure of the model is saved under stlf1/xmi.cfg and could be shown as a graph like stlf1/graph.pfd) , output of each node of the model (for example stlf1/node_data/narx1.csv shows the output of the narx1 node in the model) and the predictions (stlf1/predictions.hdf5) and forecasts (stlf1/forecasts.hdf5)  and so the models can be recreated exactly. In addition, the Prophet model and timings code used , 

