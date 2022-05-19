# XMI

### This repository contains the different experimenets we did for the purpose of the CIKM paper (Building Modular Network models through the XMI). 
### We do not include the XMI itself as this is a proprietary.

#### We Mainly conducted three experimets:
##### 1st experiment: Data pipeline timing (This experiment will be found under "QMI_pipeline_timing" directory):
###### We experimented with several testing files consisting of 50M rows with different combination of categorical and continuous features ( the data generation process script is in  dgp.py ) and compared the QMI (Queue Machine Interface), the Tensorflows One Shot Iterator (tf_records.py) and the Pytorchs Dataloader (pytorch.py) timings as we cycle through the data.

##### 2nd experiment: Electrical demand model vs prophet:
###### The electrical demand data can be found under "Elect_model_vs_Prophet/elect_data.zip" and the python script used to generate the 10 models can be found under "Elect_model_vs_Prophet/xmi_run_elect.py. The output for the 10 trained models will be found under "Elect_model_vs_Prophet/stlf_models.zip"(each directory corresponds to the output of a certain model, for example stlf1 is for the output of model 1, stlf2 is for model 2, etc...). 

###### These output folders will be including:
######   * the model coefficients (for example stlf1/coeffs/narx1_W.csv contains the weight matrix for the narx1 which is node 1)
######   * The structure (the structure of the model is shown as a graph like stlf1/graph.pfd)
######   * The output of each node of the model (for example stlf1/node_data/narx1.csv shows the output of the narx1 node in the model) 
######   * The predictions (stlf1/predictions.hdf5)
######   * The forecasts (stlf1/forecasts.hdf5)  

###### and so the models can be recreated exactly. In addition, the Prophet model and timings code used is under "Elect_model_vs_Prophet/train_prophet_models.py".  

##### 3rd experiment: Kernel models
