# XMI #

This repository contains the different experimenets we did for the purpose of the AAAI paper (Building Modular Network models through the XMI). 

PS: We do not include the XMI itself as this is a proprietary but the models and their parameters may be created from the code above.

We Mainly conducted three experimets:

### 1st Experiment ###
1.  Data pipeline timing: This experiment will be found under [QMI_pipeline_timing](QMI_pipeline_timing) directory. We experimented with several testing files consisting of 50M rows with different combination of categorical and continuous features (the data generation process script is in [dgp.py](QMI_pipeline_timing/dgp.py) ) and compared the QMI (Queue Model Interface), the Tensorflows One Shot Iterator [tf_records.py](QMI_pipeline_timing/tf_records.py) and the Pytorchs Dataloader [pytorch_dataloader.py](QMI_pipeline_timing/pytorch_dataloader.py) timings as we cycle through the data.

#### How to run the Data pipeline timing experiment ####

The following steps should allow you to run the experiment locally.

Install these requirements:

     Python
     h5py
     numpy
     tensorflow
     torch

Open an IDE (for example Spider).

Set the path of the data that will be generated in [dgp.py](QMI_pipeline_timing/dgp.py).

Set the path of the generated data that will be used in [tf_records.py](QMI_pipeline_timing/tf_records.py) and [pytorch_dataloader.py](QMI_pipeline_timing/pytorch_dataloader.py).

Run the files and the results of the timing will be shown.

### 2nd Experiment ###

2.  Electrical demand model vs prophet: The electrical demand data can be found under [elect_data.zip](Elect_model_vs_prophet/elect_data.zip) and the python script used to generate the 10 models can be found in [xmi_run_elec.py](Elect_model_vs_prophet/xmi_run_elec.py). The output for the 10 trained models will be found under [QMI_pipeline_timing](QMI_pipeline_timing) directory. Each output directory corresponds to the output of a certain model, for example stlf1 is for the output of model 1, stlf2 is for model 2, etc... 

These output folders will be including:
* the model coefficients (for example 'stlf1/coeffs/narx1_W.csv' contains the weight matrix for the narx1 which is node 1)

* The structure (the structure of the model is shown as a graph like 'stlf1/graph.pdf')

* The output of each node of the model (for example 'stlf1/node_data/narx1.csv' shows the output of the narx1 node in the model) 

* The predictions ('stlf1/predictions.hdf5')

* The forecasts ('stlf1/forecasts.hdf5')  

and so the models can be recreated exactly. In addition, the Prophet model and timings code used is in [train_prophet_models.py](Elect_model_vs_prophet/train_prophet_models.py).

### 3rd Experiment ###

3.  Kernel models: The evolution of the training algorithm are illustrated in the mp4 video located in [Kernel_models](Kernel_models) directory. The output folder created by the xmi using [run_3_kernels.py](Kernel_models/run_3_kernels.py) script  is located in  [kernel3.zip](Kernel_models/kernel3.zip) and the model specification file is located in there as xmi.cfg.
