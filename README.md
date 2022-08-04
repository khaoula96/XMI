# XMI #

This repository contains the different experimenets we did for the purpose of the AAAI paper (Building Modular Network models through the XMI). 

PS: We do not include the XMI itself as this is a proprietary but the models and their parameters may be created from the code above.

We Mainly conducted three experimets:
1.  Data pipeline timing 
This experiment will be found under [QMI_pipeline_timing](QMI_pipeline_timing) directory. We experimented with several testing files consisting of 50M rows with different combination of categorical and continuous features (the data generation process script is in [dp.py](QMI_pipeline_timing/dgp.py) ) and compared the QMI (Queue Model Interface), the Tensorflows One Shot Iterator [tf_records.py](QMI_pipeline_timing/tf_records.py) and the Pytorchs Dataloader (QMI_pipeline_timing/pytorch.py) timings as we cycle through the data.

2.  Electrical demand model vs prophet: 
The electrical demand data can be found under [elect_data.zip](Elect_model_vs_Prophet/elect_data.zip) and the python script used to generate the 10 models can be found in [xmi_run_elect.py](Elect_model_vs_Prophet/xmi_run_elect.py). The output for the 10 trained models will be found in [stlf_models.zip] (Elect_model_vs_Prophet/stlf_models.zip) 
PS: each directory corresponds to the output of a certain model, for example stlf1 is for the output of model 1, stlf2 is for model 2, etc... 

These output folders will be including:
* the model coefficients (for example stlf1/coeffs/narx1_W.csv contains the weight matrix for the narx1 which is node 1)

* The structure (the structure of the model is shown as a graph like stlf1/graph.pfd)

* The output of each node of the model (for example stlf1/node_data/narx1.csv shows the output of the narx1 node in the model) 

* The predictions (stlf1/predictions.hdf5)

* The forecasts (stlf1/forecasts.hdf5)  

and so the models can be recreated exactly. In addition, the Prophet model and timings code used is under "Elect_model_vs_Prophet/train_prophet_models.py".  

3. 3rd experiment: Kernel models: The evolution of the training algorithm are illustrated in the mp4 video located under kernel_models directory. The output folder created by the xmi (using run_3_kernels.py script ) is located under "kernel_models/kernel" and the model specification file is located in there as xmi.cfg.
