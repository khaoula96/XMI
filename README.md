# XMI #

This repository contains the different experimenets we did for the purpose of the AAAI paper (Building Modular Network models through the XMI). 

PS: We do not include the XMI itself as this is a proprietary but the models and their parameters may be created from the code above.

We Mainly conducted three experimets:

### 1st Experiment: Data pipeline timing ###
This experiment will be found under [QMI_pipeline_timing](QMI_pipeline_timing) directory. 
We experimented with several testing files consisting of 50M rows with different combination of categorical and continuous features (the data generation process script is in [dgp.py](QMI_pipeline_timing/dgp.py) ) and compared the QMI (Queue Model Interface), the Tensorflows One Shot Iterator [tf_records.py](QMI_pipeline_timing/tf_records.py) and the Pytorchs Dataloader [pytorch_dataloader.py](QMI_pipeline_timing/pytorch_dataloader.py) timings as we cycle through the data.

#### How to run the Data pipeline timing experiment ####

The following steps should allow you to run the experiment locally.

1. Install these requirements:

        Python
        h5py
        numpy
        tensorflow
        torch

2. Open an IDE (for example Spider IDE).

3. Set the path of the data that will be generated in [dgp.py](QMI_pipeline_timing/dgp.py).

4. Set the path of the generated data that will be used in [tf_records.py](QMI_pipeline_timing/tf_records.py) and [pytorch_dataloader.py](QMI_pipeline_timing/pytorch_dataloader.py).

5. Run the files and the results of the timings will be shown in the console.

### 2nd Experiment: Electrical demand model###

This experiment will be found under [Elect_model_vs_prophet](Elect_model_vs_prophet). In this exepriment we used  [xmi_run_elec.py](Elect_model_vs_prophet/xmi_run_elec.py) python script with the electrical demand data that will be found in  [elect_data.zip](Elect_model_vs_prophet/elect_data.zip) to generate 10 models, each one has it's own structure and configuration.

The output of the 10 trained models will be found under [Elect_model_vs_prophet](Elect_model_vs_prophet) directory.  Each zip file contains the output of a certain model, for example `stlf1.zip` contains the output of model 1, `stlf2.zip` is for model 2, etc... 

Each zip file contains the following subfiles:

1. The model configuration (for example [stlf.cfg](Elect_model_vs_prophet/stlf.cfg)): It contains the different parts that composes the structure of the model, from specifying the train and test datasets, defining the general parameters of the model(learning rate, number of iterations, loss function) to specifying the computational nodes like `node1` which takes an embedding layer and apply it for the `location` categorical feature. 

This model configuration file will be created in every zip file under the name of `xmi.cfg`.

2. The structure: the structure of the model is shown as a pdf graph. For example the structure of the first model `stlf1/graph.pdf` is shown as ![stlf1](Elect_model_vs_prophet/plots/stlf1.pdf). The structure of the sixth model `stlf6/graph.pdf` is shown as ![stlf6](Elect_model_vs_prophet/plots/stlf6.pdf).

2. The model coefficients: Which represents the trained parameters for a model. For example `stlf1/coeffs/narx1_W.csv` file contains the weight matrix `W` for the `node1` which is named `narx1` for the first model `stlf1`.

4. The output of each node of the model: For example we have `stlf1/node_data/narx1.csv` file which contains the output of `narx1` node for the first model `stlf1`

5. The predictions: For example we  have `stlf1/predictions.hdf5` file which contains the final output of the model `stlf1` for the train data.

6. The forecasts: This file for example `stlf1/forecasts.hdf5` containes the final output of the model `stlf1` for the test data.

and so the models can be recreated exactly. 

In addition, the Prophet model and timings code used is in [train_prophet_models.py](Elect_model_vs_prophet/train_prophet_models.py).

### 3rd Experiment: Kernel models ###

The evolution of the training algorithm are illustrated in an mp4 video located in [Kernel_models](Kernel_models) directory. The output folder created by the xmi using [run_3_kernels.py](Kernel_models/run_3_kernels.py) script  is located in  [kernel3.zip](Kernel_models/kernel3.zip) and the model specification file is located in there as `xmi.cfg`.
