# MAcPNN
This repository contains the code used for the experimentation shown in the paper that will be presented at IEEE Internation Conference on Big Data 2024 .

## 1) Installation
execute:

`conda create -n env python=3.8`

`conda activate env`

`pip install -r requirements.txt`

## 2) Project structure
The project is composed of the following directories.
#### datasets
Download `datasets.zip` [here](https://polimi365-my.sharepoint.com/:f:/g/personal/10780444_polimi_it/EvTEYrtfJ7ZAiMjb7NYUXJUB4J2sogaM9KkQK1iZAMQdiw?e=mZonOb)
Extract it in the main folder of the project.
It contains the generated data streams.
Each file's name has the following structure: **\<generator\>\_federated\_\<id_configuration\>conf\_node\<id_device\>\_\<id_concept\>task_\<start\>.csv**.

<ins>Generators:</ins>
* sine: SRW.
* weather: Weather
* air_quality: AirQuality.

For each configuration, the data stream's concepts of each device are split into different csv files.\
Concept IDs are incremental and starts from 0.\
In case of devices 1 and 2, the first concept's data stream (task0) is split in two parts to start the following device at the end of the first concept's initial part. The initial part is indicated as "task0_start".

#### models
It contains the python modules implementing cPNN, cLSTM and MAcPNN.
### evaluation
It contains the python modules to implement the prequential evaluation used for the experiments.
#### data_gen
It contains the python modules implementing the data stream generator.

## 3) Evaluation
#### evaluation/test_ma.py
It runs the prequential evaluation using the specified configurations. Change the variables in the code for different settings (see the code's comments for the details).

Run it with the command `python -m evaluation.test_ma`.

The execution stores the pickle files containing the results in the folder specified by the variable `PATH_PERFORMANCE`. For the details about the pickle files, see the documentation in **evaluation/prequential_evaluation.py**.

#### evaluation/build_perf_table.py
After executing the experiments with `test_ma.py`, it builds the performance table and write it as csv and xlsx.\
The column **T\<i\>\_\<metric\>\_\<model\>** represents the metric (start, end) associated with concept i.\
The column **\<metric\>\_\<model\>** represents the metric (start, end) averaged on the concepts (the first concept is excluded).\
Run it with the command `python -m evaluation.build_perf_table`.

#### evaluation/build_perf_plot.py
After executing the experiments with `test_ma.py`, it builds the performance files to make the plots in Fig. 7.
Run it with the command `python -m evaluation.build_perf_plot`.

#### evaluation/plot.py
After making the files with `build_perf_plot.py`, it makes the plots in Fig. 7 and saves it in the `performance/macpnn` folder.\
Run it with the command `python -m evaluation.plot`.




