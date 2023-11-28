# DS-Viz: A Method for Visualising Design Spaces

## Method overview
![DS_Viz](https://github.com/epz0/DS_Viz/blob/main/data/images/DS_Viz.png)

## Retrieving the dataset
To use the scripts linked here you first need to download the dataset used in the paper which is available on [Apollo](https://doi.org/10.17863/CAM.104076). 
Once you have downloaded it, save the file on the data directory within the DS_Viz folder (e.g., C:/Users/.../DS_Viz/data)

## Installing the environment
### Windows
You can install the environment on a Windows machine from the [DS_Viz_Env.yml](https://github.com/epz0/DS_Viz/blob/main/env/DS_Viz_Env.yml) file. To install the environment on via Conda you can run the following code:

```
$ conda create --name <env> --file DS_Viz_Env.yml
```

It will install all packages used with the exception of the gower package that has to be installed via pip:
```
$ pip install gower
```
### Linux
A Linux based environment will be created and made available here. In the meantime one can use the list of packages in the [DS_Viz_Env.yml](https://github.com/epz0/DS_Viz/blob/main/env/DS_Viz_Env.yml) to create an environment from scratch. 

## Running the scripts
After installing the environments you can run the [interactive_run.py](https://github.com/epz0/DS_Viz/blob/main/scripts/interactive_run.py) or [test_run.py](https://github.com/epz0/DS_Viz/blob/main/scripts/test_run.py) to check if the installation was successful.
