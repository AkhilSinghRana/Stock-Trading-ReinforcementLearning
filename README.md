***

# Stock-Trading-ReinforcementLearning

<summary>Table of Contents</summary>

- [Stock-Trading-ReinforcementLearning](#stock-trading-reinforcementlearning)
  - [Introduction:](#introduction)
  - [Dependencies:](#dependencies)
  - [Setup Instructions](#setup-instructions)
  - [Options for main.py](#options-for-mainpy)
  - [Approach](#approach)
  - [Results:](#results)
  - [Future Work](#future-work)


## Introduction:


## Dependencies:

Complete dependencies for this project are written in requirements.txt file.

Main Dependencies:
- Jupyterlab
- yfinance , A python library to gather stock data easily
- openAI-gym, needed to create a small wrapper to create custom Stock Trading environments
- tensorflow [CPU/GPU], according to your machine, I have run this entire project on my notebook with Quadro 3000 GPU.
- Stable-Baselines , for training the agents with different algorithms
- numpy

Proposed Optionals:
- Ubuntu OS
- python3.7.5 or above
- virtualenv
- pip 20.0.2 or above

## Setup Instructions

This project was run on Ubuntu environment, this is not a dependency, but Windows machine was not tested(It would most probably still work on windows). Feel free to contact me for any troubleshooting.

Proposed steps for a virtualenvironment Setup:

```shell
Step1:
virtualenv env_name -p python3 

Step2:
source env_name/bin/activate

Step3:
pip install -e .
```

Now you should be ready to start training the models provided with this repository. You can run the main.py file, or simply browse throguh the provided notebooks. Below you find a sample training command in case you like CLIs.

Sample Train command:
	
- python main.py --mode train 

Run Sample Jupyter Notebook --> [Jupyter Notebook](./notebooks/yFinance_libtest.ipynb)
- jupyter lab 


## Options for main.py
To check all the options that are provided for flexibilty run below command and feel to play around with these parameters. TO achieve the results run the default parameters.

-	python main.py --help


## Approach




## Results:

## Future Work

