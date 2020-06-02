import numpy as np
import os, random, math, sys, signal
from shutil import copyfile
from importlib import reload
from envs import stockTradingEnv as tradingEnv
reload(tradingEnv)

# for debugging
#import ptvsd
#ptvsd.enable_attach(log_dir= os.path.dirname(__file__))
#ptvsd.wait_for_attach(timeout=15)

#Importing Stable Baselines
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy, CnnPolicy, CnnLnLstmPolicy,FeedForwardPolicy
from stable_baselines.common import make_vec_env # For multiprocessing support
from stable_baselines.common.vec_env import VecFrameStack, VecEnv, DummyVecEnv

########Import Custom CNN POlicy
#from customPolicy import CustomPolicy
# from util.customPolicy import CustomPolicy
# from util import customCallbacks as ccb


import tensorflow as tf
from util import options
# ===================== Base directory for the module ===================== =#
BASE_PATH = os.path.dirname(os.path.realpath(__file__))


def save_training_setup(save_path=".", file_list=[], exp_name=""):
    os.makedirs(save_path, exist_ok=True)
    file_list= file_list.split(",")
    for s_file in file_list:
        #Create a folder if needed only 1 level folder is supported at the moment
        folder_name = s_file.split("/")
        if len(folder_name)>1:
                os.makedirs(os.path.join(save_path,folder_name[0]), exist_ok=True)
        copyfile(os.path.join(BASE_PATH, s_file), os.path.join(save_path, s_file))


#Create a function and define what it shall do
###  as an example below function computes the difference between open to close values for every timestamp
request_params = ["Open","Close","Volume"] # Define the Columns/Parameters that the environment should pass to the function

def Open2Close(params=None):
        """
        -params is a placeholder, which is waiting to be filled by the environment!
        -params is the dictionary of list, with the key names equal to the request_params name
        #Asssumptions, the function name is choosen as the new column name in the pandas dataframe
        # You need to take care of how to handle the passed data in this function
        """
        open = params["Open"]
        close = params["Close"]

        return close-open

def OpenPlusClose(params=None):
        """
        -params is a placeholder, which is waiting to be filled by the environment!
        -params is the dictionary of list, with the key names equal to the request_params name
        #Asssumptions, the function name is choosen as the new column name in the pandas dataframe
        # You need to take care of how to handle the passed data in this function
        """
        open = params["Open"]
        close = params["Close"]
        
        return close+open

ext_func_list = [Open2Close, OpenPlusClose]

def train(args):
        #Using Stable Baselines
        """
        Train the algorithm (with a given policy)
        """
        
        env_info = {"args":args, "external_func":ext_func_list, "params":request_params}
        env = make_vec_env(tradingEnv.TradingEnvironment, n_envs=args.num_envs, env_kwargs={"env_info": env_info})
        #env = VecFrameStack(env, n_stack = 4)
        #Uncomment to enable visualizations!
        print("Vectorized env created")
        print("Creating model") 
        
        #Constants for saving logs and models
        exp_name = args.exp_name
        save_dir = os.path.join(BASE_PATH,'logs_models', exp_name)
        
        # Create PPO2 model now
        model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=save_dir, full_tensorboard_log=False)
        
        # Train the model and save the results
        print("Training the network")
        try:
            #train for first 1 million Epochs
            steps_per_batch, num_envs = model.n_steps, env.num_envs
            
            model.learn(total_timesteps=args.num_epochs, tb_log_name=exp_name, log_interval=10)
            
            print("First training done")
            model.save(save_path=os.path.join(BASE_PATH,'logs_models',exp_name,exp_name+'_finished'))
                
        except Exception as e:
                print("Exception occured during training", e)
                model_name = os.path.join(save_dir, "PPO2_error")
                model.save(model_name)
                import traceback
                traceback.print_exc() 

        
        print("model saved")

        return
        

# TODO        
def continueTrain():
        pass

def test(args):
        
        print("testing the trained environment")
        
        env_info = {"args":args, "external_func":ext_func_list, "params":request_params}
        env = make_vec_env(tradingEnv.TradingEnvironment, n_envs=args.num_envs, env_kwargs={"env_info": env_info})
        
        
        #Constants for saving logs and models
        exp_name = args.exp_name
        save_dir = os.path.join(BASE_PATH,'logs_models', exp_name)
        
        model_name = os.path.join(save_dir, "Trading_exp_1_finished")
        
        model = PPO2.load(model_name)
        

        obs = env.reset()
        #Test for n steps
        for i in range(1000):
                # test_model_load.start_innvestigate(new_model, obs)
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, dones, info = env.step(action)

                if dones:
                        print("RESET")
                        if args.visualize:
                                env.render()
                        #break
                
                

def checkEnv():
        from stable_baselines.common.env_checker import check_env

        env = tradingEnv.TradingEnvironment()
        # It will check your custom environment and output additional warnings if needed
        check_env(env)
        
if __name__ == "__main__":
        args = options.ArgumentParser()

        if args.mode == "train":
                print("Training")
                # train()
                train(args)
        elif args.mode=="test":
                print("Testing")
                test(args)
        elif args.mode=="continueTrain":
                continueTrain()
        elif args.mode=="checkEnv":
                #This function checks and gives warnings about your environments!
                checkEnv()
        elif args.mode=="colab":
                pass
        else:
                raise Exception 