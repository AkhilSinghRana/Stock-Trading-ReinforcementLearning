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
######## DQN
# from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN

import tensorflow as tf
# tf.enable_eager_execution()

import argparse

def ArgumentParser():
        parser = argparse.ArgumentParser()

        # parser.add_argument('--mode', help="Mode can be train, test or continueTrain", default="train", type=str)
        parser.add_argument("--mode", type=str, default="train", help="train/test/continueTrain")
        parser.add_argument("--visualize","-visualize", action="store_true")

        parser.add_argument("--num_envs", type=int, default=1, help="number of parallel environments to train with")
        # parser.add_argument("--customPolicy", type=str, default="C:\\Airbus\\PAXSEM_Docker_r22990\\data_experiments\\2018_scenarios\\ReinforcementLearning\\paxsem_rl\\UseCase_TaskingAndPlanning\\customPolicy.py")
        # parser.add_argument("--model", type=str, default="C:\\Airbus\\PAXSEM_Docker_r22990\\data_experiments\\2018_scenarios\\ReinforcementLearning\\paxsem_rl\\UseCase_TaskingAndPlanning\\logs_models\\AREA_TEST__2020-02-01_14-02\\AREA_TEST__2020-02-01_14-02finished.pkl")
        parser.add_argument("--exp_name", type=str, default="GridWorld_MultiLevelGame", help="experiment name (used to save model&env)")
        parser.add_argument("--filesToBeSaved", type=str, default="envs/TaskingEnvironment_LevelGame_MultiDiscrete.py,EnvConstants.py,Rendering/renderingFunctionsPyglet.py,train_scenario.py,customPolicy.py,RasterArea.py,Agent.py", help="experiment name (used to save model&env)")
        # parser.add_argument("--instances", type=int, default=16)

        # Arguments for the STOCK MARKET
        parser.add_argument("--s_ticker", type=str, default="MSFT", help="This defines which stock will be used for training or testing, defaults to MICROSOFT")
        args = parser.parse_args()
        return args

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

def train(args):
        #Using Stable Baselines
        """
        Test if the algorithm (with a given policy)
        """
        env = make_vec_env(tradingEnv.TradingEnvironment, n_envs=args.num_envs)
        #env = VecFrameStack(env, n_stack = 4)
        #Uncomment to enable visualizations!
        #env = PPODummyVecEnv([lambda: droneEnv.DroneEnv(envObject)])
        print("Vectorized env created")
        print("Creating model") 
        env.reset()
        #Constants for saving logs and models
        exp_name = args.exp_name
        save_dir = os.path.join(BASE_PATH,'logs_models', exp_name)
        #Save Training Setup incase needed in future!
        save_training_setup(save_path=save_dir, file_list=args.filesToBeSaved, exp_name=exp_name)

        # Create PPO2 model now
        model = PPO2(CustomPolicy, env, verbose=0, tensorboard_log=save_dir, full_tensorboard_log=False) # PAXSEM throws exception if fulltensorboardlogs are enabled
        # model = DQN(CnnPolicy, env, verbose=1, tensorboard_log=save_dir, batch_size=64)
        
        # Train the model and save the results
        print("Training the network")
        try:
            #train for first 1 million Epochs
            steps_per_batch, num_envs = model.n_steps, env.num_envs
            ccb.createSignalHandler(env=env, model=model, exp_name=exp_name)
            
            callback_1 = ccb.saveIntermediateModels(exp_name, steps_per_batch, num_envs)
            callbacks = ccb.generateCallbackList()
            
            model.learn(total_timesteps=int(100e7), callback=callbacks, tb_log_name=exp_name, log_interval=10)
            
            print("First training done")
            model.save(save_path=os.path.join(BASE_PATH,'logs_models',exp_name,exp_name+'_finished'))
                
        except Exception as e:
                print("Exception occured during training", e)
                model_name = os.path.join(save_dir, "PPO2_error")
                model.save(model_name) 

        
        print("model saved")

        return
        

# TODO        
def continueTrain():
        print("Continue training here")
        env = make_vec_env(taskEnv.TaskEnvironment, n_envs=32)
        env = VecFrameStack(env, n_stack = 4)
        save_dir = "./logs_models/Level_Game_1"
        model_name = os.path.join(save_dir, "PPO2_ShrinkGoal_32x32_1")
        # model_name = os.path.join(save_dir, "PPO2_CNN_LSTM_success_1")
        # model_name = os.path.join(save_dir, "DQN_success_1")
        
        model = PPO2.load(model_name, env=env)
        model.learn(total_timesteps=2000000, tb_log_name = "exp_2")
        model_name = os.path.join(save_dir, "PPO2_DynamicGoal_ContinueTrain_SG_32x32_1")
        
        model.save(model_name) 
                
        

        
        print("model saved")

def test():
        import time
        print("testing the trained environmen")
        env = make_vec_env(taskEnv.TaskEnvironment, n_envs=1)
        env = VecFrameStack(env, n_stack = 4)
        env.env_method("setTesting")
        save_dir = "./logs_models/EncodeDeadAgents"
        model_name = os.path.join(save_dir, "EncodeDeadAgents_162570240")
        # model_name = os.path.join(save_dir, "PPO2_CNN_LSTM_success_1")
        # model_name = os.path.join(save_dir, "DQN_success_1")
        # save_dir_1 = ".\\logs_models\\grid_world_training_2"
        # model = PPO2(policy=None,env=env,tensorboard_log=save_dir_1, _init_setup_model=False)
        model = PPO2.load(model_name)
        
        # Enjoy trained agent
        obs = env.reset()
        env.render(mode="graphics")
        
        #Test for n steps
        for i in range(10000):
                 
                action, _states = model.predict(obs, deterministic=False)
                obs, rewards, dones, info = env.step(action)
                env.render(mode="graphics")
                time.sleep(0.06)
                # asdsaf 

def checkEnv():
        from stable_baselines.common.env_checker import check_env

        env = taskEnv.TaskEnvironment()
        # It will check your custom environment and output additional warnings if needed
        check_env(env)
        
if __name__ == "__main__":
        args = ArgumentParser()

        if args.mode == "train":
                print("Training")
                # train()
                train(args)
        elif args.mode=="test":
                print("Testing")
                test()
        elif args.mode=="continueTrain":
                continueTrain()
        elif args.mode=="checkEnv":
                #This function checks and gives warnings about your environments!
                checkEnv()
        else:
                raise Exception 