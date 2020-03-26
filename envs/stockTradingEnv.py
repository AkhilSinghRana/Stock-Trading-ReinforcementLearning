import os,sys
from importlib import reload
import numpy as np


#'Import gym', we will create a small wrapper with gym environment recommendations
import gym
from gym import spaces

#Import Environment Utility class
from util import environmentUtils
class TradingEnvironment(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.env_utils =  environmentUtils.EnvironmentUtils(s_Ticker="MSFT")
        # Define action and observation space for the environment
        self.observation_space = gym.spaces.Box(low= -np.inf, high= np.inf, 
                                                shape=(self.env_utils.n_days_obs, self.env_utils.num_features_to_consider), dtype=np.float32)
        
        #we allow movement only 1 element far ... so 8 neighboring elements are max possible elements to move to
        # 9 possible actions +1 to encode no movement
        self.action_space = gym.spaces.Discrete(self.env_utils.num_actions)
        
        
        self.done=False
        self.episodeCounter = 0 #Keep track of number of elapsed episodes during the run
        self.episodeLength = 0 #Keep track of the length of the episode, might be important for the reward scheme
        
        #Keep track of the rewards received by the agent over the time frame
        self.step_reward = 0
        self.episode_reward = 0
        
        print("Env Initialized")
    
    def setObservation(self):
        # generate the observation space either directly form the online stock library or local file
        
        self.observation_space = self.env_utils._getNextObservation()

    def reset(self):
        """
            Reset the observation to the random time or just change the Ticket/stock company
        """
        # reset episode steps
        self.episodeLength=0
        self.episodeCounter += 1
        self.episode_reward = 0
        
        self.done = False
        
        self.setObservation() 
        # self.render(mode="graphics")
        return self.observation_space
            
    def render(self, mode='human'):
        """
            Live rendering of the agents actions and the stock market data enable the flag to visualize it
        """
        if mode == "human":
            print(self.observation_space)
        
        else:
            # No Rendering or prints
            print("no rendering")
            pass

    def step(self, action, agentID=0):

        self.episodeLength+=1
        # Perform an action by the agent
        self.performAction(action)
        self.episode_reward+=self.step_reward
        
        self.done=False # Change to True when the environment is finished #TO-DO
        self.setObservation()
        info = {} #Generate extra information for debug
        return self.observation_space, self.step_reward, self.done, {}
    
    def performAction(self, action):
        #### Maps an Algorithm's action to the Agent's action
        assert (action in self.action_space), "Oh, no action is invalid, check the action space of the environment"
        self.step_reward = -0.01 # Define how much reward has to be assinged for the action
       
        
        if action == 0:
            # HOLD
            pass
        elif action == 1:
            # BUY
            pass
        elif action == 2:
            # SELL
            pass
            
        # Calculate the rewards now

            
    def calculateRewards(self):
        '''
        Design the reward Scheme here
        '''
        #Reward calculations
        reward = 0
        return reward


    def close(self):
        #Close the environment, do garbage Cleaning if needed!
        pass

if __name__=="__main__":
    # test the Environment with random actions here
    import time
    #TO-DO
    start_time = time.time()
    env = TradingEnvironment()
    env.reset()
    end_time = time.time()

    print("TIme taken by the environment ==>", end_time-start_time)