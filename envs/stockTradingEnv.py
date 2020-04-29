import os,sys
from importlib import reload
import numpy as np
import random

#'Import gym', we will create a small wrapper with gym environment recommendations
import gym
from gym import spaces

#Import Environment Utility class
from util import environmentUtils
class TradingEnvironment(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    def __init__(self, env_info={}):
        self.s_ticker = env_info["s_ticker"]
        self.trade_interval = env_info["trade_interval"]
        
        self.getData_fromCSV = env_info["fromCSV"]
        self.agent_mode = env_info["agent_mode"] #train/test
        self.env_utils =  environmentUtils.EnvironmentUtils(s_Ticker=self.s_ticker, trade_mode=self.trade_interval, agent_mode=self.agent_mode, args=env_info["args"])
        self.env_utils.ACOUNT_BALANCE = env_info["account_balance"] # Account balance to start with user defines it!
        # Define action and observation space for the environment
        self.observation_space = gym.spaces.Box(low= -np.inf, high= np.inf, 
                                                shape=(self.env_utils.n_obs_hist, self.env_utils.num_features_to_consider), dtype=np.float32)
        
        #we allow movement only 1 element far ... so 8 neighboring elements are max possible elements to move to
        # 9 possible actions +1 to encode no movement
        self.action_space = gym.spaces.Box(low= np.array([0, 0.1]) , high= np.array([self.env_utils.num_actions, 1]), dtype=np.float32)
        self.ACOUNT_BALANCE = self.env_utils.ACOUNT_BALANCE
        self.reward_range = (0, self.ACOUNT_BALANCE*10000)
        
        
        self.done=False
        self.episodeCounter = 0 #Keep track of number of elapsed episodes during the run
        self.episodeLength = 0 #Keep track of the length of the episode, might be important for the reward scheme
        
        
        #Keep track of the rewards received by the agent over the time frame
        self.step_reward = 0
        self.episode_reward = 0

        #Stock Share options for agent
        self.SHARES_HELD = 0 #Number of shares Held in total

        self.daily_shares_sold = 0
        self.daily_shares_bought = 0
        self.daily_profit = 0
        self.daily_account_balance = self.ACOUNT_BALANCE

        self.stats_template = "Total Shares Bought {} \t\t Total Shares Sold {} \t\t Total Shares HELD {}"
        self.account_stat_template = "Account balance at Start {} \t ACCOUNT BALANCE at End {} \t NetWorth {} \t PROFIT for today {}"
        
        print("Env Initialized")
    
    def setObservation(self, mode="step"):
        # generate the observation space either directly form the online stock library or local file
        if self.getData_fromCSV:
            self.observation_space, self.done = self.env_utils._getNextObservation(mode=mode, getData="fromCSV")    
        else:
            self.observation_space, self.done = self.env_utils._getNextObservation(mode=mode)
            
        self.observation_space = np.asarray(self.observation_space).reshape(self.env_utils.n_obs_hist, self.env_utils.num_features_to_consider)
        

    def reset(self):
        """
            Reset the observation to the random time or just change the Ticket/stock company
        """
        self.setObservation(mode="reset") 

        # reset episode steps
        self.episodeLength=0
        self.max_episode_length = self.env_utils.len_data # Length of the data!
        self.episodeCounter += 1
        self.episode_reward = 0

        #Stock Share options for agent
        self.ACOUNT_BALANCE = self.daily_account_balance = self.env_utils.ACOUNT_BALANCE

        self.SHARES_HELD = 0 #Number of shares bought in this episode
        self.SHARES_SOLD = 0 #Number of shares bought in this episode

        self.MONEY_SPENT = 0
        self.MONEY_EARNED = 0
        self.net_worth = self.max_net_worth = self.ACOUNT_BALANCE
        
        
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
        self.episode_reward += self.step_reward
        
        # Change to True when the net worth is 0
        
        self.setObservation()
        if self.done:
            # Finish the episode, and reset agent's variables, like account balance, sample a random day again!
            print("Agent's statistics for today: {}".format(self.env_utils.current_date))
            self.showStats()
            print("=======================================")
            self.daily_account_balance = self.ACOUNT_BALANCE
        
        info = {} #Generate extra information for debug
        
        return self.observation_space, self.step_reward, self.done, info
    
    def performAction(self, action):
        #### Maps an Algorithm's action to the Agent's action
        assert (action in self.action_space), "Oh, no action is invalid, check the action space of the environment"
        action_type = action[0]
        amount = action[1]
        
        # Set the current price to a random price within the time step
        current_price = random.uniform(self.env_utils.open, self.env_utils.close)
        
        if action_type < 1:
            # HOLD
            pass
        elif action_type < 2:
            if self.ACOUNT_BALANCE > 0:
                # BUY a percentage amount only if Account Balance is in positive
                total_possible = self.ACOUNT_BALANCE / current_price
                shares_bought = total_possible * amount
                
                additional_cost = shares_bought * current_price
                self.MONEY_SPENT += additional_cost
                self.ACOUNT_BALANCE -= additional_cost
                self.SHARES_HELD += shares_bought
                self.daily_shares_bought += shares_bought
            else:
                self.step_reward = -10
                return
            
        elif action_type < 3:
            # SELL a percentage amount
            shares_sold = self.SHARES_HELD * amount 
            
            self.ACOUNT_BALANCE += shares_sold * current_price
            self.SHARES_HELD -= shares_sold
            self.SHARES_SOLD += shares_sold
            self.MONEY_EARNED += shares_sold * current_price
            self.daily_shares_sold += shares_sold

            
        self.net_worth = self.ACOUNT_BALANCE + self.SHARES_HELD * current_price
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
        
        # Calculate the rewards now
        self.step_reward = self.calculateRewards()

            
    def calculateRewards(self):
        '''
        Reward Scheme focussed on account balance
        '''
        #Reward calculations
        delay_modifier = (self.episodeLength / self.max_episode_length)
        reward = self.ACOUNT_BALANCE * delay_modifier #Do we want our agent to increase the networth or Account Balance
        
        return reward

    
    def showStats(self):
        self.daily_profit = self.ACOUNT_BALANCE - self.daily_account_balance
        print(self.stats_template.format(self.daily_shares_bought, self.daily_shares_sold, self.SHARES_HELD))
        
        
        print(self.account_stat_template.format(self.daily_account_balance, self.ACOUNT_BALANCE, self.net_worth, self.daily_profit))
        
        #Reset the daily stats
        self.daily_shares_bought = self.daily_shares_sold = 0 

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