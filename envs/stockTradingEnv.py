import os,sys
from importlib import reload
import numpy as np
import random
import pandas as pd
from collections import deque
#'Import gym', we will create a small wrapper with gym environment recommendations
import gym
from gym import spaces

import matplotlib.pyplot as plt



#Import Environment Utility class
from util import environmentUtils
class TradingEnvironment(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    def __init__(self, env_info={}):
        
        self.args = env_info["args"]
        self.external_func = None
        if self.args.pass_external_func:
            self.external_func = env_info["external_func"]
            params = env_info["params"]

        self.getData_fromCSV = self.args.fromCSV #boolean
        self.env_utils =  environmentUtils.EnvironmentUtils(args= self.args, external_func=self.external_func, external_params = params)
        self.env_utils.ACOUNT_BALANCE = self.args.account_balance # Account balance to start with user defines it!
        
        self.ACOUNT_BALANCE = self.env_utils.ACOUNT_BALANCE
        self.reward_range = (0, self.ACOUNT_BALANCE*10)
        
        self.agentState = {"long": 1, "hold":0, "short":-1}
        self.agentState_history = deque([0 for _ in range(self.args.obs_hist_window)], maxlen=self.args.obs_hist_window)
        
        #number of columns for observation space
        self.num_cols = self.env_utils.num_features_to_consider + 1#agentstate
        
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
        self.total_daily_transactions = 0
        self.total_buy_transactions = 0
        self.total_sold_transactions = 0
        
        self.daily_log = {"Buy": {"Time": list(),
                                  "Price":list() 
                                   },
                          "Sell":{"Time": list(),
                                  "Price":list(),
                                  "Profit":list()
                                   }
                            }

        self.stats_template = "Total Shares Bought {} \t Total Shares Sold {} \t Total Shares HELD {} \t Held Shares Worth{}"
        self.account_stat_template = "Account balance at Start {} \t ACCOUNT BALANCE at End {} \t NetWorth {} \t PROFIT for today {}"
        self.transaction_stat_template = "Total Transactions: {} \t Buy: {} \t Sold: {} "
        
        #initialize the spaces
        self._init_spaces()
        print("Env Initialized")
   
    def _init_spaces(self):
        # Define action and observation space for the environment
        
        self.env_obs_shape = (self.env_utils.n_obs_hist, self.num_cols)
        self.observation_space = gym.spaces.Box(low= -np.inf, high= np.inf, 
                                                shape=self.env_obs_shape, dtype=np.float32)
        
        self.action_space = gym.spaces.Discrete(3)
        # Uncomment to change to Box Action Space, usefull if you require multiple actions by agent
        #self.action_space = gym.spaces.Box(low= np.array([0, 0.1]) , high= np.array([self.env_utils.num_actions, 1]), dtype=np.float32)
        
    def setObservation(self, mode="step"):
        # generate the observation space either directly form the online stock library or local file
        if self.getData_fromCSV:
            self.stock_observation_space, self.done = self.env_utils._getNextObservation(mode=mode, getData="fromCSV")    
        else:
            self.stock_observation_space, self.done = self.env_utils._getNextObservation(mode=mode)
        

        self.stock_observation_space = np.asarray(self.stock_observation_space).reshape(self.env_utils.n_obs_hist, self.env_utils.num_features_to_consider)
        
        
        agent_states = np.asarray(self.agentState_history).reshape(self.env_utils.n_obs_hist,1)
        self.observation_space = np.append(self.stock_observation_space, agent_states, axis=1)
        
        
        
    def reset(self):
        """
            Reset the observation to the random time or just change the Ticket/stock company
        """
        self.env_utils.filled_Obs.clear()
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

        #Reset the transactions
        self.total_daily_transactions = 0
        self.total_buy_transactions = 0
        self.total_sold_transactions = 0
        self.daily_log["Buy"]["Time"].clear()
        self.daily_log["Buy"]["Price"].clear()
        self.daily_log["Sell"]["Time"].clear()
        self.daily_log["Sell"]["Price"].clear()
        self.daily_log["Sell"]["Profit"].clear()
        
        #print("RESET")
        #print(self.observation_space, self.observation_space.shape)
        #print(self.observation_space.shape)
        return self.observation_space
            
    def render(self, mode='human'):
        """
            Live rendering of the agents actions and the stock market data enable the flag to visualize it
        """
        if mode == "human":
            print("Rendering Agent Profit graph")
            self.transposed_transactions.plot(x='Sell_Time', y='Sell_Profit', style='o')
            plt.show()
        
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
            self.step_reward+=self.calculateRewards()
            # Finish the episode, and reset agent's variables, like account balance, sample a random day again!
            print("Agent's statistics for today: {}".format(self.env_utils.current_date))
            self.showStats()
            print("=======================================================================================================================")
            self.daily_account_balance = self.ACOUNT_BALANCE
            if self.args.visualize:
                self.render()
        
        info = {} #Generate extra information for debug
        #print(self.observation_space, self.observation_space.shape)
        #print(self.observation_space.shape)
        return self.observation_space, self.step_reward, self.done, info
    
    def performAction(self, action):
        #### Maps an Algorithm's action to the Agent's action
        assert (action in self.action_space), "Oh, no action is invalid, check the action space of the environment"
        self.step_reward = 0
        # Set the current price to a random price within the time step
        current_price = random.uniform(self.env_utils.open, self.env_utils.close)
        if action == 0:
            # HOLD
            agentState = "hold"
            agentpenalty = self.args.trading_fees if abs(self.agentState[agentState] - self.agentState_history[-1]) == 1 else 0 
            pass

        else:
            self.total_daily_transactions += 1
            if action == 1:
                agentState = "long"
                agentpenalty = self.args.trading_fees if abs(self.agentState[agentState] - self.agentState_history[-1]) == 1 else self.args.trading_fees * 2 
                self.ACOUNT_BALANCE = self.ACOUNT_BALANCE - (agentpenalty * self.ACOUNT_BALANCE)
                
                if self.ACOUNT_BALANCE > 0:
                    self.total_buy_transactions += 1 # Buy Only when Feasible
                    self.daily_log["Buy"]["Time"].append(self.env_utils.current_time)
                    self.daily_log["Buy"]["Price"].append(current_price)
                
                    # BUY a percentage amount only if Account Balance is in positive
                    total_possible = self.ACOUNT_BALANCE / current_price
                    shares_bought = total_possible 
                    
                    additional_cost = shares_bought * current_price
                    self.MONEY_EARNED = 0
                    self.MONEY_SPENT += additional_cost
                    self.ACOUNT_BALANCE -= additional_cost
                    self.SHARES_HELD += shares_bought
                    self.daily_shares_bought += shares_bought
                
            
            elif action == 2:
                agentState = "short"
            
                agentpenalty = self.args.trading_fees if abs(self.agentState[agentState] - self.agentState_history[-1]) == 1 else self.args.trading_fees * 2 
                
                self.held_shares_worth = (self.SHARES_HELD * current_price) - (agentpenalty* (self.SHARES_HELD * current_price))
            
                if self.held_shares_worth > 0:
                    self.total_sold_transactions += 1 #Increment only when feasible
                    
                    # SELL a percentage amount, for now we sell it all
                    shares_sold = self.SHARES_HELD 
                    
                    self.ACOUNT_BALANCE += self.held_shares_worth
                    self.SHARES_HELD -= shares_sold
                    self.SHARES_SOLD += shares_sold
                    self.MONEY_EARNED += shares_sold * current_price
                    self.transaction_profit = self.MONEY_EARNED - self.MONEY_SPENT
                    self.MONEY_SPENT = 0
                    self.daily_shares_sold += shares_sold

                    self.daily_log["Sell"]["Time"].append(self.env_utils.current_time)
                    self.daily_log["Sell"]["Price"].append(current_price)
                    self.daily_log["Sell"]["Profit"].append(self.transaction_profit)
                    
                    #self.step_reward = self.transaction_profit

        self.held_shares_worth = self.SHARES_HELD * current_price
        self.net_worth = self.ACOUNT_BALANCE + self.held_shares_worth
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
        
        # Calculate the rewards now
        self.step_reward -= agentpenalty

        self.agentState_history.append(self.agentState[agentState])
            
    def calculateRewards(self):
        '''
        Reward Scheme focussed on account balance
        '''
        #Reward calculations
        #delay_modifier = (self.episodeLength / self.max_episode_length)
        self.daily_profit = self.ACOUNT_BALANCE - self.daily_account_balance
        reward = self.daily_profit / self.daily_account_balance #Do we want our agent to increase the networth or Account Balance
        
        return reward

    
    def showStats(self):
        self.daily_profit = self.ACOUNT_BALANCE - self.daily_account_balance
        print(self.stats_template.format(self.daily_shares_bought, self.daily_shares_sold, self.SHARES_HELD, self.held_shares_worth))
        
        print(self.account_stat_template.format(self.daily_account_balance, self.ACOUNT_BALANCE, self.net_worth, self.daily_profit))
        
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(self.transaction_stat_template.format(self.total_daily_transactions, self.total_buy_transactions, self.total_sold_transactions))

        # Create a pandas dataframe from tabular structure, easy to read and manipulate
        transaction_data = pd.DataFrame.from_dict(self.daily_log)
        transaction_data = pd.concat({
                                        k: pd.DataFrame.from_dict(v, 'index') for k, v in self.daily_log.items()
                                    }, 
                                    axis=0)

        self.transposed_transactions = transaction_data.T
        print(self.transposed_transactions) # Transposed Dataframe to show correct values


        # Drop the upper index, #done for ploting the transaction
        self.transposed_transactions.columns = ['_'.join(col) for col in self.transposed_transactions.columns]
        
        
        #print(transaction_data.T["Profit"].sum(axis = 0, skipna = True, level=[0]) )
        #print(self.daily_log["Sell"])
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

    print("Time taken by the environment ==>", end_time-start_time)