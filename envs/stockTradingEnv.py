import os,sys
from importlib import reload
import numpy as np
import random
import pandas as pd
from collections import deque
from enum import Enum
#'Import gym', we will create a small wrapper with gym environment recommendations
import gym
from gym import spaces

import matplotlib.pyplot as plt

#Import Environment Utility class
from util import environmentUtils


# possible states agent can transition to
class POSSIBLE_STATES(Enum):
    LONG = 1
    HOLD = 0
    SHORT = -1

# possible positions agent can be in a particular timestep
class POSSIBLE_POSITIONS(Enum):
    OPEN = 1
    CLOSE = 0


class TradingEnvironment(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    def __init__(self, env_info={}):
        
        self.args = env_info["args"]
        self.external_func = None
        params = None
        if self.args.pass_external_func:
            self.external_func = env_info["external_func"]
            params = env_info["params"]

        self.getData_fromCSV = self.args.fromCSV #boolean
        self.env_utils =  environmentUtils.EnvironmentUtils(args= self.args, external_func=self.external_func, external_params = params)
        self.env_utils.ACOUNT_BALANCE = self.args.account_balance # Account balance to start with user defines it!
        
        self.ACOUNT_BALANCE = self.env_utils.ACOUNT_BALANCE
        self.reward_range = (0, self.ACOUNT_BALANCE*10)
        
        
        self.possible_agent_states = {"long": 1, "hold":0, "short":-1} 
      

        
            
        self.agentState = POSSIBLE_STATES.HOLD # initialize teh agent current state to hold
        self.agentPosition = POSSIBLE_POSITIONS.CLOSE # initialize the agent to be in close position

        self.agentState_history = deque([self.agentState.value for _ in range(self.args.obs_hist_window)], maxlen=self.args.obs_hist_window)
        
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
                                  "Price":list(),
                                  "Profit":list(),
                                  "Commision":list()
                                   },

                          "Sell":{"Time": list(),
                                  "Price":list(),
                                  "Profit":list(),
                                  "Commision":list()
                                   },
                            "GeneralInfo":{"NetWorth":list()
                                            }
                            }

        self.daily_stats_template = {   "Time": list(),
                                        "Type": list(),
                                        "Price":list(),
                                        "Account_Balance":list(),
                                        "Shares_Held":list(),
                                        "Profit":list(),
                                        "Commission":list(),
                                        "Net_Worth":list(),
                                        "Return":list(),
                                        "Agent_Position":list()
                                        }

        #self.stats_template = "Total Shares Bought {} \t Total Shares Sold {} \t Total Shares HELD {} \t Held Shares Worth{}"
        
        self.account_stat_template = "Account balance at Start {} \t ACCOUNT BALANCE at End {} \t NetWorth {} \t PROFIT for today {}"
        self.account_stat_template_reduced = "Return today: {}% \t # of Trades: {}"
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
        
        self.agentState = POSSIBLE_STATES.HOLD # Agent is holding on to cash
        
        """
        # RESET Daily Logs
        self.daily_log["Buy"]["Time"].clear()
        self.daily_log["Buy"]["Price"].clear()
        self.daily_log["Buy"]["Profit"].clear()
        self.daily_log["Buy"]["Commision"].clear()
        self.daily_log["Sell"]["Time"].clear()
        self.daily_log["Sell"]["Price"].clear()
        self.daily_log["Sell"]["Profit"].clear()
        self.daily_log["Sell"]["Commision"].clear()
        self.daily_log["GeneralInfo"]["NetWorth"].clear()
        """
        self.daily_stats_template["Time"].clear()
        self.daily_stats_template["Type"].clear()
        self.daily_stats_template["Price"].clear()
        self.daily_stats_template["Account_Balance"].clear()
        self.daily_stats_template["Shares_Held"].clear()
        
        self.daily_stats_template["Profit"].clear()
        self.daily_stats_template["Commission"].clear()
        self.daily_stats_template["Net_Worth"].clear()
        self.daily_stats_template["Return"].clear()
        self.daily_stats_template["Agent_Position"].clear()
        
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
            self.transposed_transactions.plot(x='Time', y='Profit', style='o')
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
            # Check if the agent is in Open Position
            #### if Agent Position = Open ==> Close the Position by changing the position to HOLD
            if self.agentPosition == POSSIBLE_POSITIONS.OPEN:
                # Either simply give a negative reward to keep the position open or forcefully close the position
                if self.agentState == POSSIBLE_STATES.LONG:
                    # Sell the shares
                    self.agentPosition = POSSIBLE_POSITIONS.CLOSE
                    self.performTransaction(transaction_type="Sell")
                else:
                    # Buy the Shares
                    self.agentPosition = POSSIBLE_POSITIONS.CLOSE
                    self.performTransaction(transaction_type="Buy")
                
                self.agentState = POSSIBLE_STATES.HOLD # Transition to HOLD at the end of episode!

            #self.step_reward+=self.calculateRewards() # Also include the daily Profit in the reward at the ends

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
    
    def compute_networth(self):
        # Agent's networth
        self.held_shares_worth = self.SHARES_HELD * self.env_utils.close
        self.net_worth = self.ACOUNT_BALANCE + self.held_shares_worth
    
    def mapActions(self, action):
        if action == 0:
            agent_transition_state = POSSIBLE_STATES.HOLD
        elif action == 1:
            agent_transition_state = POSSIBLE_STATES.LONG
        else:
            agent_transition_state = POSSIBLE_STATES.SHORT
        
        return agent_transition_state
    
    
    
    def performTransaction(self, transaction_type = None, double_transaction=False):
        """
            Buy or Sell transactions are done here
            transaction_type = Buy/Sell
            double_transaction = True means agent has decided to transition from Long/Short to Short/Long directly
        """

        # Set the current price to a random price within the time step
        #current_price = random.uniform(self.env_utils.open, self.env_utils.close)
        current_price = self.env_utils.close #current price is the current closing price


        self.compute_networth()
        self.total_daily_transactions += 1

        agentpenalty = self.args.trading_fees 
        
        if transaction_type == "Buy":
            #buy stocks
            #Check if the agent needs to sell the negative shares!, in case of short selling
            if self.agentPosition==POSSIBLE_POSITIONS.CLOSE:
                # Buy Stocks worth of negative shares, and close the position
                shares_to_recover = -self.SHARES_HELD
                
                net_transaction_price = shares_to_recover * current_price  # Price required to buy negative shares  
                # agent_penalty * TP = SP
                transaction_price = (net_transaction_price / (1-agentpenalty) )   #You have to pay more to cover the transaction commision
                
                commision = transaction_price - net_transaction_price
                
                total_possible = shares_to_recover
            else:
                # Buy shares worth of account balance
                commision = (agentpenalty * self.net_worth)
                # Take away transaction fee from the networth*
                self.net_worth = self.net_worth - commision
                self.ACOUNT_BALANCE -= commision 
                transaction_price = self.ACOUNT_BALANCE
                total_possible = self.ACOUNT_BALANCE / current_price
                


            self.total_buy_transactions += 1 # Buy Only when Feasible
            # BUY a percentage amount only if Account Balance is in positive
            shares_bought = total_possible
            self.SHARES_HELD += shares_bought

            
            self.ACOUNT_BALANCE -= transaction_price
            self.compute_networth()# Compute the new NetWorth, ACC_BAL + HeldPrice
            
            """
            self.daily_shares_bought += shares_bought
            self.daily_log["Buy"]["Time"].append(self.env_utils.current_time)
            self.daily_log["Buy"]["Price"].append(current_price)
            self.daily_log["Buy"]["Commision"].append(commision)
            self.daily_log["Buy"]["Profit"].append(self.net_worth - self.daily_account_balance)
            self.daily_log["GeneralInfo"]["NetWorth"].append(self.net_worth )
            """
        else:
            #Sell stocks
            self.total_sold_transactions += 1 #Increment only when feasible
            
            self.held_shares_worth = (self.SHARES_HELD * current_price) 

            
            if self.agentPosition == POSSIBLE_POSITIONS.CLOSE:
                # SELL a percentage amount, for now we sell it all
                shares_sold = self.SHARES_HELD

                transaction_price = shares_sold * current_price
                commision = (agentpenalty * transaction_price)
                price_gained = transaction_price - commision


            else:
                # Short Sell, max possible is daily account balance or may be better just the account balance in hand
                # shares_sold*current_price = daily_account_balance
                transaction_price = self.ACOUNT_BALANCE
                commision = (agentpenalty * transaction_price)
                transaction_price -= commision
                shares_sold = transaction_price / current_price
                
                price_gained = transaction_price
            

            
            self.ACOUNT_BALANCE += price_gained
            self.SHARES_HELD -= shares_sold
            self.compute_networth()

            
            """
            self.daily_log["Sell"]["Time"].append(self.env_utils.current_time)
            self.daily_log["Sell"]["Price"].append(current_price)
            self.daily_log["Sell"]["Commision"].append(commision)
            self.daily_log["Sell"]["Profit"].append(self.net_worth - self.daily_account_balance)
            """
        
        
        transaction_profit = self.net_worth - self.daily_account_balance # Computes the transaction profit
        transaction_return = (transaction_profit/self.daily_account_balance) * 100 # Computes the transaction return
        self.daily_stats_template["Time"].append(self.env_utils.current_time)
        self.daily_stats_template["Type"].append(transaction_type)
        self.daily_stats_template["Price"].append(current_price)
        self.daily_stats_template["Account_Balance"].append(round(self.ACOUNT_BALANCE,2))
        self.daily_stats_template["Shares_Held"].append(round(self.SHARES_HELD,2))
        
        self.daily_stats_template["Profit"].append(round(transaction_profit,2)) # Compute profit when transaction is closed
        self.daily_stats_template["Commission"].append(round(commision,2))
        self.daily_stats_template["Net_Worth"].append(round(self.net_worth,2))
        self.daily_stats_template["Return"].append(round(transaction_return,2))
        self.daily_stats_template["Agent_Position"].append(self.agentPosition)
        
        # assign the reward for this transaction
        self.step_reward += (-(agentpenalty * 2) + transaction_return/10)
        
    def performAction(self, action):
        #### Maps an Algorithm's action to the Agent's action
        assert (action in self.action_space), "Oh, no action is invalid, check the action space of the environment"
        
        self.step_reward = 0
        
        # Map the action
        agent_next_transition_state = self.mapActions(action) #MAPS the action to the possible transitions STATES

        
        #Check if the agent can transition to the next state, and perform transaction
        if agent_next_transition_state == self.agentState:
            # agent is already in the current state
            # 1->1, 0->0, -1->-1
            pass 
        else:
            if self.agentState == POSSIBLE_STATES.HOLD:
                #Agent is holding on to Cash
                if agent_next_transition_state == POSSIBLE_STATES.LONG:
                    # Agent has a request to Buy, and go to open buy position
                    self.agentPosition = POSSIBLE_POSITIONS.OPEN
                    self.performTransaction(transaction_type="Buy")
                    
                else:
                    # Agent has a request to Sell and open sell position (ShortSell)
                    self.agentPosition = POSSIBLE_POSITIONS.OPEN
                    self.performTransaction(transaction_type="Sell")
                    
            
            if self.agentState == POSSIBLE_STATES.LONG:
                #Agent is in long position
                if agent_next_transition_state == POSSIBLE_STATES.HOLD:
                    #agent is requesting to sell, and close it's position
                    self.agentPosition = POSSIBLE_POSITIONS.CLOSE
                    self.performTransaction(transaction_type="Sell")
                    
                else:
                    # agent is requesting to go to Short state, sell twice! and open it's position
                    self.agentPosition = POSSIBLE_POSITIONS.CLOSE
                    self.performTransaction(transaction_type="Sell")
                    self.agentPosition = POSSIBLE_POSITIONS.OPEN 
                    self.performTransaction(transaction_type="Sell") # second sell (ShortSell)
                    
            
            if self.agentState == POSSIBLE_STATES.SHORT:
                #Agent is in short position
                if agent_next_transition_state == POSSIBLE_STATES.HOLD:
                    #agent is requesting to Buy, and close it's position
                    self.agentPosition = POSSIBLE_POSITIONS.CLOSE
                    self.performTransaction(transaction_type="Buy")
                
                else:
                    # agent is requesting to go to long state, --> Buy twice! and open it's position
                    self.agentPosition = POSSIBLE_POSITIONS.CLOSE
                    self.performTransaction(transaction_type="Buy")

                    #Open posiiton again
                    self.agentPosition = POSSIBLE_POSITIONS.OPEN
                    self.performTransaction(transaction_type="Buy") # second buy
                    
        
        self.agentState = agent_next_transition_state
        self.agentState_history.append(self.agentState.value)


        
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
        self.compute_networth()
        self.daily_profit = self.net_worth - self.daily_account_balance
        
        pct_return = round((self.daily_profit/self.daily_account_balance) * 100, 2)
        print(self.account_stat_template_reduced.format(pct_return, self.total_sold_transactions))
        
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(self.transaction_stat_template.format(self.total_daily_transactions, self.total_buy_transactions, self.total_sold_transactions))

        # Create a pandas dataframe from tabular structure, easy to read and manipulate
        #transaction_data = pd.DataFrame.from_dict(self.daily_log)
        transaction_data = pd.DataFrame.from_dict(self.daily_stats_template)
        """
        transaction_data = pd.concat({
                                        k: pd.DataFrame.from_dict(v, 'index') for k, v in self.daily_stats_template.items()
                                    }, 
                                    axis=0)
        """
        #self.transposed_transactions = transaction_data.T
        self.transposed_transactions = transaction_data
        pd.options.display.float_format = '{:,.2f}'.format
        print(self.transposed_transactions) # Transposed Dataframe to show correct values


        # Drop the upper index, #done for ploting the transaction
        #self.transposed_transactions.columns = ['_'.join(col) for col in self.transposed_transactions.columns]
        
        
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