#Import the necessary packages
import yfinance as yf
import pandas as pd
import numpy as np
import random

from collections import deque
class EnvironmentUtils():
    """A clas that holds some variables important to dynamically create and modify custom Trading environments"""
    def __init__(self, s_Ticker="MSFT", trade_mode="interday"):
        """
            Initialize the main Variables!
        """
        #Variables required for Stock market
        self.pandasData = None
        self.stockTicker = s_Ticker # Ticker defines which stock to chosse, default is Microsoft, for this example
        self.trade_mode = trade_mode
        self.MAX_SHARE_PRICE = None #Max price of the share used to normalize the data
        self.MAX_NUM_SHARES = None #Max number of Shares/Volume that the data can have
        self.init_loc = None        #Initial lcoation where the observation will start from, random every episode
        self.loc = self.init_loc

        #Observation based variables for the environment defined below
        self.n_obs_hist = 5 #Number of days/hours(for minute mode) to keep in the Observation Agent will have history for n days/hours
        self.features=["Open", "Low", "High", "Close", "Volume"]
        self.num_features_to_consider = len(self.features) #Number of features that agent get's for every observation
        self.filled_Obs = deque(maxlen=self.n_obs_hist)
        
        #Action based variables
        self.num_actions = 3 # Number of actions agent should sample from, Buy, Hold Sell
        self.ACOUNT_BALANCE = 5000 #USD
         
        self.agent_reaction_time = 1 #How often does the agent reacts every n hours or days, necesaary to define the length of a step

    def _getNextObservation(self, getData="online", mode="step"):
        """
            Gets the observation for the next step from the data Frame we support Pandas for now.
            The data can be either downloaded live online from yahooFinace library or can be read from CSV file
            Options:
                getData = 'online/fromCSV'
                loc , Defines which row to get
        """
        
        if self.pandasData is None :
            print("No data found downloading now")
            if getData=="online":
                self.getStockObservation_online()
            elif getData=="fromCSV":
                print("Loading from CSV")
                self.getStockObservation_fromCSV()
            else:
                raise NotImplementedError
                
        
        if mode=="reset":
            self.MAX_NUM_SHARES = self.pandasData["Volume"].max()
            self.MAX_SHARE_PRICE = self.pandasData["High"].max()
            self.len_data = self.pandasData.shape[0]
            self.init_loc = random.randint(0, self.len_data) 
            self.loc=int(self.init_loc)
            print("Starting the episode from day ==>",self.loc)

        
        if self.loc==self.len_data-1:
            self.loc=0
        
    
        if len(self.filled_Obs)<self.n_obs_hist:
            while len(self.filled_Obs)<self.n_obs_hist:
                obs = self.pandasData.iloc[self.loc]
                
                self.open = obs["Open"].round(2)
                self.close = obs["Close"].round(2)
                obs = obs[self.features].round(2) if not getData=="fromCSV" else obs[self.features]
                
                
                self.loc+=1
                if self.loc==self.len_data-1:
                    self.loc=0
                self.filled_Obs.append([obs.values])

        else:
            obs = self.pandasData.iloc[self.loc]
            obs = obs[self.features].round(2) if not getData=="fromCSV" else obs[self.features]
            self.loc+=1
            if self.loc==self.len_data-1:
                self.loc=0
        
            self.filled_Obs.append([obs.values])
    
        return obs
    

    def getStockObservation_online(self):
        """
            Get the stock observation directly from the yFinance or other frameworks.
            If you have your data already written to CSV file please use methode 'getStockObservation_fromCSV'
        """
        print("Downloading {} data now".format(self.stockTicker))
        self.tickerData = yf.Ticker(self.stockTicker)
        self.pandasData = self.tickerData.history(interval=self.trade_mode)
        
        print(self.pandasData.shape)
        
        #Split to trainTest 
        split_percent = 0.1 #10 percent
        num_test_rows = int(split_percent* self.pandasData.shape[0])
        self.testData = self.pandasData.tail(num_test_rows)
        self.pandasData = self.pandasData[:-num_test_rows]
        
        print("writing the test data to a csv file for later...")
        self.testData.astype(np.float32)
        self.testData.to_csv("./test.csv",float_format='%.2f')
        

    def getStockObservation_fromCSV(self, filename="./test.csv"):
        self.pandasData = pd.read_csv(filename)
        