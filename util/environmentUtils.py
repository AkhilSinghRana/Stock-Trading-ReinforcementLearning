#Import the necessary packages
import yfinance as yf
import pandas as pd
import random

from collections import deque
class EnvironmentUtils():
    """A clas that holds some variables important to dynamically create and modify custom Trading environments"""
    def __init__(self, s_Ticker="MSFT"):
        """
            Initialize the main Variables!
        """
        #Variables required for Stock market
        self.pandasData = None
        self.stockTicker = s_Ticker # Ticker defines which stock to chosse, default is Microsoft, for this example
        
        self.MAX_SHARE_PRICE = None #Max price of the share used to normalize the data
        self.MAX_NUM_SHARES = None #Max number of Shares/Volume that the data can have
        self.init_loc = None        #Initial lcoation where the observation will start from, radnom every episode
        self.loc = self.init_loc

        #Observation based variables for the environment defined below
        self.n_days_obs = 5 #Number of days to keep in the Observation Agent will have history for n days
        self.features=["Open", "Low", "High", "Close", "Volume"]
        self.num_features_to_consider = len(self.features) #Number of features that agent get's everyday of every hour
        self.filled_Obs = deque(maxlen=self.n_days_obs)
        
        #Action based variables
        self.num_actions = 3 # Number of actions agent should sample from, Buy, Hold Sell
        self.ACOUNT_BALANCE = 50000
         
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
                self.getStockObservation_fromCSV()
            else:
                raise NotImplementedError
                
        
        if mode=="reset":
            self.MAX_NUM_SHARES = self.pandasData["Volume"].max()
            self.MAX_SHARE_PRICE = self.pandasData["High"].max()
            self.len_data = self.pandasData.shape[0]
            self.init_loc = random.randint(0, self.len_data) 
            self.loc=self.init_loc
            print("Starting the episode from day ==>",self.loc)

        
        if self.loc==self.len_data-1:
            self.loc=0
        
    
        if len(self.filled_Obs)<self.n_days_obs:
            while len(self.filled_Obs)<self.n_days_obs:
                obs = self.pandasData.iloc[self.loc]
                self.open = obs["Open"].round(2)
                self.close = obs["Close"].round(2)
                obs = obs[self.features].round(2)
                
                
                self.loc+=1
                if self.loc==self.len_data-1:
                    self.loc=0
                self.filled_Obs.append([obs.values])
        else:
            obs = self.pandasData.iloc[self.loc]
            obs = obs[self.features].round(2)
            self.loc+=1
            if self.loc==self.len_data-1:
                self.loc=0
        
            self.filled_Obs.append([obs.values])
    
        return obs
    

    def getStockObservation_online(self, start_date="2010-01-01", end_date=""):
        """
            Get the stock observation directly from the yFinance or other frameworks.
            If you have your data already written to CSV file please use methode 'getStockObservation_fromCSV'
        """
        self.pandasData = yf.download(self.stockTicker, start=start_date)
        

    def getStockObservation_fromCSV(self):
        pass
