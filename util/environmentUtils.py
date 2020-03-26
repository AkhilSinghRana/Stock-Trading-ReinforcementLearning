#Import the necessary packages
import yfinance as yf
import pandas as pd

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

        #Observation based variables for the environment defined below
        self.n_days_obs = 5 #Number of days to keep in the Observation Agent will have history for n days
        self.features=["Open", "Low", "High", "Close", "Volume"]
        self.num_features_to_consider = len(self.features) #Number of features that agent get's everyday of every hour
        self.filled_Obs = deque(maxlen=self.n_days_obs)
        self.num_actions = 3 # Number of actions agent should sample from
        self.agent_reaction_time = 1 #How often does the agent reacts every n hours or days, necesaary to define the length of a step

    def _getNextObservation(self, loc=0, getData="online"):
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
                
        
        #Read the data at specified location depends on the episode length of the environment
        self.MAX_NUM_SHARES = self.pandasData["Volume"].max()
        self.MAX_SHARE_PRICE = self.pandasData["High"].max()
        obs = self.pandasData.iloc[loc]
        print(obs, self.pandasData.ndim)
        print(self.filled_Obs)
        obs = obs[self.features]
        print(obs)
        if len(self.filled_Obs)<self.n_days_obs:
            self.filled_Obs.append([obs.values])
        print(self.filled_Obs, len(self.filled_Obs))
        return obs
    

    def getStockObservation_online(self, start_date="2010-01-01", end_date=""):
        """
            Get the stock observation directly from the yFinance or other frameworks.
            If you have your data already written to CSV file please use methode 'getStockObservation_fromCSV'
        """
        self.pandasData = yf.download(self.stockTicker, start=start_date)
        

    def getStockObservation_fromCSV(self):
        pass
