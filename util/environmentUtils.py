#Import the necessary packages
import yfinance as yf
import pandas as pd
import numpy as np
import random, os

from collections import deque
class EnvironmentUtils():
    """A clas that holds some variables important to dynamically create and modify custom Trading environments"""
    def __init__(self, s_Ticker="MSFT", trade_mode="2m", agent_mode="train", args=None):
        """
            Initialize the main Variables!
        """
        #Variables required for Stock market
        self.pandasData = None
        self.args = args
        self.stockTicker = s_Ticker # Ticker defines which stock to chosse, default is Microsoft, for this example
        self.trade_mode = trade_mode
        self.agent_mode = agent_mode
        self.MAX_SHARE_PRICE = None #Max price of the share used to normalize the data
        self.MAX_NUM_SHARES = None #Max number of Shares/Volume that the data can have
        self.init_loc = None        #Initial lcoation where the observation will start from, random every episode
        self.loc = self.init_loc

        #Observation based variables for the environment defined below
        
        self.agent_reaction_time = 60 #number of minutes agent needs in order to react, ===== 60 == 1 hour
        self.n_obs_hist = int((self.agent_reaction_time ) / int(trade_mode[0]))  #Number of days/hours(for minute mode) to keep in the Observation Agent will have history for n days/hours
        
        self.features=["Open", "Low", "High", "Close", "Volume"]
        
        self.num_features_to_consider = len(self.features) #Number of features that agent get's for every observation
        self.filled_Obs = deque(maxlen=self.n_obs_hist)
        
        #Action based variables
        self.num_actions = 3 # Number of actions agent should sample from, Buy, Hold Sell
        self.ACOUNT_BALANCE = 5000 #USD user can define it before starting training, check main.py or colab notebook!
         
        self.trainData = list() # Holds subset of training dataFrames
        self.already_sampled_DataIndices = list() # Holds the list indexes which have already been sampled in an episode

    def groupData(self):
        # Create Groups for everyday data
        self.data_grouped_by_date = self.pandasData.groupby(pd.Grouper(key="Datetime", freq='D'))

        #Create subsets of data Frames
        i=0
        for k, v in self.data_grouped_by_date:
            if v.shape[0] > 0:
                self.trainData.append(v) 
            
        

    def _getNextObservation(self, getData="fromCSV", mode="step"):
        """
            Gets the observation for the next step from the data Frame we support Pandas for now.
            The data can be either downloaded live online from yahooFinace library or can be read from CSV file
            Options:
                getData = 'online/fromCSV'
                loc , Defines which row to get
        """
        
        if self.pandasData is None :
            #Download the data if csv file is not available
            self.getStockObservation_fromCSV(mode=self.agent_mode)
            self.groupData() # Groups the data in subsets grouped on daily basis


        if mode=="reset":
            if len(self.already_sampled_DataIndices) == len(self.trainData):
                self.already_sampled_DataIndices.clear()
            while True:
                index = random.randint(0, len(self.trainData))
                if index in self.already_sampled_DataIndices:
                    pass
                else:
                    self.already_sampled_DataIndices.append(index)
                    self.data = self.trainData[index]
                    self.len_data = self.data.shape[0]
                    self.current_date = self.data["Datetime"].iloc[0].date()
                    self.loc = 0 # Location within the day, starts from 1st datapoint
                    

                    break

                    """
                    
                    self.MAX_NUM_SHARES = self.pandasData["Volume"].max()
                    self.MAX_SHARE_PRICE = self.pandasData["High"].max()
                    self.len_data = self.pandasData.shape[0]
                            
                    self.init_loc = random.randint(0, self.len_data) 
                    self.loc=int(self.init_loc)

                    
                    if self.loc==self.len_data-1:
                        self.loc=0
                    """
    
        if len(self.filled_Obs)<self.n_obs_hist:
            
            while len(self.filled_Obs)<self.n_obs_hist:
                obs = self.data.iloc[self.loc]
                
                #self.current_date = pd.to_datetime(obs["Datetime"]).date()
                
                self.open = obs["Open"].round(2)
                self.close = obs["Close"].round(2)
                obs = obs[self.features].round(2) if not getData=="fromCSV" else obs[self.features]
                
                
                self.loc+=1
                #if self.loc==self.len_data-1:
                #    self.loc=0
                self.filled_Obs.append([obs.values])
            

        else:
            obs = self.data.iloc[self.loc]
            #self.current_date = pd.to_datetime(obs["Datetime"]).date()
            obs = obs[self.features].round(2) if not getData=="fromCSV" else obs[self.features]
            self.loc+=1

            self.filled_Obs.append([obs.values])

        return self.filled_Obs , self.loc == self.len_data
    

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
        
        print("writing the train and test data to a csv file for later...")
        #Write train data
        self.pandasData.astype(np.float32)
        self.pandasData.to_csv("./train.csv",float_format='%.2f')
        #write test data
        self.testData.astype(np.float32)
        self.testData.to_csv("./test.csv",float_format='%.2f')
        

    def getStockObservation_fromCSV(self, mode="train", path="./"):
        filename = os.path.join(path, mode+".csv")
        if not os.path.exists(filename) or self.args.reload:
            self.getStockObservation_online()
        self.pandasData = pd.read_csv(filename, engine='python')

        self.pandasData["Datetime"] = pd.to_datetime(self.pandasData["Datetime"])
        self.pandasData = self.pandasData.set_index("Datetime", drop=False) # Set the date time as index, this is also default index while downloading online