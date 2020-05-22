#Import the necessary packages
import yfinance as yf
import pandas as pd
import numpy as np
import random, os, glob

from collections import deque
class EnvironmentUtils():
    """A clas that holds some variables important to dynamically create and modify custom Trading environments"""
    def __init__(self, args=None, external_func=None, external_params=None):
        """
            Initialize the main Variables!
        """
        self.args = args
        #get the external function
        self.external_func = external_func
        self.external_params = external_params
            
        
        #Variables required for Stock market
        self.pandasData = None
        self.stockTicker = self.args.s_ticker # Ticker defines which stock to chosse, default is Microsoft, for this example
        self.num_stocks = len(self.stockTicker) # NUmber of stocks, whose features will be used, the first one will be traded for
        
        self.trade_mode = self.args.trade_interval # Trade interval 2minutes deafualt
        self.agent_mode = self.args.mode #train/test modes for agent
        self.MAX_SHARE_PRICE = None #Max price of the share used to normalize the data
        self.MAX_NUM_SHARES = None #Max number of Shares/Volume that the data can have

        #Observation based variables for the environment defined below
        
        self.agent_reaction_time = self.args.wait_time #number of minutes agent needs in order to react, ===== 60 == 1 hour
        self.n_obs_hist = self.args.obs_hist_window  #Observation window for the agent, agent holds history of window size n
        
        self.init_loc = int(((self.agent_reaction_time ) / int(self.trade_mode[0])) - self.n_obs_hist)        #Initial lcoation where the observation will start from, random every episode
        self.loc = self.init_loc
       
        self.features= self.args.s_features
        print(self.features)
        
        self.num_features_to_consider = len(self.features) #Number of features that agent get's for every observation
        self.num_features_to_consider = self.num_features_to_consider * self.num_stocks #THe obsevation space is now increased according to number of stocks
        self.filled_Obs = deque(maxlen=self.n_obs_hist)
        
        #Action based variables
        self.num_actions = 3 # Number of actions agent should sample from, Buy, Hold Sell
        self.ACOUNT_BALANCE = 5000 #USD user can define it before starting training, check main.py or colab notebook!
         
        self.trainData = list() # Holds subset of training dataFrames
        self.already_sampled_DataIndices = list() # Holds the list indexes which have already been sampled in an episode

        self.current_time = None #keeps track of the current time

    def groupData(self):
        self.trainData.clear()
        # Create Groups for everyday data
        grouped_dict=dict()
        for ticker in self.stockTicker:
            #Creates a groups in date time for every day and for each stock
            grouped_dict["{}".format(ticker)] = self.pandasData[ticker].groupby(pd.Grouper(key="Datetime", freq='D'))
            

        #Create subsets of data Frames
        ticker_sub_group = []
        for k, v in grouped_dict["{}".format(self.stockTicker[0])]:
            ticker_sub_group.clear()
            if v.shape[0] > 0:

                i=1
                ticker_sub_group.append(v)
                while i in range(len(self.stockTicker)):
                    #SUbset and merge the days into single train data for every day!
                    v = grouped_dict["{}".format(self.stockTicker[i])].get_group(k)
                    ticker_sub_group.append(v)
                    i+=1
                
                df = pd.concat(ticker_sub_group, axis=1, ignore_index=False, keys= self.stockTicker)
                
                #Test function support
                if self.args.singleDayTestMode:
                    if self.args.date == str(df[self.stockTicker[0]]["Datetime"].iloc[0].date()):
                        self.trainData.append(df) 
                else:
                    self.trainData.append(df) 

        print("Total groups available --> ", len(self.trainData))

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
                index = random.randint(0, len(self.trainData)-1)
                if index in self.already_sampled_DataIndices:
                    pass
                else:
                    self.already_sampled_DataIndices.append(index)
                    self.data = self.trainData[index]
                    self.len_data = self.data.shape[0]
                    
                    if self.len_data < self.n_obs_hist:
                        continue #Skips the days which don't have atleast the number of mins specified by the window size! might never happen
                    
                    # Compute the percentage data only when the flag is enabled
                    if self.args.compute_pct_change:
                        self.pct_change_data = []
                        self.pct_change_data.clear()
                        
                        for stock in self.stockTicker:
                            self.pct_change_data.append(self.data[stock][self.features].pct_change().dropna())
                        self.pct_change_data = pd.concat(self.pct_change_data, axis = 1, ignore_index=True)
                        self.len_data = self.pct_change_data.shape[0]

                    
                    self.current_date = self.data[self.stockTicker[0]]["Datetime"].iloc[0].date()
                    self.loc = self.init_loc # Location within the day, starts from agent cooldown period(wait time)
                    
                    
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
                
                self.current_time = self.data[self.stockTicker[0]]["Datetime"].iloc[self.loc].time() #Keeps track of the current time in the observation
                if self.args.compute_pct_change:
                    obs = self.pct_change_data.iloc[self.loc]
                    # get open for only first main stock
                    self.open = self.data[self.stockTicker[0]]["Open"].iloc[self.loc+1].round(2)
                    self.close = self.data[self.stockTicker[0]]["Close"].iloc[self.loc+1].round(2)
                    
                    self.filled_Obs.append([obs]) 
                    
                else:
                    obs = self.data.iloc[self.loc]
                    if any(obs.isnull()):
                        continue
                    #print(obs )
                    
                    # get open for only first main stock
                    self.open = obs[self.stockTicker[0]]["Open"].round(2)
                    self.close = obs[self.stockTicker[0]]["Close"].round(2)
                
                    #self.current_date = pd.to_datetime(obs["Datetime"]).date()
                    # Stretch observation acros single axis
                    combined_obs = [] # combined observation from multiple stocks
                    combined_obs.clear()
                    for stock in self.stockTicker:
                        combined_obs.append(obs[stock][self.features].round(2) if not getData=="fromCSV" else obs[stock][self.features])
                    
                    combined_obs = pd.concat(combined_obs, axis = 0, ignore_index=True)
                    self.filled_Obs.append([combined_obs.values])   
                
                self.loc+=1
                #if self.loc==self.len_data-1:
                #    self.loc=0
                
                # fill the observation for the agent
                
        else:
            self.current_time = self.data[self.stockTicker[0]]["Datetime"].iloc[self.loc].time() #Keeps track of the current time in the observation
            if self.args.compute_pct_change:
                obs = self.pct_change_data.iloc[self.loc]
                # get open for only first main stock
                self.open = self.data[self.stockTicker[0]]["Open"].iloc[self.loc+1].round(2)
                self.close = self.data[self.stockTicker[0]]["Close"].iloc[self.loc+1].round(2)
                
                self.filled_Obs.append([obs])
            else:
                obs = self.data.iloc[self.loc]
                if any(obs.isnull()):
                    self.loc+=1
                    return self.filled_Obs , self.loc == self.len_data
                #print(obs )
                # get open for only first main stock
                self.open = obs[self.stockTicker[0]]["Open"].round(2)
                self.close = obs[self.stockTicker[0]]["Close"].round(2)
            
                #self.current_date = pd.to_datetime(obs["Datetime"]).date()
                # Stretch observation acros single axis
                combined_obs = [] # combined observation from multiple stocks
                combined_obs.clear()
                for stock in self.stockTicker:
                    combined_obs.append(obs[stock][self.features].round(2) if not getData=="fromCSV" else obs[stock][self.features])
                
                combined_obs = pd.concat(combined_obs, axis = 0, ignore_index=True)

                self.filled_Obs.append([combined_obs.values])
            
            """obs = self.data.iloc[self.loc]
            #self.current_date = pd.to_datetime(obs["Datetime"]).date()
            # Stretch observation acros single axis
            combined_obs = [] # combined observation from multiple stocks
            combined_obs.clear()
            for stock in self.stockTicker:
                combined_obs.append(obs[stock][self.features].round(2) if not getData=="fromCSV" else obs[stock][self.features])
                
            combined_obs = pd.concat(combined_obs, axis = 0, ignore_index=True) # stretch the obsevration to single axis
            """
            self.loc+=1

        return self.filled_Obs , self.loc == self.len_data
    

    
    def getStockObservation_online(self):
        """
            Get the stock observation directly from the yFinance or other frameworks.
            If you have your data already written to CSV file please use methode 'getStockObservation_fromCSV'
        """
        print("Downloading following stock data now --> {}".format(self.stockTicker))
        
        self.pandasData = yf.download(tickers=self.stockTicker, interval=self.trade_mode,
                                         group_by=self.stockTicker, 
                                         rounding=True,
                                         start=None,
                                         end=None, 
                                         period="1mo",
                                         threads=True)
        """
        # The history call makes sure to get the max possible data                                         
        if len(self.stockTicker)>1:
            #self.tickerData = yf.Tickers(self.stockTicker) 
        else:
            #self.tickerData = yf.Ticker(self.stockTicker[0]) 
        """
        #self.pandasData = self.tickerData.history(interval=self.trade_mode, group_by=self.stockTicker, rounding=True)
        #add extra columns if defined in args
        if self.args.pass_external_func:
            param_dict = dict()
            param_dict.clear()
            
            # Create a list of dictionary for each parameter
            if self.external_params is not None:
                for param in self.external_params:
                    param_dict["{}".format(param)] = self.pandasData[self.stockTicker[0]][param]
            else:
                # Iterate over entire columns
                pass
                
            
            #output = self.external_func(self.pandasData[self.stockTicker[0]]["Open"], self.pandasData[self.stockTicker[0]]["Close"])
            
            for func in self.external_func:
                    
                output = func(param_dict)
                output = output.values.tolist()
            
                # By default the function name is choosen as the Column name for the data frame
                func_name = func.__name__ 
                self.pandasData[self.stockTicker[0], func_name] = output
            
            print(self.pandasData)



        total_nan = 0
                
        for i in range(len(self.pandasData.index)):
            row = self.pandasData.iloc[i].isnull()
            if any(row):
                total_nan +=1
        print(self.pandasData,self.pandasData.shape, "total rows with null values ====>",total_nan)
        

        multi_data = [] #list that stores multiple data frames, usefull to concatenate each of them later
        for i, ticker in enumerate(self.stockTicker):
            data = self.pandasData[ticker]
            data["Datetime"] = data.index
            multi_data.append(data)

        self.pandasData = pd.concat(multi_data, axis=1, ignore_index=False, keys= self.stockTicker)
        print("Data grouping, shuffling, megring and saving in process ::::::::::")
        #Group the data now!
        self.groupData()
        #Split to trainTest 
        split_percent = self.args.test_split_percentage #10 percent
        num_test_rows = int(split_percent* len(self.trainData))
        print("Total Test dates =====>", num_test_rows)
        # Shuffle the data and then take 20 percent as test!
        random.shuffle(self.trainData)
        random.shuffle(self.trainData)
        self.test_data = self.trainData[-num_test_rows:]
        self.trainData = self.trainData[:-num_test_rows]
        
        self.pandasData = pd.concat(self.trainData, axis=0, ignore_index=False, verify_integrity=True, sort=False)
        self.testData = pd.concat(self.test_data, axis=0, ignore_index=False, verify_integrity=True, sort=False)
        
        #for ticker in self.stockTicker:
        self.pandasData.drop("Datetime", axis=1, inplace=True, level=1)
        self.testData.drop("Datetime", axis=1, inplace =True, level=1)
        
        print("writing the train and test data to a csv file for later...")
        #Write train data
        save_dir = "./"
        if len(self.stockTicker)>1:
            for ticker in self.stockTicker:
                train_file_name = os.path.join(save_dir,"train_{}.csv".format(ticker))
                test_file_name = os.path.join(save_dir,"test_{}.csv".format(ticker))
                self.pandasData.astype(np.float32)
                self.pandasData[ticker].to_csv(train_file_name, float_format='%.2f')
                #write test data
                self.testData.astype(np.float32)
                self.testData[ticker].to_csv(test_file_name, float_format='%.2f')
            
        else:
                ticker = self.stockTicker[0]
                train_file_name = os.path.join(save_dir,"train_{}.csv".format(ticker))
                test_file_name = os.path.join(save_dir,"test_{}.csv".format(ticker))
                self.pandasData.astype(np.float32)
                self.pandasData.to_csv(train_file_name, float_format='%.2f')
                #write test data
                self.testData.astype(np.float32)
                self.testData.to_csv(test_file_name, float_format='%.2f')
            
    def getStockObservation_fromCSV(self, mode="train", path="./"):
        filenames = glob.glob(os.path.join(path, mode+"_*.csv"))
        if not filenames or not os.path.exists(filenames[0]) or self.args.reload:
            self.getStockObservation_online()
        filenames = glob.glob(os.path.join(path, mode+"_*.csv"))
        multi_data = [] #list that stores multiple data frames, usefult to concatenate each of them later
        for i, ticker in enumerate(self.stockTicker):
            for filename in filenames:
                if ticker in filename:
                    filename = filename
                    break
            data = pd.read_csv(filename, header=0, engine='python')
            data["Datetime"] = pd.to_datetime(data["Datetime"])
            if i==0:
                data = data.set_index("Datetime", drop=False) # Set the date time as index, this is also default index while downloading onlineTrading
            else:
                data = data.set_index("Datetime", drop=False) # Set the date time as index, this is also default index while downloading onlineTrading
            
            multi_data.append(data)
        
        
        self.pandasData = pd.concat(multi_data, axis=1, ignore_index=False, keys= self.stockTicker)

        