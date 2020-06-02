import argparse

def ArgumentParser():
        parser = argparse.ArgumentParser()

        # parser.add_argument('--mode', help="Mode can be train, test or continueTrain", default="train", type=str)
        parser.add_argument("--mode", type=str, default="train", help="train/test/continueTrain")
        
        parser.add_argument("--singleDayTestMode", action="store_true")
        parser.add_argument("--date", type=str, default="2020-04-30", help="date to test the agent")

        parser.add_argument("--fromCSV", type=bool, default=False, help="Read the data from CSV")
        parser.add_argument("--reload", action="store_true")
        parser.add_argument("--visualize","-visualize", action="store_true")

        #Arguments for RL algorithm/ taining / testing
        parser.add_argument("--num_envs", type=int, default=1, help="number of parallel environments to train with")
        parser.add_argument("--num_epochs", type=int, default=1000, help="number of epochs to train the learn function of algorithm")
        parser.add_argument("--exp_name", type=str, default="Trading_exp_1", help="experiment name (used to save model&env)")
        
        # Arguments for the STOCK MARKET
        parser.add_argument("--s_ticker", nargs="+", default=["MSFT"], help="This defines which stock will be used for training or testing, defaults to MICROSOFT")
        parser.add_argument("--s_features", nargs="+", default=["Open" , "Low", "High", "Close", "Volume"], help="Features to consider for stock")
        parser.add_argument("--compute_pct_change", action="store_true")
        parser.add_argument("--test_split_percentage", type=float, default=0.10, help="Percentage of data to be used for Testing")
        parser.add_argument("--obs_hist_window", type=int, default=5, help="Agent Observation window, defaults to 5")
        parser.add_argument("--wait_time", type=int, default=60, help="Agent waits for the specified number of mins everyday before it is ready for trade")
        parser.add_argument("--trade_interval", type=str, default="2m", help="interval to record 1m/1h/1d ... default 2m please refer to yFinance for availabel options")
        parser.add_argument("--account_balance", type=int, default=5000, help="Account balance to start with $USD default: 5000 $USD")
        parser.add_argument("--trading_fees", type=float, default=0.01, help="percentage fees agent pays everytime it makes a transaction")

        #Arguments for passing external methods
        parser.add_argument("--pass_external_func", action="store_true")
        args = parser.parse_args()
        return args