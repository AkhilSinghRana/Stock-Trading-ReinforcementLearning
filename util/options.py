import argparse

def ArgumentParser():
        parser = argparse.ArgumentParser()

        # parser.add_argument('--mode', help="Mode can be train, test or continueTrain", default="train", type=str)
        parser.add_argument("--mode", type=str, default="train", help="train/test/continueTrain")
        parser.add_argument("--fromCSV", type=bool, default=False, help="Read the data from CSV")
        parser.add_argument("--visualize","-visualize", action="store_true")

        #Arguments for RL algorithm/ taining / testing
        parser.add_argument("--num_envs", type=int, default=1, help="number of parallel environments to train with")
        parser.add_argument("--num_epochs", type=int, default=1000, help="number of epochs to train the learn function of algorithm")
        parser.add_argument("--exp_name", type=str, default="Trading_exp_1", help="experiment name (used to save model&env)")
        
        # Arguments for the STOCK MARKET
        parser.add_argument("--s_ticker", type=str, default="MSFT", help="This defines which stock will be used for training or testing, defaults to MICROSOFT")
        parser.add_argument("--trade_interval", type=str, default="1h", help="interval to record 1m/1h/1d ... default 1 hour please refer to yFinance for availabel options")
        parser.add_argument("--account_balance", type=int, default=5000, help="Account balance to start with $USD default: 5000 $USD")
        args = parser.parse_args()
        return args