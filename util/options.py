import argparse

def ArgumentParser():
        parser = argparse.ArgumentParser()

        # parser.add_argument('--mode', help="Mode can be train, test or continueTrain", default="train", type=str)
        parser.add_argument("--mode", type=str, default="train", help="train/test/continueTrain")
        parser.add_argument("--visualize","-visualize", action="store_true")

        parser.add_argument("--num_envs", type=int, default=1, help="number of parallel environments to train with")
        parser.add_argument("--exp_name", type=str, default="Trading_exp_1", help="experiment name (used to save model&env)")
        
        # Arguments for the STOCK MARKET
        parser.add_argument("--s_ticker", type=str, default="MSFT", help="This defines which stock will be used for training or testing, defaults to MICROSOFT")
        args = parser.parse_args()
        return args