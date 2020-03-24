class EnvironmentUtils():
    """A clas that holds some variables important to dynamically create and modify custom Trading environments"""
    def __init__(self):
        self.n_days_obs = 5 #Number of days to keep in the Observation Agent will have history for n days
        self.num_features_to_consider = 4 #Number of features that agent get's everyday of every hour
        self.num_actions = 3 # Number of actions agent should sample from
        self.agent_reaction_time = 1 #How often does the agent reacts every n hours or days, necesaary to define the length of a step

        """
            Define all other helper functions below
        """
