import numpy as np
import minerl

class MineRlDemoProvider:
    """
    This object helps to load Minecraft demonstrations with the help of a minerl data loader.
    It offers a method next_sequence so that it can be used, e.g., in an AmpEnv.
    """
    def __init__(self, env_name, data_dir):
        self.data_loader = minerl.data.make(env_name, data_dir=data_dir)
        self.reference_names = [name for name in self.data_loader.get_trajectory_names() if not minerl.data.DataPipeline._is_blacklisted(name)]
    
    def next_sequence(self):
        #TODO consider stacking these observations if needed.
        #return [next_obs for obs, act, reward, next_obs, done in self.data_loader.load_data(np.random.choice(self.reference_names))]
        
        #return [{"action":act["vector"], "pov":obs["pov"]} for obs, act, reward, next_obs, done in self.data_loader.load_data(np.random.choice(self.reference_names))]
        return [{"pov":obs["pov"]} for obs, act, reward, next_obs, done in self.data_loader.load_data(np.random.choice(self.reference_names))]

