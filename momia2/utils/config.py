import configparser

class Config:
    def __init__(self, configfile):
        self.config = configparser.ConfigParser()
        self.config.read(configfile)
        
    def update(self,config_dict):
        """
        future func
        """
        return None