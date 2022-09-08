import numpy as np
import pandas as pd
from skimage import measure, morphology, filters, features

class GenericInstanceModel:
    
    def __init__(self, func, **kwargs):
        self.func = func
        self.kwargs = kwargs
        
    def predict(self, mask, intensity_image=None):
        return self.func(mask)