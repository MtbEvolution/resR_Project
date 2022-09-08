from .utils import *
from .utils import _make_mask

class MakeMask:
    
    def __init__(self, model, description = None, n_classes=1, n_channels=1, threshold=[0.5], verbose = True, **kwargs):
        """
        This function allows the users to conviniently use pre-implemented or separately trained models to do
        single- or multi-class mask prediction (semantic segmentation)
        """
        self.model = model
        self.description = description
        self.verbose = bool(verbose)
        self.threshold = threshold
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.kwargs = kwargs
    
    def predict(self, targets):
        """
        targets
        """
        if not isinstance(targets, list):
            targets = [targets]
            
        if self.verbose:
            if self.description is not None:
                print("Here's some extra information the user provided: {}.".format(self.description))
        
        # mask predictions are exported as list rather than np.array, considering instances where target images have different shapes
        predictions = []
        
        # check target channels
        for i,t in enumerate(targets):
            t = dimension_scaler(t)
            shape = t.shape
            expected_shape = tuple(list(shape[:-1])+[self.n_channels])
            if shape[-1]!=self.n_channels:
                error_msg = 'Expected image dimension is {} but {} was given.'.format(expected_shape, shape)
                raise ValueError(error_msg)
            probabilities = self.model.predict(t, **self.kwargs)
            binary = probabilities
            predictions.append(binary)
        return predictions
    
    
class BinarizeLegend:
    
    def __init__(self,method=5,**kwargs):
        self.method = method
        self.kwargs = kwargs
    
    def predict(self,target):
        target = np.squeeze(target)
        target = target.astype(float)
        target /= target.mean()
        return _make_mask(target,method=self.method,**self.kwargs)

