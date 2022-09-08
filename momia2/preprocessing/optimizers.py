import numpy as np
from ..utils import bandpass, generic, Config, correction

class GenericOptimizer:
    
    def __init__(self, func, channels='all', **kwargs):
        self.func = func
        self.kwargs = kwargs
        self.channels = channels
        
    def optimize(self, target_patch):
        self.channels = _assign_channel(target_patch, self.channels)
        for c in self.channels:
            image = target_patch.data[c]
            try:
                target_patch.data[c] = self.func(image, **self.kwargs)
            except:
                print('Something is wrong')

class DualBandpass(GenericOptimizer):
    
    def __init__(self):
        super().__init__(func=bandpass.dual_bandpass)
        
    def optimize(self, target_patch):
        if bool(int(target_patch.config['image']['bandpass'])):
            min_struct = float(target_patch.config['image']['bandpass_high'])
            max_struct = float(target_patch.config['image']['bandpass_low'])
            filtered = self.func(target_patch.get_ref_image(),
                                 target_patch.pixel_microns,
                                 min_struct,max_struct)
            target_patch.data[target_patch.ref_channel] = filtered
            
            
class RollingBall(GenericOptimizer):
    
    def __init__(self, **kwargs):
        super().__init__(func=correction.rolling_ball_bg_subtraction, **kwargs)
        
    def optimize(self, target_patch):
        from skimage.filters import gaussian
        for channel, img in target_patch.data.items():
            if channel != target_patch.ref_channel:
                img = self.func(img, **self.kwargs)
                target_patch.data[channel] = (gaussian(img, sigma=0.5) * 65535).astype(np.uint16)
                del img


class ToyOptimizer(GenericOptimizer):

    def __init__(self, **kwargs):
        super().__init__(func=toy_func, **kwargs)

            
            
            
def _assign_channel(target_patch, channels):
    if isinstance(channels, str):
        if channels == 'all':
            channels = list(target_patch.channels)
        elif channels == 'fluorescence':
            channels = [c for c in target_patch.channels if c != target_patch.ref_channel]
        elif channels == 'ref':
            channels = [target_patch.ref_channel]
        else:
            raise ValueError('Can not recognize provided channels "{}".'.format(channels))
    elif isinstance(channels, list):
        if len(channels) == 0:
            channels = list(target_patch.channels)
    return channels


def toy_func(data, **kwargs):
    print("Science is damn hard!")
    return data