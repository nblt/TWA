import numpy as np

class CosineInc:
    def __init__(self, std:float, steps: int, inc: int):
    	self.base = std
    	self.halfwavelength_steps = steps
    	self.inc = inc

    def __call__(self, step):
        scale_factor = -np.cos(step * np.pi / self.halfwavelength_steps) * 0.5 + 0.5
        self.current = self.base * (scale_factor * self.inc + 1)
        return self.current