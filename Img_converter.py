import numpy as np
import os
from PIL import Image

class nimage():
    def __init__(self, im):
        try:
            self.x = np.asarray(Image.open(im).resize((128,128)))
        except IOError:
            self.x = np.zeros((128, 128))
            print("Can't find file")
