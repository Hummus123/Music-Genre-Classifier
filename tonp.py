import numpy as np
import os

files = [f for f in os.listdir('.') if os.path.isfile(f)]
files = [i for i in files if i[-3:] == 'jpg' or i[-3:] == 'png']

