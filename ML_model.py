import matplotlib.pyplot as plt
import numpy as np
import pandas
import os
from Img_converter import nimage
from Query import genres

files = [f for f in os.listdir('.') if os.path.isfile(f)]
files = [i for i in files if i[-3:] == 'jpg' or i[-3:] == 'png']
lis = []
genre_conv = [genres.index(i) for i in lis]

count = 0
for i in files:
    img = nimage(i)
    if type(img.x) != bool:
        if img.x.shape == (128, 128, 3):
            lis.append((nimage(i),i[:i.index('usic')+4]))

print(np.unique(np.array([genres.index(i[1]) for i in lis])))
tags = np.reshape(np.array([genres.index(i[1]) for i in lis]), (len([genres.index(i[1]) for i in lis]), 1))
dat = []
print(tags.shape)
print(len(dat))
g = np.append(np.reshape(np.array([i[0] for i in lis]), (len([i[0] for i in lis]), 1)), tags, 1)
np.save("Data", g)