import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, Rectangle, Arc
import matplotlib as mpl
import random
import os
import numpy as np

exp_dir = os.getcwd()
stm_dir = os.path.join(exp_dir, 'stimulus_02')
if not os.path.exists(stm_dir):
    os.makedirs(stm_dir)
fig = plt.figure(frameon=False)
dpi = fig.get_dpi()
img_side = 512
half_side = img_side/2
width, height = img_side * 0.4, img_side * 0.4
fig.set_size_inches(img_side/dpi, img_side/dpi)
lw=3
orig = [0,0]

ax = plt.subplot(111)
ax.axis('off')
ax.set_xlim(-half_side, half_side)
ax.set_ylim(-half_side, half_side)
ax.axes.get_xaxis().set_visible(False)  #remove white padding
ax.axes.get_yaxis().set_visible(False)


upper_left = (int(orig[0] - width / 2), int(orig[1] - height / 2))
square1 = Rectangle(upper_left, width, height, fill=False, lw=lw)
ax.add_patch(square1)

circle1 = Ellipse(orig, width/2, height/2, fill=False, lw=lw)
circle2 = Ellipse(orig, width, height, fill=False, lw=lw)
circle3 = Ellipse((orig[0], width/2), width, height, fill=False, lw=lw)
circle4 = Ellipse((orig[0]+width/4, orig[1]+width/4), width, height, fill=False, lw=lw)
ax.add_patch(circle3)

fig.savefig(os.path.join(stm_dir, 'copy3.png'))