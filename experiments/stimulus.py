import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, Rectangle, Arc
import matplotlib as mpl
import random
import os
import numpy as np

exp_dir = os.getcwd()
stm_dir = os.path.join(exp_dir, 'stimulus')

img_side = 512  # image size is (img_size, img_size)
half_side = int(img_side / 2) # half of the image side
orig = (0,0)

num_of_pairs = 50 # the total number of stimuli pairs
min_snum, max_snum = 3, 6 # the range of possible numbers of primitives in each composite shape
max_scale,min_scale = 1.2, 0.8
max_rotation, min_rotation = 360, 0
max_move, min_move = int(half_side/4), -int(half_side/4)
default_width = half_side * 0.6
default_height = half_side * 0.6
lw = 3

for p in range(num_of_pairs):
    snum = random.randint(min_snum, max_snum)
    all_shape = [] # a list that contains all shape paths

    for i in range(snum):
        # randomly select a shape and its scaling, rotation, and translation parameters
        shape = random.randint(1,6)
        width, height = int(default_width * random.uniform(min_scale, max_scale)), int(default_height * random.uniform(min_scale, max_scale))
        rotation = random.randint(min_rotation, max_rotation)
        tx, ty = random.randint(min_move, max_move), random.randint(min_move, max_move)
        half_scalex, half_scaley = int(width / 2), int(height / 2)

        if shape == 1:
            shape_path = Ellipse(orig, width, height, fill=False, lw=lw)

        elif shape == 2:
            upper_left = (int(orig[0]-default_width/2), int(orig[1]-default_height/2))
            shape_path = Rectangle(upper_left, width, height, fill=False, lw=lw)

        elif shape == 3: # triangle
            half_scalex, half_scaley = int(width/2), int(height/2)
            p1, p2 = (orig[0]- half_scalex, orig[1]+ half_scaley), (orig[0]+half_scalex, orig[1]+half_scaley)
            p3 = (orig[0] + random.randint(0, width),orig[1]-half_scalex)
            shape_path = Polygon([p1, p2, p3], fill=False, lw=lw)

        elif shape == 4: #line
            p1, p2 = (orig[0] - half_scalex, orig[1]-half_scaley), (orig[0] + half_scalex, orig[1] + half_scaley)
            shape_path = Polygon([p1, p2], fill=False, lw=lw)

        elif shape == 5: # trapezoid
            half_width2 = int(default_width * random.uniform(min_scale, max_scale)/2)
            move = random.randint(min_move, max_move)
            p1, p2 = (orig[0] - half_scalex, orig[1] + half_scaley), (orig[0] + half_scalex, orig[1] + half_scaley)
            p3, p4 = (orig[0] + move+ half_width2, orig[1] - half_scalex), (orig[0] + move -half_width2, orig[1] - half_scalex)
            shape_path = Polygon([p1, p2, p3, p4], fill=False, lw=lw)

        elif shape == 6:
            degree = random.randint(60, 270)
            shape_path = Arc(orig, width, height, theta1=0, theta2=degree, fill=False, lw=lw)

        all_shape.append({'shape': shape_path, 'rotation':rotation, 'tx':tx, 'ty':ty})


    # create the stimulus in the experimental condition
    fig = plt.figure(frameon=False)
    dpi = fig.get_dpi()
    fig.set_size_inches(img_side/dpi, img_side/dpi)

    ax = plt.subplot(111)
    ax.axis('off')
    ax.set_xlim(-half_side, half_side)
    ax.set_ylim(-half_side, half_side)
    ax.axes.get_xaxis().set_visible(False)  #remove white padding
    ax.axes.get_yaxis().set_visible(False)

    for item in all_shape:
        transform = mpl.transforms.Affine2D().rotate_deg(item['rotation']).translate(item['tx'], item['ty']) + ax.transData
        item['shape'].set_transform(transform)
        ax.add_patch(item['shape'])

    fig.savefig(os.path.join(stm_dir, '{:03}_exp'.format(p + 1)))

    # create the stimulus in the control condition
    quarter_side = int(half_side / 2)  # 1/4 of the image width
    control_half_width = int((snum + 1) / 2) * quarter_side
    padding = img_side/32

    fig = plt.figure(frameon=False)
    fig.set_size_inches(control_half_width * 2 / dpi, img_side/ dpi)
    ax = plt.subplot(111)
    ax.axis('off')

    ax.set_xlim(-control_half_width-padding, control_half_width+padding)
    ax.set_ylim(-half_side-padding, half_side+padding)
    ax.axes.get_xaxis().set_visible(False)  #remove white padding
    ax.axes.get_yaxis().set_visible(False)

    for index, item in enumerate(all_shape):
        tx, ty = 0, (quarter_side + padding) * np.power(-1, index)
        center1 = int( (len(all_shape)-1)/2 )
        center2 = center1 + 1

        if center1 %2 == 0: # if there are an odd number of pairs of shapes
            if index > center2:
                tx = (half_side + padding) * int((index - center2 + 1) / 2)
            elif index < center1:
                tx = - (half_side + padding) * int((center1 - index + 1) / 2)
        else:
            if index <= center1:
                tx = - quarter_side - padding / 2 - (half_side + padding) * int((center1 - index) / 2)
            elif index >= center2:
                tx = quarter_side + padding / 2 + (half_side + padding) * int((index - center2) / 2)

        item['shape'].axes, item['shape'].figure = None, None # remove shapes from the old axes
        transform = mpl.transforms.Affine2D().translate(tx, ty) + ax.transData
        item['shape'].set_transform(transform)
        ax.add_patch(item['shape'])

    fig.savefig(os.path.join(stm_dir, '{:03}_control'.format(p + 1)), dpi=dpi)









