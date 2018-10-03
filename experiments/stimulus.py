import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, Rectangle, Arc
import matplotlib as mpl
import random
import os
import numpy as np

exp_dir = os.getcwd()
stm_dir = os.path.join(exp_dir, 'stimulus')
if not os.path.exists(stm_dir):
    os.makedirs(stm_dir)

img_side = 512  # image size is (img_size, img_size)
half_side, quarter_side = int(img_side / 2), int(img_side / 4) # half of the image side
orig = (0,0)

num_of_pairs = 16  # the number of same/diff stimuli pairs
min_shape_num, max_shape_num = 4, 4
tran_angle = range(0, 360, 72)

# default attributes of primitive shapes
default_width = half_side * 0.6
default_height = half_side * 0.6
move = int(half_side / 8)
lw = 2.5
same_shape_list = [[True, False]] * num_of_pairs

def choose_shape(shape, default_width, default_height, move, angle, min_scale=1, max_scale=1, min_rotation=0, max_rotation=360, tran_rad=int(img_side / 6)):
    width, height = int(default_width * random.uniform(min_scale, max_scale)), int(
        default_height * random.uniform(min_scale, max_scale))
    rotation = random.randint(min_rotation, max_rotation)
    half_scalex, half_scaley = int(width / 2), int(height / 2)
    tx, ty = int(tran_rad * np.cos(np.radians(angle))), int(tran_rad * np.sin(np.radians(angle)))

    if shape == 1:  # ellipse
        shape_path = Ellipse(orig, width, height, fill=False, lw=lw)

    elif shape == 2:  # rectangle
        upper_left = (int(orig[0] - default_width / 2), int(orig[1] - default_height / 2))
        shape_path = Rectangle(upper_left, width, height, fill=False, lw=lw)

    elif shape == 3:  # triangle
        half_scalex, half_scaley = int(width / 2), int(height / 2)
        p1, p2 = (orig[0] - half_scalex, -orig[1] - half_scaley), (orig[0] + half_scalex, -orig[1] - half_scaley)
        p3 = (orig[0], orig[1] + half_scaley)
        shape_path = Polygon([p1, p2, p3], fill=False, lw=lw)

    elif shape == 4:  # line
        p1, p2 = (orig[0] - half_scalex, orig[1] - half_scaley), (orig[0] + half_scalex, orig[1] + half_scaley)
        shape_path = Polygon([p1, p2], fill=False, lw=lw)

    elif shape == 5:  # trapezoid
        p1, p2 = (orig[0] - half_scalex - move, orig[1] - half_scaley), (
        orig[0] + half_scalex + move, orig[1] - half_scaley)
        p3, p4 = (orig[0] - half_scalex + move, orig[1] + half_scaley), (
        orig[0] + half_scalex - move, orig[1] + half_scaley)
        shape_path = Polygon([p1, p2, p4, p3], fill=False, lw=lw)

    elif shape == 6:  # diamond
        p1, p3 = (orig[0], orig[1] + half_scaley + move), (orig[0], orig[1] - half_scaley - move)
        p2, p4 = (orig[0] - half_scalex, orig[1]), (orig[0] + half_scalex, orig[1])
        shape_path = Polygon([p1, p2, p3, p4], fill=False, lw=lw)

    elif shape == 7:  # rectangle
        width += (move * 2)
        upper_left = (int(orig[0] - width / 2), int(orig[1] - height / 2))
        shape_path = Rectangle(upper_left, width, height, fill=False, lw=lw)

    elif shape == 8:  # arc
        degree = random.randint(180, 180)
        shape_path = Arc(orig, width, height, theta1=0, theta2=degree, fill=False, lw=lw)

    return {'shape': shape_path, 'rotation':rotation, 'tx':tx, 'ty':ty}


# generate shapes
for sdi, sd_pairs in enumerate(same_shape_list):
    # for each same-shape/different-shape condition
    # fix the first shape and vary the other three shapes
    num_of_shape = random.randint(min_shape_num, max_shape_num)
    shape_move = np.random.choice(tran_angle, num_of_shape, replace=False)
    shape_init = np.random.choice(range(1, 9), num_of_shape, replace=False)
    shape1 = choose_shape(shape_init[0], default_width, default_height, move, shape_move[0])

    for p, same_shape in enumerate(sd_pairs):
        all_shape = [shape1]
        shape1['shape'].axes, shape1['shape'].figure = None, None  # remove shapes from the old axes

        if same_shape: # the stimulus only contains the same primitive shape
            shape_num = [shape_init[0]] * (num_of_shape - 1)
            file_code = 'same'
        else: # the stimulus contains different shapes
            shape_num = shape_init[1:]
            file_code = 'diff'

        for si, shape in enumerate(shape_num):
            # randomly select a shape and its scaling, rotation, and translation parameters
            shape_path = choose_shape(shape, default_width, default_height, move, shape_move[1:][si])
            all_shape.append(shape_path)


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

        fig.savefig(os.path.join(stm_dir, '{:03}_{}_composed'.format(sdi + 1, file_code)))

        # create the stimulus in the control condition
        control_half_width = int((num_of_shape + 1) / 2) * quarter_side
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

        fig.savefig(os.path.join(stm_dir, '{:03}_{}_grid'.format(sdi + 1, file_code)), dpi=dpi)








