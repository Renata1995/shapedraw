import math
import numpy as np


def draw_rect(draw, halfs, init_factor=0.8, scale=(1,1), translate=(0,0), rotation=0, fill=None, outline='black'):
    side = halfs * init_factor
    rect = np.array( [[-side/2, -side/2],[side/2, -side/2],[side/2, side/2], [-side/2, side/2]] )
    draw.polygon(transform(rect, halfs, scale, translate, rotation), fill=fill, outline=outline)


def draw_ellipse(draw, halfs, init_factor=0.8, scale=(1,1), translate=(0,0), fill=None, outline='black'):
    h, v = halfs * init_factor*scale[0]/2,halfs * init_factor*scale[1]/2
    pos_x, pos_y = halfs + translate[0], halfs + translate[1]
    draw.ellipse((pos_x-h, pos_y-v, pos_x+h, pos_y+v), fill=fill, outline=outline)


def draw_triangle(draw, halfs, init_factor=0.8, scale=(1,1), translate=(0,0), rotation=0, fill= None, outline='black'):
    side = halfs * init_factor
    triangle = np.array( [[-side/2, side/2],[side/2, side/2], [0, -side/2]] )
    draw.polygon(transform(triangle, halfs, scale, translate, rotation), fill=fill, outline=outline)


def draw_line(draw, halfs, init_factor=0.8, scale=(1,1), translate=(0,0), rotation=0, fill='black'):
    side = halfs * init_factor
    line = np.array( [[-side/2, 0],[side/2, 0]] )
    draw.line(transform(line, halfs, scale, translate, rotation), fill=fill)


def transform(shape, halfs, scale=(1,1), translate=(0,0), rotation=0):
    smatrix = np.array([[scale[0], 0], [0, scale[1]]])
    rmatrix = np.array([[math.cos(rotation), math.sin(rotation)], [-math.sin(rotation), math.cos(rotation)]])
    shape_sr = np.matmul( np.matmul(smatrix,rmatrix), shape.transpose() )
    pos_x, pos_y = halfs + translate[0], halfs + translate[1]
    shape_final = np.array([np.add(shape_sr[0], pos_x), np.add(shape_sr[1], pos_y)]).transpose()

    return ndarray_to_tuple(shape_final)


def ndarray_to_tuple(ndarray):
    final = []
    for item in ndarray:
        final.append(tuple(item))
    return tuple(final)




# def random_shape(draw, color):
#     shape = random.randint(1,9)
#     offset = 5
#     width = random.randint(halfs-offset, halfs+offset)
#     height = random.randint(halfs-offset, halfs+offset)



