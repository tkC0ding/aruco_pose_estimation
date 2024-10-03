import numpy as np


def get_neighbours(P, D):
    neighbour_intensities = []

    P = list(P)

    if(D == 0):
        []

def border_following(image):
    rows = image.shape[0]
    cols = image.shape[1]

    image  = np.pad(image, ((1,1), (1,1)), mode="constant", constant_values=0)

    first_foreground_pixel = 0

    neighbour_pixels = {0:(-1, 0), 1:(-1, 1), 2:(0, 1), 3:(1, 1),
                        4:(1, 0), 5:(1, -1), 6:(0, -1), 7:(-1, -1)}

    for i in rows:
        for j in cols:
            if(image[i, j] ==  1):
                first_foreground_pixel = (i,j)
                break
        break

    P = first_foreground_pixel
    D = 0

    boundary = [P]