import numpy as np

def border_following(image):
    rows = image.shape[0]
    cols = image.shape[1]

    image  = np.pad(image, ((1,1), (1,1)), mode="constant", constant_values=0)

    first_foreground_pixel = 0

    neighbour_pixels = {"North":(-1, 0), "North-East":(-1, 1), "East":(0, 1), "South-East":(1, 1),
                        "South":(1, 0), "South-West":(1, -1), "West":(0, -1), "North-West":(-1, -1)}

    for i in rows:
        for j in cols:
            if(image[i, j] ==  1):
                first_foreground_pixel = (i,j)
                break
        break

    P = first_foreground_pixel
    D = "North"

    boundary = [P]