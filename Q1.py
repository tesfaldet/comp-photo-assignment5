import numpy as np
import matplotlib
matplotlib.use('tkagg')  # for rendering to work
import matplotlib.pyplot as plt
from scipy import misc
import glob
from scipy.ndimage import *
import time
import math


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])  # MATLAB style


def computeCumulativeOptCosts(image):
    # compute e1
    gray = rgb2gray(image)
    fx = correlate1d(gray, weights=[-1, 0, 1], axis=0)
    fy = correlate1d(gray, weights=[-1, 0, 1], axis=1)
    e1 = np.abs(fx) + np.abs(fy)

    # initialize M and N
    M = np.pad(e1, ((0,), (1,)), 'constant', constant_values=float('inf'))
    N = np.zeros((image.shape[0], image.shape[1])) + float('nan')

    for i in range(1, M.shape[0]):
        choices = [M[i-1, :-2], M[i-1, 1:-1], M[i-1, 2:]]
        min_vals = np.amin(choices, axis=0)
        min_inds = np.argmin(choices, axis=0)
        M[i, 1:-1] += min_vals
        N[i, :] = min_inds - 1  # optimal direction is either -1, 0, or 1

    return M[:, 1:-1], N


# assuming vertical seam
def optimalSeams(M, N, k=1):
    # find top k minimal costs at last row and start from there
    r = M.shape[0] - 1
    c = np.argsort(M[r])[:k]

    # storing k top seams
    seams = []

    for i in range(c.shape[0]):
        # initialize seam
        s = [(r, c[i])]

        # follow optimal path up to first row
        direction = N[r, c[i]]
        while not math.isnan(direction):
            r -= 1
            c[i] += int(direction)
            s.append((r, c[i]))
            direction = N[r, c[i]]

        seams.append(s)

        r = M.shape[0] - 1

    if k == 1:
        return np.array(seams)[0]
    else:
        return np.array(seams)


def removeSeam(image, s):
    # remove optimal seam
    mask = np.ones((image.shape[0], image.shape[1]), dtype=bool)
    mask[s[:, 0], s[:, 1]] = False
    image = image[mask].reshape((image.shape[0], -1, 3))

    return image


def expandSeams(image, seams):
    # expand by taking average of left and right neighbour
    pass


# retrieve images
filelist = glob.glob('T1/*.bmp')

# stack them into SxHxW
image1 = misc.imread(filelist[0])

# interactive imshow
image_old = image1.copy()
obj = plt.imshow(image1)
plt.ion()

# animate removal of seams
for i in range(1):
    # compute cumulative optimal energy and optimal directions
    M, N = computeCumulativeOptCosts(image1)

    # retrieve optimal seam
    s = optimalSeams(M, N)

    # highlight optimal seam
    image1[s[:, 0], s[:, 1], :] = [255.0, 0.0, 0.0]

    # display seam
    obj.set_data(image1)
    plt.pause(5)

    # remove optimal seam
    image1 = removeSeam(image1, s)

    # replace displayed image with seam removed image
    obj.set_data(image1)

plt.ioff()
plt.subplot(121)
plt.imshow(image_old)
plt.subplot(122)
plt.imshow(image1)
plt.show()
