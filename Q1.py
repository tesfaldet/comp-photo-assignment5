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


def computeCumulativeOptCosts(image, mask=None):
    # compute e1
    gray = rgb2gray(image)
    fx = correlate1d(gray, weights=[-1, 0, 1], axis=0)
    fy = correlate1d(gray, weights=[-1, 0, 1], axis=1)
    e1 = np.abs(fx) + np.abs(fy)

    # mask represents places to avoid
    if mask is not None:
        e1[mask[:, 0], mask[:, 1]] = float('inf')

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
def optimalSeam(M, N):
    # find minimal cost at last row and start from there
    r = M.shape[0] - 1
    c = np.argmin(M[r])

    # initialize seam
    s = [(r, c)]

    # follow optimal path up to first row
    direction = N[r, c]
    while not math.isnan(direction):
        r -= 1
        c += int(direction)
        s.append((r, c))
        direction = N[r, c]

    return np.array(s)


def removeSeam(image, s):
    # remove optimal seam
    mask = np.ones((image.shape[0], image.shape[1]), dtype=bool)
    mask[s[:, 0], s[:, 1]] = False
    image = image[mask].reshape((image.shape[0], -1, 3))

    return image


def animateSeamRemoval(image, remove_rows, remove_cols):
    # interactive imshow
    image_old = image.copy()
    obj = plt.imshow(image)
    plt.ion()

    def removeSeams(image, obj, remove, transpose=False):
        if transpose:
            image = np.transpose(image, (1, 0, 2))

        # animate removal of seams
        for i in range(remove):
            # compute cumulative optimal energy and optimal directions
            M, N = computeCumulativeOptCosts(image)

            # retrieve optimal seam(s)
            s = optimalSeam(M, N)

            # highlight optimal seam
            image[s[:, 0], s[:, 1], :] = [255.0, 0.0, 0.0]

            # display seam
            if transpose:
                obj.set_data(np.transpose(image, (1, 0, 2)))
            else:
                obj.set_data(image)
            plt.pause(0.05)

            # remove optimal seam
            image = removeSeam(image, s)

            # replace displayed image with seam removed image
            if transpose:
                obj.set_data(np.transpose(image, (1, 0, 2)))
            else:
                obj.set_data(image)

        if transpose:
            return np.transpose(image, (1, 0, 2))
        else:
            return image

    # remove columns
    image = removeSeams(image, obj, remove_cols)

    # remove rows
    image = removeSeams(image, obj, remove_rows, transpose=True)

    plt.ioff()
    plt.subplot(121)
    plt.imshow(image_old)
    plt.subplot(122)
    plt.imshow(image)
    plt.show()

    return image


def animateSeamExpansion(image, expand_rows, expand_cols):
    image = image.astype('float32')

    # interactive imshow
    image_old = image.copy()
    obj = plt.imshow(image.astype('uint8'))
    plt.ion()

    def expandSeams(image, obj, expand, transpose=False):
        if transpose:
            image = np.transpose(image, (1, 0, 2))

        seams = []
        mask = None

        # get vertical seams
        for j in range(expand):
            # compute cumulative optimal energy and optimal directions
            M, N = computeCumulativeOptCosts(image, mask)

            # retrieve optimal seam(s)
            s = optimalSeam(M, N)
            seams.append(s)

            mask = np.concatenate(seams)

        # expand by replacing seam with two seams:
        # 1. seam which is the average of the optimal seam and its left
        # neighbour,
        # 2. seam which is the average of the optimal seam and its right
        # neighbour
        for k in range(len(seams)):
            # take seam
            s = seams[k]

            image_seam = image[s[:, 0], s[:, 1], :]

            # pad left and right sides with zeros to deal with out of bounds
            image = np.pad(image, ((0, 0), (1, 1), (0, 0)), 'constant')
            avg_left = (image_seam + image[s[:, 0], s[:, 1], :]) / 2.0
            avg_right = (image_seam + image[s[:, 0], s[:, 1]+2, :]) / 2.0

            # truncate left and right sides
            image = image[:, 1:-1, :]

            # highlight optimal seam
            image[s[:, 0], s[:, 1], :] = [255.0, 0.0, 0.0]

            # display seam
            if transpose:
                obj.set_data(np.transpose(image.astype('uint8'), (1, 0, 2)))
            else:
                obj.set_data(image.astype('uint8'))
            plt.pause(0.05)

            # place back old seam
            image[s[:, 0], s[:, 1], :] = image_seam

            # shift all columns to the right of this seam by one
            image = np.pad(image, ((0, 0), (0, 1), (0, 0)), 'constant')
            for i in range(s.shape[0]):
                image[s[i, 0], s[i, 1]+1:, :] = image[s[i, 0], s[i, 1]:-1, :]

            # shift all other seams to the right of this seam by one
            for i in range(k+1, len(seams)):
                # check if other seam's first pixel is to the right of the
                # current seam's first pixel
                if seams[i][0, 1] > s[0, 1]:
                    seams[i][:, 1] += 1  # shift to the right

            # insert new average seams
            image[s[:, 0], s[:, 1], :] = avg_left
            image[s[:, 0], s[:, 1]+1, :] = avg_right

            # replace displayed image with seam expanded image
            if transpose:
                obj.set_data(np.transpose(image.astype('uint8'), (1, 0, 2)))
            else:
                obj.set_data(image.astype('uint8'))

        if transpose:
            return np.transpose(image, (1, 0, 2))
        else:
            return image

    # expand columns
    image = expandSeams(image, obj, expand_cols)

    # expand rows
    image = expandSeams(image, obj, expand_rows, transpose=True)

    plt.ioff()
    plt.subplot(121)
    plt.imshow(image_old.astype('uint8'))
    plt.subplot(122)
    plt.imshow(image.astype('uint8'))
    plt.show()

    return image.astype('uint8')


# retrieve images
filelist = glob.glob('T1/*.bmp')

image1 = misc.imread(filelist[0])
image1 = animateSeamRemoval(image1, remove_rows=10, remove_cols=10)
image1 = animateSeamExpansion(image1, expand_rows=10, expand_cols=10)

image2 = misc.imread(filelist[1])
image2 = animateSeamRemoval(image2, remove_rows=10, remove_cols=10)
image2 = animateSeamExpansion(image2, expand_rows=10, expand_cols=10)
