# pylint: disable=redefined-outer-name, unreachable, missing-function-docstring
"""
Image Processing Project 1: Adapted from Project 0
"""


import os
import string
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.colors as colors
import skimage.measure as measure
import skimage.morphology as morph
import skimage.exposure as exp
import skimage.util as imageutil

import time

def getHistogram(im, bins):
    histogram = np.zeros(bins, np.uint32) 
    if (np.any(im > 1)):
        im = im / 255  # normalize pixel intensity to [0,1]
    im = np.int32(np.round(im * (bins - 1)))  # scale pixel intensity to [0, bins-1]
    flat_im = np.around(im.flatten()) # flatten matrix to 1D array
    for i in np.arange(bins):
        histogram[i] = len(np.extract(np.equal(flat_im, i), flat_im)) # get the number of pixels equal to the bin number
    return histogram

def threshold(im, low, high):
    if (low > high): # basic error checking 
        temp = low
        low = high
        high = temp
    # if (np.any(im > 1)): # shouldn't be needed but good to check
    #     im = im / 255  # normalize pixel intensity to [0,1]
    # im = np.int32(np.around(im * 255))  # scale pixel intensity to [0, 255] (256 intensity values)
    # get binary if pixel intensity is out of bounds, then invert it to get whether it is in the threshold range
    threshold = ~np.array((im < low) | (im > high)) 
    return threshold

def connectedComponents(im, min_size):
    out_im = np.zeros_like(im,dtype=int)
    [labeled_im, num] = measure.label(im, return_num=True, connectivity=2)
    if (num > 1): # prevents case where only one label is passed to remove_small_objects() which throws a warning
        labeled_im = morph.remove_small_objects(labeled_im, min_size=min_size)
        labels = np.unique(labeled_im) # find all the labels in the image
        labels = np.delete(labels, [0]) # delete the background label
        label_num = 1
        for l in labels: # this renumbers the labels since r_small_obj doesn't
            out_im += ((labeled_im == l) * label_num)
            label_num += 1
        print("# labeled regions: " + str(label_num))
    else:
        out_im = labeled_im
    return out_im

def histogramEqualization(im):
    hist, bin_edges = np.histogram(im, 256)
    p_r = hist/im.size # normalize the histogram to get probability dist. function
    c_r = np.cumsum(p_r) # get cumulative distribution function (discrete)
    c_k = np.uint8(np.round(c_r * 255)) # scale the cumulative distribution function to entire intensity range
    out = np.zeros_like(im)

    # use orig. image values to index into scaled cumulative d. function to get output intensity, requires
    # reshaping to use the python indexing
    out = np.reshape(c_k[im.flatten()], im.shape) 
    return out

def main():
    # list of input files
    # found at https://stackoverflow.com/a/5137509
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # this directory must contain images only, no folders or other filetypes
    files = os.listdir(dir_path + "/project-1-images")
    print(files)  # print out the list of images in the project-1-images folder
    # find the number of rows and columns needed to display all images in a subplot
    # x = int(np.ceil(np.sqrt(len(files))))
    # y = int(np.ceil(len(files)/x))

    for i in range(len(files)):
        bins = 256
        # read in the image to the variable im
        im = Image.open(dir_path + "/project-1-images/" + files[i])
        im = np.asarray(im.convert(mode="L")) # convert image to gray scale (mode="L")

        # Question 1
        fig, ax = plt.subplots(2,4, figsize=(12,6))
        plt.subplot(2,4,1)
        plt.imshow(im, cmap='gray')
        plt.title("Image: " + files[i])
        t = time.time()
        histogram = getHistogram(im, bins)
        print("Histogram time: " + str(time.time() - t) + "s for " + files[i])# prints out runtime, for debugging/ speed
        plt.subplot(2,4,5)
        plt.bar(np.arange(0, 256, 256/bins), histogram)
        plt.title("Image Histogram")

        # Question 2.1
        t = time.time()
        thresh_im = threshold(im, 100, 220) # threshold limits 100 <-> 220
        print("Threshold time: " + str(time.time() - t) + "s for " + files[i]) # prints out runtime, for debugging/ speed
        plt.subplot(2,4,2)
        plt.imshow(thresh_im, cmap='gray')
        plt.title("Threshold")
        plt.subplot(2,4,6)
        thresh_hist, bin_edges = np.histogram(im * thresh_im,bins)
        plt.bar(np.arange(1, 256, 256/bins), thresh_hist[1:]) # ignore zeros, there are a lot of them
        plt.title("Thresh. Histogram")

        # Question 2.2
        t = time.time()
        cc = connectedComponents(thresh_im, int(thresh_im.size/500)) # minumum size is 1/500th of the number of pixels in the image
        print("Conn. Comp. time: " + str(time.time() - t) + "s for " + files[i])# prints out runtime, for debugging/ speed
        plt.subplot(2,4,3)
        plt.imshow(cc, cmap='tab10')
        plt.title("Conn. Comps.")

        eq_image = histogramEqualization(im) # call my equalization program
        plt.subplot(2,4,4)
        plt.imshow(eq_image, cmap='gray') # show the equalized image
        plt.title("Equalized")
        plt.subplot(2,4,8) 
        eq_hist, bin_edges = np.histogram(eq_image,bins) # get the histogram
        plt.bar(np.arange(0, 256, 256/bins), eq_hist) # show the histogram of the equalized image
        plt.title("Equalized Histogram")
        
        #fig.subplots_adjust(wspace=0.45)
        for curr_ax in ax.flatten():
            curr_ax.set_box_aspect(1.0) # set the plots to be square
            curr_ax.ticklabel_format(style='sci', axis='y', scilimits=(0,2)) # format x axis to display in scientific notation
        plt.subplot(2,4,7).remove() # there is no histogram for connected components so don't show the plot there
        plt.savefig("output" + os.path.sep + files[i].split('.')[0])
        plt.close('all')

        kern_size = [4, 500, 100, 100] # different options of kernal size to test
        clip_lim = [.5, .5, .8, .25] # different options of clip limits to test (in conjunction with kern_size)
        length = len(kern_size)
        fig, ax = plt.subplots(2,length,figsize=(8,4))
        for j in np.arange(length):
            adapt_hist_eq_image = exp.equalize_adapthist(im, kernel_size=kern_size[j], clip_limit=clip_lim[j]) # adaptive equalization
            plt.subplot(2,length,j+1)
            plt.imshow(adapt_hist_eq_image, cmap='gray')
            plt.title("K: " + str(kern_size[j]) + " C: " + str(clip_lim[j]))
            plt.subplot(2,length,length + j + 1)
            adapt_hist, bin_edges = np.histogram(eq_image,bins) # get the histogram
            plt.bar(np.arange(0, 256, 256/bins), adapt_hist) # show the histogram of the equalized image
            plt.title("Adapt. Hist.")

        for curr_ax in ax.flatten():
            curr_ax.set_box_aspect(1.0) # set the plots to be square
            curr_ax.ticklabel_format(style='sci', axis='y', scilimits=(0,2)) # format x axis to display in scientific notation
        plt.savefig("output" + os.path.sep + files[i].split('.')[0] + "_ah.png")
        plt.close('all')

        print() # add an extra space between images
    plt.show()  # make sure matplotlib shows the plot created above
    return

if __name__ == "__main__":
    main()


# def floodFill(px_x, px_y, im, threshold):
#     # neighborhood can be modified to include any pixel offsets desired
#     neighborhood = [(-1, 0), # left
#                     ( 0, 1), # up
#                     ( 1, 0), # right
#                     ( 0,-1)] # down
#     width = im.shape[1]
#     height = im.shape[0]
#     out_im = np.int(np.zeros_like(im))

#     # check if out of bounds or if the current pixel is false
#     if (px_x < 0 or px_x > width
#         or px_y < 0 or px_y > height
#         or ~threshold[px_y, px_x]): 
#         return out_im

#     # create new set of pixel locations to check (sets contain only unique values)
#     px_set = {(px_x, px_y)}
#     while(len(px_set) > 0):
#         (px_x, px_y) = px_set.pop()
#         for n in neighborhood: # loop through all the pixels in the current neighborhood
#             test_x = px_x + n[0]
#             test_y = px_y + n[1]
#             if not (test_x < 0 or test_x > width - 1 or test_y < 0 or test_y > height - 1): # check if out of bounds
#                 if (threshold[test_y, test_x] and out_im[test_y, test_x] == 0): # if it passes the test function and is not already marked as a connected component
#                     px_set.add((test_x,test_y)) # add the location to the pixel location set to later check its neighbors
#                     out_im[test_y, test_x] = 1  # mark that pixel as being part of a connected set of pixels   
#     return out_im

# def connectedComponents(im, threshold):
#     cc = np.int32(np.zeros_like(im))
#     component = 1
#     px_to_check = np.fliplr(np.transpose(np.nonzero(threshold))) # https://stackoverflow.com/a/21815619
#     for i in np.arange(np.size(px_to_check,0)):
#         curr_px = px_to_check[i]
#         if (cc[curr_px[1], curr_px[0]] == 0): # if there isn't already a connected component at that coordinate, check it
#             ff = floodFill(curr_px[0], curr_px[1], im, threshold) # make returned set have correct label number
#             if (np.sum(ff) > 200):
#                 cc += ff * component       # add the new connected component to the output buffer cc
#                 component += 1  # increment component number
#     return cc