# pylint: disable=redefined-outer-name, unreachable, missing-function-docstring
"""
Image Processing Project 1: Adapted from Project 0
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.measure as measure
import skimage.filters as filters
import skimage.morphology as morph
import skimage.restoration as restore
import skimage.transform as transform
import predict as predict

def connectedComponents(image):
    out_im = np.zeros_like(image,dtype=int)
    image = morph.remove_small_holes(image, area_threshold=6)
    [image, num] = morph.label(image, return_num=True, connectivity=1)
    image = morph.remove_small_objects(image, min_size=6)
    if (num > 1): # prevents case where only one label is passed to remove_small_objects() which throws a warning
        labels = np.unique(image) # find all the labels in the image
        labels = np.delete(labels, [0]) # delete the background label
        label_num = 1
        for l in labels: # this renumbers the labels since r_small_obj doesn't
            out_im += ((image == l) * label_num)
            label_num += 1
        print("# labeled regions: " + str(label_num))
    else:
        out_im = image
    return out_im

def ocr(image):
    predictor = predict.CharacterPredictor(model_path="model.pth")
    denoised_im = restore.denoise_nl_means(image)
    threshold_im = np.round(denoised_im < filters.threshold_otsu(denoised_im)).astype(bool)
    #dilated_im = morph.binary_opening(threshold_im) # help with connecting thin components
    labeled_image = connectedComponents(threshold_im)
    props = ["label", "bbox", "image_intensity"]
    im_props = pd.DataFrame(measure.regionprops_table(labeled_image, intensity_image=(labeled_image > 0).astype(np.int8), properties=props))
    mapper = {  "bbox-0": "min_row", "bbox-1": "min_col", "bbox-2": "max_row", "bbox-3": "max_col"}
    im_props.rename(columns=mapper, inplace=True)
    im_props.eval("sum_x_y = min_row + min_col", inplace=True)
    letter_imgs = im_props.pop('image_intensity').to_numpy()
    letter_props = im_props.to_numpy()
    chars = []
    while(letter_props.shape[0] > 0):
        chars.append("")
        idx_tl = letter_props[:, 5].argmin() # index of the top left point is the smallest sum of min_x and min_y
        current_row_idxs = np.arange(letter_props.shape[0]) 
        current_row_idxs = current_row_idxs[(letter_props[:, 1] < letter_props[idx_tl, 3])]
        current_row_props = letter_props[current_row_idxs]
        current_row_imgs = letter_imgs[current_row_idxs]
        letter_props = np.delete(letter_props, current_row_idxs, axis=0) # remove the current indexes from the list of labels to be processed
        letter_imgs = np.delete(letter_imgs, current_row_idxs, axis=0) # remove the associated images
        current_row_imgs = current_row_imgs[current_row_props[:,2].argsort()] # order images left to right based on min_col, https://stackoverflow.com/a/2828121
        # figure = plt.figure()
        i = 0
        for letter_image in current_row_imgs:
            i += 1
            scaled_img = transform.rescale(letter_image, 28 / np.array(letter_image.shape).max(), anti_aliasing=False)
            pad_x_beg = 0
            pad_x_end = 0
            pad_y_beg = 0
            pad_y_end = 0
            if(scaled_img.shape != (28, 28)):
                row, col = scaled_img.shape
                pad_x_beg = int(np.floor((28 - col)/2))
                pad_x_end = int(np.ceil( (28 - col)/2))
                pad_y_beg = int(np.floor((28 - row)/2))
                pad_y_end = int(np.ceil( (28 - row)/2))
            padded_img = np.pad(scaled_img, pad_width=((pad_y_beg, pad_y_end),(pad_x_beg, pad_x_end)))
            char_img = (padded_img > 0).astype(np.int8)
            chars[-1] += predictor.predict(char_img)
            # ax1 = plt.subplot(2, len(current_row_idxs), i)
            # ax2 = plt.subplot(2, len(current_row_idxs), i + len(current_row_idxs))
            # ax1.set_title(chars[-1][-1])
            # ax1.imshow(char_img)
            # ax2.imshow(letter_image)
            # plt.setp(figure.axes, xticks=[], yticks=[])
        # plt.close('all')
        chars[-1] += "\n"
    return chars

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__)) + os.sep # https://stackoverflow.com/a/5137509
    img_path = "test_images_serif" + os.sep
    # get example image filepaths
    example_images = os.listdir(dir_path + img_path)
    split_text = np.array([temp.split('.') for temp in example_images]) # assumes filenames only have a single '.'
    img_filenames = list()
    img_filenames_extension = list()
    for i in range(len(split_text)):
        if split_text[i,-1] in {'jpg', 'png', 'bmp'}:
            img_filenames.append(split_text[i,0])
            img_filenames_extension.append(split_text[i,-2] + '.' + split_text[i,-1])
    
    print(img_filenames)  # print out the list of images in the project-1-images folder
    # find the number of rows and columns needed to display all images in a subplot
    # x = int(np.ceil(np.sqrt(len(files))))
    # y = int(np.ceil(len(files)/x))

    for i in range(len(img_filenames)):
        # read in the image to the variable im
        im = io.imread(dir_path + img_path + img_filenames_extension[i], as_gray=True) # import and convert image to gray scale
        scanned_text = ocr(image=im)
        # https://www.geeksforgeeks.org/writing-to-file-in-python/
        f = open(dir_path + img_path + img_filenames[i] + "_out.txt", "a")
        f.writelines(scanned_text)
        f.close()
    plt.show()  # make sure matplotlib shows the plot created above
    return

if __name__ == "__main__":
    main()