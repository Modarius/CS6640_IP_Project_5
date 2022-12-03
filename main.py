# pylint: disable=redefined-outer-name, unreachable, missing-function-docstring
"""
Image Processing Project 1: Adapted from Project 0
"""


from copy import deepcopy
import os
import numpy as np
import pandas as pd
import time
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
    image = morph.remove_small_objects(image, min_size=4)
    image = morph.remove_small_holes(image, area_threshold=4)
    [labeled_im, num] = morph.label(image, return_num=True, connectivity=2)
    if (num > 1): # prevents case where only one label is passed to remove_small_objects() which throws a warning
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

def ocr(image):
    predictor = predict.CharacterPredictor(model_path="model.pth")
    denoised_im = restore.denoise_nl_means(image)
    threshold_im = np.round(denoised_im < filters.threshold_otsu(denoised_im)).astype(bool)
    labeled_im = connectedComponents(threshold_im)
    props = ["label", "bbox", "image_intensity"]
    im_props = pd.DataFrame(measure.regionprops_table(labeled_im, intensity_image=(labeled_im > 0).astype(np.int8), properties=props))
    mapper = {  "bbox-0": "min_row", "bbox-1": "min_col", "bbox-2": "max_row", "bbox-3": "max_col"}
    im_props.rename(columns=mapper, inplace=True)
    im_props.eval("sum_x_y = min_row + min_col", inplace=True)
    process_props = deepcopy(im_props)
    chars = []
    while(process_props.__len__() > 0):
        chars.append([])
        idx_tl = process_props['sum_x_y'].idxmin() # index of the top left point
        current_row = process_props.query("min_row <" + str(process_props.loc[idx_tl]['max_row'])).copy(deep=True) # get the characters in the current row
        process_props.drop(index=current_row.index,inplace=True) # remove them from the list of all properties
        current_row.sort_values(by="min_col", inplace=True) # order left to right
        i=0
        for idx in current_row.index:
            i += 1
            curr_img = current_row.loc[idx]['image_intensity']
            scaled_img = transform.rescale(curr_img, 26 / np.array(curr_img.shape).max(), anti_aliasing=False)
            pad_x_begin = 0
            pad_x_end = 0
            pad_y_begin = 0
            pad_y_end = 0
            if(scaled_img.shape != (28, 28)):
                row, col = scaled_img.shape
                pad_x_begin = int(np.floor((28 - col)/2))
                pad_x_end = int(np.ceil((28 - col)/2))
                pad_y_begin = int(np.floor((28-row)/2))
                pad_y_end = int(np.ceil((28-row)/2))
            padded_img = np.pad(scaled_img, pad_width=((pad_y_begin, pad_y_end),(pad_x_begin, pad_x_end)))
            char_img = (padded_img > 0).astype(np.int8)
            chars[-1].append(predictor.predict(char_img))
    return chars

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__)) + os.sep # https://stackoverflow.com/a/5137509
    img_path = "test_images" + os.sep
    # get example image filepaths
    example_images = os.listdir(dir_path + img_path)
    split_text = np.array([temp.split('.') for temp in example_images]) # assumes filenames only have a single '.'
    img_filenames = list()
    for i in range(len(split_text)):
        if split_text[i,-1] in {'jpg', 'png', 'bmp'}:
            img_filenames.append(split_text[i,-2] + '.' + split_text[i,-1])
    
    print(img_filenames)  # print out the list of images in the project-1-images folder
    # find the number of rows and columns needed to display all images in a subplot
    # x = int(np.ceil(np.sqrt(len(files))))
    # y = int(np.ceil(len(files)/x))

    for i in range(len(img_filenames)):
        # read in the image to the variable im
        im = io.imread(dir_path + img_path + img_filenames[i], as_gray=True) # import and convert image to gray scale
        ocr(image=im)
        print() # add an extra space between images
    plt.show()  # make sure matplotlib shows the plot created above
    return

if __name__ == "__main__":
    main()