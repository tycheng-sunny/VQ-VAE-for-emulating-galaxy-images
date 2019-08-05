#===========================================================================================
# Transfer the image data to dictionary format for the loading in VQ-VAE code
# To-do list: need to update the code to use hef5 file to save a larger dataset
#===========================================================================================
import os
import numpy as np
import pickle
from astropy.io import fits

## Function
def maxmin(data):
    """ Normalised the images data to [0,1] """
    max = np.max(data)
    min = np.min(data)
    if max != min:
        data = ( data - min )/ np.float( max - min )
    else:
        data = data - min
    return data

## directory
PATH_TO_IMAGES = 'data/' # path to images
PATH_output_IMAGES = 'data/' # path to the output of dictionary
image_paths = os.listdir(PATH_TO_IMAGES)

## variables
image_size = 84

list_of_id, list_of_imgfn, list_of_images = [], [], []
for im in image_paths:
    image = fits.open(PATH_TO_IMAGES + im)
    img_data = image[0].data.astype(np.float32)
    img_data = maxmin(img_data)
    width, height = img_data.shape[0], img_data.shape[1]
    if (width, height) == (image_size, image_size):
        #print(width, height)
        list_of_id.append(im[:-7])
        list_of_imgfn.append(im)
        list_of_images.append(img_data)
list_of_images = np.array(list_of_images)
imdict = {'id': list_of_id, 'filename': list_of_imgfn, 'images': list_of_images}

#print(list_of_image_data)
with open(PATH_output_IMAGES + 'data.dict', 'wb') as handle:
    pickle.dump(imdict, handle, protocol=pickle.HIGHEST_PROTOCOL)
