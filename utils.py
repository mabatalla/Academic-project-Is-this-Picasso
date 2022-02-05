# IMPORTS
import cv2
import matplotlib.pyplot as plt
import numpy as np

from collections import Counter
from sklearn.cluster import KMeans


# FUNCS
# To extract colors from image
def color_quant(image, bins, num_of_colors=10, show_chart=True):
    '''
    This function applies a color quantization based on cv2.kmeans function
    as described in the OpenCV docs to reduce the colors present in an image.
    '''
    
    # Reshape of the image to get np.array 1D (KMeans requirement)
    image = image.reshape(image.shape[0]*image.shape[1], 3)
    
    # Use of KMeans to generate num_of_colors of clusters
    model_clf = KMeans(n_clusters=num_of_colors)
    labels = model_clf.fit_predict(image)
    
    counts = Counter(labels)
    
    # Sort the counts to ensure correct color percentage
    counts = dict(sorted(counts.items()))
    
    colors = model_clf.cluster_centers_
    
    # Transform color indexes to the reduced palette
    for color in colors:
            color[0] = process_color(color[0], bins)
            color[1] = process_color(color[1], bins)
            color[2] = process_color(color[2], bins)
    
    # Get ordered colors and presence iterating through the keys
    ordered_colors = [colors[i] for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    
    # Transform RGB index to HEX index
    hex_colors = [rgb_to_hex(ordered_colors[i]).upper() for i in counts.keys()]

    if (show_chart):
        # Inform user
        print('Colors found:')
        
        plt.figure(figsize=(10, 10))
        plt.pie(counts.values(),
                labels=hex_colors,
                colors=hex_colors)
        
    # return rgb_colors
    return {hex_col: hex_col for hex_col in hex_colors}

# Transform HEX index to RGB index
def hex_to_rgb(color):
    c = color.lstrip('#')
    for i in (0, 2, 4): color = tuple(int(c[i: i+2], 16))
    
    return color

# To determine the filling_ratio
def fill_ratio(image):
    w = 0
    non_w = 0

    for h in range(image.shape[0]):
        for w in range(image.shape[1]):
            if (image[h][w] == 255).all():
                w+=1
            else:
                non_w+=1
        
    filling_ratio = (non_w*100)/(image.shape[0]*image.shape[1])
    
    return filling_ratio

# To import image in RGB mode
def get_img_rgb(image_path):
    '''
    By default, OpenCV reads image in the sequence Blue Green Red (BGR).
    Thus, to view the actual image we need to convert the rendering
    to Red Green Blue (RGB).
    '''
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image

# To resize keeping ratio
def resize_img(image, height):
    ratio = image.shape[0]/image.shape[1]
    width = int(height / ratio)
     
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

# Transform RGB index to HEX index
def rgb_to_hex(color):
    
    return f'#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}'

# To map a rgb channel color value to a reduced palette
def process_color(channel_value, bins):
    '''
    This function is used to map the values of RGB in a pixel that
    go from 0 to 255 to a simple version with X values that lead to
    a palette of Y colors (Xx Xx Xx = Y).
    Eg. X = 4, then y = 4x4x4 = 64 colors
    '''
    
    if channel_value >= 255: processed_value = 255
    else:
        preprocessed_value = np.floor((channel_value*bins)/255)
        processed_value = abs(int(preprocessed_value*(255/(bins-1))))
    
    return processed_value

# To map all pixels of an image to a reduced palette
def reduce_col_palette(image, bins, info=False):
    '''
    This function iterate through every pixel of an image to map
    each rgb channel color value to a reduced palette.
    '''
    
    # Capture image dimensions
    img = image.flatten()
       
    # Iterate the array to transform the value of the pixels
    for px in range(len(img)):
        if img[px] == 255: img[px] = 255
        else: img[px] = process_color(img[px], bins)
    
    # Restore image shape
    img = np.reshape(img, image.shape)
        
    # Inform user
    if info:
        print(f'Palette reduced to {bins**3} colors.')
            
    return img