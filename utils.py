# IMPORTS
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

from collections import Counter
from collections.abc import Iterable
from mpl_toolkits.axes_grid1 import ImageGrid
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
    return hex_colors, {hex_col: hex_col for hex_col in hex_colors}

# To crop image to a square shape
def crop_img(image):    
    img_h_saxis = image.shape[0]//2
    img_w_saxis = image.shape[1]//2
    crop_saxis = None

    if img_h_saxis <= img_w_saxis:
        crop_saxis = img_h_saxis
    else:
        crop_saxis = img_w_saxis

    center = (img_h_saxis, img_w_saxis)
    cropped_img = image[(center[0]-crop_saxis): (center[0]+ crop_saxis),
                        (center[1]-crop_saxis): (center[1]+ crop_saxis)]

    return cropped_img

# To determine the bw ratio
def chiaroscuro(image):
    w = 0
    b = 0

    for h in range(image.shape[0]):
        for w in range(image.shape[1]):
            if (image[h][w] == 255).all():
                w+=1
            elif (image[h][w] == 0).all():
                b+=1
            else:
                continue
        
    chiaroscuro = (w/b)
    
    return chiaroscuro

# To extract data from every image in a collection
def extract_img_data(path, img_collection, data_collection, target_class):
    img_errors = 0
    img_errors_log = []
    
    for i in range(len(img_collection)):
        # Get image and resize
        img_path = path + img_collection[i]
        img = get_img_rgb(img_path)
        
        # Resize and reduce color palette to 125 colors
        img = resize_img(img, 100)
        img = reduce_col_palette(img, 5)
        
        # Extract data of each image
        try:
            img_name = img_collection[i].split(sep='.')[0]
            img_ratio = round(round((img.shape[0] / img.shape[1]) * 2, ndigits=2) / 2, ndigits=5)
            img_colors, img_palette = color_quant(img, 5, num_of_colors=10, show_chart=False)
            img_fill = round(fill_ratio(img), ndigits=5)
            img_chi_osc = round(chiaroscuro(img), ndigits=5)
            
        except:
            img_errors += 1
            img_errors_log.append(img_collection[i])
            continue
         
        # Generate flat list with all the data   
        img_data = [img_name, target_class, img_ratio, img_fill, img_chi_osc]
                
        for i in range(len(img_colors)):
            img_data.append(img_colors[i])
        
        # Add img_data to features_list
        item_to_lists(img_data, data_collection)
        
    return img_errors, img_errors_log

# Transform HEX index to RGB index
def hex_to_rgb(color):
    c = color.lstrip('#')
    for i in (0, 2, 4): color = tuple(int(c[i: i+2], 16))
    
    return color

# To easily add each data in item to the correct list in lists
def item_to_lists(item, lists):
    for i in range(len(item)):
        lists[i].append(item[i])
        
    return lists

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

# To get all paths to valid images
def get_collection(working_path: str='', extensions: list=[]):
    '''
    This function will generate a list will all the paths of archives found
    in the selected working directory that have a valid extension.
    '''
    
    # Locate path and get all candidates path
    candidates = os.listdir(working_path)
    ### TODO - Get files in subfolders

    # Declare an empty list to append valid files path found iterating candidates
    collection = []
    valid_extension = extensions

    for file in candidates:
        # Check if image has a valid extension
        file_ext = os.path.splitext(file)[1]
        
        if file_ext.lower() in valid_extension: collection.append(file)
        else: continue
        
    return working_path, collection

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

# For quick empty list assignment
def mklist(n):
    for i in range(n):
        yield []
        
# To return the name of an object
def nameof(obj, namespace):
    return [name for name in namespace if namespace[name] is obj][0]

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

# To show all images from a collection
def show_collection(path, collection, cols):
    fig = plt.figure(figsize=(20, 20))
    grid = ImageGrid(fig,
                    111,
                    nrows_ncols=((len(collection)//cols)+1, cols),
                    axes_pad=0.1)

    for ax, im in zip(grid, collection):
        
        img_path = path + im
        image = mpimg.imread(img_path)
        image_res = resize_img(image, 50)
        
        # Iterating over the grid returns the Axes.
        ax.imshow(image_res)

    plt.show()
            
    return None