# IMPORTS
import csv
from email import message
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import pathlib

from collections import Counter
from collections.abc import Iterable
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.cluster import KMeans
from typing import Collection


# FUNCS
# To extract colors from image
def color_quant(image, bins=5, num_of_colors=10, show_chart=True):
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
        # # Inform user
        # print('Colors found:')
        
        # plt.figure(figsize=(10, 10))
        # plt.pie(counts.values(),
        #         labels=hex_colors,
        #         colors=hex_colors)
        
        plot_colors({hex_col: hex_col for hex_col in hex_colors},
                    'Colors found',
                    sort_colors=True,
                    emptycols=1)

        plt.show()
        
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
def chi_osc(image):
    b = 0
    w = 0

    for h in range(image.shape[0]):
        for w in range(image.shape[1]):
            if (image[h][w] == 0).all():
                b+=1
            elif (image[h][w] == 255).all():
                w+=1
            else: 
                continue
        
    chiaroscuro = (w/b)
    
    return chiaroscuro

# To extract data from every image in a collection
def extract_img_data(img_collection,
                     square=False,
                     resize=True,
                     height=100,
                     limit_colors=True,
                     colors_per_channel=5,
                     target_class='',
                     save=False,
                     save_path=''):

    data_collection = []
    
    errors = 0
    errors_log = []

    if save:
        origin = os.getcwd()
        
        # Create new folder in save_path
        os.chdir(save_path)
        os.mkdir(target_class)
        
        # Go back to working path
        os.chdir(origin)
    
    for work in img_collection:
        work = str(work)
        
        # Get image
        img = get_img_rgb(work)
        
        # Crop and resize image
        if square: img = crop_img(img)
        if resize: img = resize_img(img, height)
                
        # Extract basic data
        img_name = work.split(sep='/')[-1].split(sep='.')[0]
        
        img_data = [img_name, target_class, img.shape[0], img.shape[1]]
    
        # Reduce color palette to set colors_per_channel and extract colors
        if limit_colors:
            try:
                img = reduce_col_palette(img,
                                         colors_per_channel)
                img_colors, img_palette = color_quant(img,
                                         bins=5,
                                         num_of_colors=10,
                                         show_chart=False)
                
                # Expand img_data with color cluster information
                img_data.append(round(whitespace(img), ndigits=5))
                img_data.append(round(chi_osc(img), ndigits=5))
                for i in range(len(img_colors)):
                    img_data.append(img_colors[i])
                
            except:
                errors += 1
                errors_log.append(work)
                continue
        
        # Add img_data to data_collection            
        data_collection.append(img_data)
        
        # Save image
        if save:
            filename = work.split(sep='/')[-1]
            
            # Revert img to BGR before saving
            img_to_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Save image
            cv2.imwrite(f'{save_path}/{target_class}/{filename}', img_to_save)
    
    # Save img_data
    if save:
        with open(f'{save_path}/{target_class}/{target_class}.csv', "w", newline="") as datafile:
            writer = csv.writer(datafile)            
            writer.writerows(data_collection)
            
    # Inform user
    message = f'{errors} errors raised from {len(img_collection)} pictures in {target_class} collection.'
    
    print(message)
    errors_log.append(message)
        
    return data_collection, errors_log

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

# To get all paths to valid images
def get_collection(path, extensions=[]):
    '''
    This function will generate a list will all the paths of archives found
    in the selected working directory that have a valid extension.
    '''
    
    # Declare an empty list to append valid files path
    collection = []
    
    # Iterate through the files tree from path
    for path, folders, files in os.walk(path):
        # If image has a valid extension append path to collection
        for name in files:   
            file_ext = os.path.splitext(name)[1]
            
            if file_ext.lower() in extensions:
                collection.append(pathlib.PurePath(path, name))
            else:
                continue
        
    return collection

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

# To resize img keeping ratio
def resize_img(image, height):
    ratio = image.shape[0]/image.shape[1]
    width = int(height/ratio)
     
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

# Transform RGB index to HEX index
def rgb_to_hex(color):
    
    return f'#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}'

# To plot colors found in an image
def plot_colors(colors, title, sort_colors=False, emptycols=0):
    cell_width = 200
    cell_height = 50
    swatch_width = 100
    margin = 12
    topmargin = 40

    ### TODO - Sort colors by RGB index
        
    names = list(colors)
    
    n = len(names)
    ncols = 4 - emptycols
    nrows = n // ncols + int(n % ncols > 0)

    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + margin + topmargin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, 
                        margin/height,
                        (width-margin)/width,
                        (height-topmargin)/height)
    
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows),
                -cell_height/2.)
    
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    
    ax.set_axis_off()
    
    ax.set_title(title,
                 fontsize=24,
                 loc="left",
                 pad=10)

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x,
                y+(cell_height/4),
                name,
                fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(Rectangle(xy=(swatch_start_x, y-9),
                               width=swatch_width,
                               height=40,
                               facecolor=colors[name],
                               edgecolor='0.5'))

    return fig

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
def show_collection(collection, cols: int=5):
    fig = plt.figure(figsize=(20, 20))
    grid = ImageGrid(fig,
                    111,
                    nrows_ncols=((len(collection)//cols)+1, cols),
                    axes_pad=0.5)

    for ax, work in zip(grid, collection):
        img_name = str(work).split(sep='/')[-1].split(sep='.')[0]
        ax.set_title(img_name)
        
        img = mpimg.imread(str(work))
        img_res = resize_img(img, 50)
        
        ax.imshow(img_res)

    plt.show()
            
    return None

# To determine the filling_ratio
def whitespace(image):
    w = 0
    non_w = 0

    for h in range(image.shape[0]):
        for w in range(image.shape[1]):
            if (image[h][w] == 255).all():
                w+=1
            else:
                non_w+=1
        
    whitespace_ratio = (w*100)/(image.shape[0]*image.shape[1])
    
    return whitespace_ratio