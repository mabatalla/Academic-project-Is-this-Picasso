import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# To import image in RGB mode
def get_image_rgb(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (1200, 600))
    return image

# To import image in greyscale mode
def get_image_gre(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (1200, 600))
    return image

# Transform RGB index to HEX index
def rgb2hex(color):
    return f'#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}'

def get_colors(image, num_of_colors, show_chart=True):
    # Resize the image to reduce processing time
    image_res = cv2.resize(image,
                           (600, 400),
                           interpolation=cv2.INTER_AREA)
    
    # Reshape of the image to get np.array 1D (KMeans requirement)
    image_res = image_res.reshape(image_res.shape[0]*image_res.shape[1], 3)
    
    # Use of KMeans to generate num_of_colors of clusters
    model_clf = KMeans(n_clusters=num_of_colors)
    labels = model_clf.fit_predict(image_res)
    
    counts = Counter(labels)
    
    # Sort the counts to ensure correct color percentage
    counts = dict(sorted(counts.items()))
    
    colors = model_clf.cluster_centers_
    
    # Get ordered colors iterating through the keys
    ordered_colors = [colors[i] for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    
    # Transform RGB index to HEX index
    hex_colors = [rgb2hex(ordered_colors[i]) for i in counts.keys()]

    if (show_chart):
        plt.figure(figsize=(10, 10))
        plt.pie(counts.values(),
                labels=hex_colors,
                colors=hex_colors)
    
    return rgb_colors