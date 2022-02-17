# Modules used for data handling / test
import csv
import time

from utils import get_collection


# Modules used for image processing
from utils import extract_img_data


start_time = time.time()

valid_extensions = ['.jpg', '.jpeg']

dset_caravaggio = get_collection('./images/raw_museum/caravaggio/', valid_extensions)
dset_degas = get_collection('./images/raw_museum/degas', valid_extensions)
dset_goya = get_collection('./images/raw_museum/goya', valid_extensions)
dset_hokusai = get_collection('./images/raw_museum/hokusai', valid_extensions)
dset_kahlo = get_collection('./images/raw_museum/kahlo', valid_extensions)
dset_kandinsky = get_collection('./images/raw_museum/kandinsky', valid_extensions)
dset_klimt = get_collection('./images/raw_museum/klimt', valid_extensions)
dset_lichtenstein = get_collection('./images/raw_museum/lichtenstein', valid_extensions)
dset_mondrian = get_collection('./images/raw_museum/mondrian', valid_extensions)
dset_monet = get_collection('./images/raw_museum/monet', valid_extensions)
dset_picasso = get_collection('./images/raw_museum/picasso', valid_extensions)
dset_pollock = get_collection('./images/raw_museum/pollock', valid_extensions)
dset_sorolla = get_collection('./images/raw_museum/sorolla', valid_extensions)
dset_velazquez = get_collection('./images/raw_museum/velazquez', valid_extensions)
dset_warhol = get_collection('./images/raw_museum/warhol', valid_extensions)

# By now, the museum is just a index
museum = [dset_caravaggio, dset_degas, dset_goya, dset_hokusai, dset_kahlo,
          dset_kandinsky, dset_klimt, dset_lichtenstein, dset_mondrian, dset_monet,
          dset_picasso, dset_pollock, dset_sorolla, dset_velazquez, dset_warhol]

artists = ['caravaggio',
           'degas',
           'goya',
           'hokusai',
           'kahlo',
           'kandinsky',
           'klimt',
           'lichtenstein',
           'mondrian',
           'monet',
           'picasso',
           'pollock',
           'sorolla',
           'velazquez',
           'warhol']

big_data_collections = [artist + '_big_collection' for artist in artists]
big_errors_log = [artist + '_big_errors' for artist in artists]

# Build big_museum
big_museum = []

for artist, data_collection, img_collection, errors_log in zip(artists, big_data_collections, museum, big_errors_log):
    data_collection, errors_log = extract_img_data(img_collection,
                                                   square_crop=False,
                                                   resize=True,
                                                   height=500,
                                                   limit_colors=True,
                                                   colors_per_channel=6,
                                                   target_class=artist,
                                                   save=True,
                                                   save_path='./images/big_museum')
    
    big_museum.append(data_collection)
    
big_museum    = [item for sublist in big_museum for item in sublist]

with open('./data/big_museum.csv', "w", newline="") as datafile:
    writer = csv.writer(datafile)            
    writer.writerows(big_museum)
    
print(f'Execution time: {time.time() - start_time}')