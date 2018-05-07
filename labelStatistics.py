"""
Computes statistics about labeled/annotated training dataset.
"""

from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from PIL import Image

import os
import logging
import sys
import argparse


# Set up logging
logger = logging.getLogger()

#Set up file handler:
#fh = logging.FileHandler(os.path.basename(__file__) + str(datetime.now()).replace(" ", "_") + '.log')
# Pipe output to stdout
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)


#------------------------------------------------------------------------------#
#   Funciton definitions
#------------------------------------------------------------------------------#

def getFileList(direc_path: str, ext: str = 'png') -> List[str]:
    # Returns a list of the absolute paths of all files in direc_path with extension ext
    files = []
    for file in os.listdir(direc_path):
        if file.endswith(ext):
            files.append(os.path.join(direc_path, file))

    return files

def before(string: str, stop_string:str = '_') -> str:
    # Returns the string before stop_string (which on default is everything before the firs underscore '_')
    pos = string.find(stop_string)
    if pos == -1:
        return ""
    else:
        return string[0:pos]

def uniqueCounts(a: np.ndarray) -> dict:
    # Returns a dictionary of the counts of each unique value from numpy array a
    #>>> a = numpy.array([0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 4])
    #>>> unique, counts = numpy.unique(a, return_counts=True)    
    #>>> dict(zip(unique, counts))
    #{0: 7, 1: 4, 2: 1, 3: 2, 4: 1}
    unique, counts = np.unique(a, return_counts=True)
    return dict(zip(unique, counts))

def readImage(path: str) -> Image:
    try:
        # load image
        with open(path, 'rb') as f:
            with Image.open(f) as image:
                return np.array(image)
    except IOError:
        raise RuntimeError(logger.error('Cannot retrieve image. Please check path: %s' % path))


def makeFileListDF(files: List[str]) -> pd.DataFrame:
    """
    Returns a pandas DataFrame where each row describes each file in files. 
    Each integer column contains the number of pixels in the image with that label
    """
    # convert paths to relative:
    basenames = list(map(os.path.basename, files))
    columns = ['filename', 'dataset']
    df = pd.DataFrame(columns = columns)
    df['filename'] = basenames

    # get dataset:
    df['dataset'] = df.filename.apply(before)

    # After getting dataset, set file basename as index:
    df.set_index('filename', inplace=True)

    # Loop through images and count label occurences:
    num_files = len(basenames)
    i = 0
    percentage = .01
    for file, basename in zip(files, basenames):
        #Read in image:
        image = readImage(file)    

        d = uniqueCounts(image)

        # Get unique IDs and number of pixels of each ID:
        for k, v in d.items():
            if k not in df.columns:
                # add column named by label ID
                df[k] = np.nan
                df.at[basename, k] = v
            else:
                df.at[basename, k] = v

        i += 1
        if i/num_files >= percentage:
            print(percentage)
            percentage += .01

    
    return df


#------------------------------------------------------------------------------#
#   Main
#------------------------------------------------------------------------------#

path_to_labels = '/Users/seanmhendryx/githublocal/rob_devkit/segmentation/datasets_kitti2015/training/semantic'

# Get list of labeled image files:
lab_files = getFileList(path_to_labels)

df = makeFileListDF(lab_files)

#len(df.columns)
#Out[14]: 76

# Number of images from each dataset:
uniqueCounts(df.dataset)
# {'Cityscapes': 3475, 'Kitti2015': 200, 'ScanNet': 24353, 'WildDash': 70}


#uniqueCounts(df.drop(['filename', 'dataset'], axis=1))
# Are the IDs unique across datasets? 
# Check to make sure label IDs with pixel counts (values, not nan) are exclusive to 
# each dataset

