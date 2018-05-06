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
logger.setLevel(logging.DEBUG)
#Set up file handler:
#fh = logging.FileHandler(os.path.basename(__file__) + str(datetime.now()).replace(" ", "_") + '.log')
# Pipe output to stdout
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.DEBUG)
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

def makeFileListDF(files: List[str]) -> pd.DataFrame:
    """
    Returns a pandas DataFrame where each row describes each file in files. 
    Each integer column contains the number of pixels in the image with that label
    """
    # convert paths to relative:
    basenames = map(os.path.basename, files)
    columns = ['dataset']
    df = pd.DataFrame(index = basenames, columns = columns)
    df.rename_axis('filename')

    # For each image:
    for file, basename in zip(files, basenames):
        #Read in image:
        image = readImage(file)    

        d = uniqueCounts(image)

        for k, v in d.items():
            if k not in df.columns:
                # add column named by label ID
                df[k] = 0
                df.at[basename, k] = v
            else:
                df.at[basename, k] = v

        # parse file name and record:
        #    dataset ID (i.e. parent domain of: {cityscapes, scannet, wilddash, kitti})
        #    image name
        #   Get unique IDs and number of pixels of each ID
        image.close()

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
        with Image.open(path) as image:
            return np.array(image)
        #logger.info('image opened')
    except IOError:
        raise RuntimeError(logger.error('Cannot retrieve image. Please check path: %s' % path))

#------------------------------------------------------------------------------#
#   Main
#------------------------------------------------------------------------------#

path_to_labels = '/Users/seanmhendryx/githublocal/rob_devkit/segmentation/datasets_kitti2015/training/semantic'

# Get list of labeled image files:
lab_files = getFileList(path_to_labels)

df = makeFileListDF(lab_files)
