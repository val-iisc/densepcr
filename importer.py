import argparse
import os
import random
import re
import sys
import time

from os import listdir, makedirs
from os.path import join, exists, isdir, dirname, abspath, basename

import cv2
import numpy as np
import tensorflow as tf
import tflearn

from utils.data_utils import *
from utils.encoders_decoders import *
from utils.model_utils import *

from utils.chamfer import tf_nndistance
from utils.emd.tf_auctionmatch import auction_match
from utils.emd.tf_approxmatch import approx_match, match_cost

random.seed(1024)
np.random.seed(1024)
tf.set_random_seed(1024)


pcl_16k_fname = 'pointcloud_16k.npy'
pcl_4k_fname = 'pointcloud_4096.npy'
pcl_1k_fname = 'pointcloud_1024.npy'


# constants
# image
HEIGHT = 128
WIDTH = 128
# densenet
UPSAMPLING_FACTOR = 4 # should be a perfect square
GRID_SIZE = int(math.sqrt(UPSAMPLING_FACTOR))
hierarchies = [1024,4096,16384]
radius = {1024:0.1, 4096:0.05} # round(0.025*math.log(16384//in_res, 2),2)