import os
from collections import namedtuple

NUM_THREADS = min(8, max(1, os.cpu_count() - 1))

data_info_tuple = namedtuple(
    'data_info_tuple',
    'name, image, mask'
)