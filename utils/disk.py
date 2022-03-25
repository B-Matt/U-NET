"""
    Improved clever Disk cache method from Franko Hržić with implementing zlib compression instead of gzip.
    GitHub repo: https://github.com/fhrzic/U-Net/blob/main/scripts/utils/disk.py
"""

import zlib
import diskcache
from diskcache.core import io
from io import BytesIO

# Constants
CACHE_DIRECTORY = 'cache/'

# Logging
from utils.logging import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Caching
class DeflatedDisk(diskcache.Disk):
    """
    Overrides diskcache Disk class with implementation of zlib library for compression.
    """
    def store(self, value, read, key=None):
        """
        Override from base class diskcache.Disk.
        Chunking is due to needing to work on pythons < 2.7.13:
        
        :param value: value to convert
        :param bool read: True when value is file-like object
        :return: (size, mode, filename, value) tuple for Cache table
        """
        
        if read is True:
            value = value.read()
            read = False

        print('cache1', value[0])
        print('cache2', value[0])
        print('\n')
        bytes_value = BytesIO(value)
        value = zlib.compress(bytes_value.getvalue(), zlib.Z_BEST_COMPRESSION)
        return super(DeflatedDisk, self).store(value, read)

    def fetch(self, mode, filename, value, read):
        """
        Override from base class diskcache.Disk.

        :param int mode: value mode raw, binary, text, or pickle
        :param str filename: filename of corresponding value
        :param value: database value
        :param bool read: when True, return an open file handle
        :return: corresponding Python value
        """
        value = super(DeflatedDisk, self).fetch(mode, filename, value, read)
        if not read:
            bytes_value = BytesIO(value)
            value = zlib.decompress(bytes_value.getvalue())

        print('aa', value)
        return value

def getCache(scope_str):
    return diskcache.FanoutCache(
        f'{CACHE_DIRECTORY}/{scope_str}',
        disk=DeflatedDisk,
        shards=100,
        timeout=1,
        size_limit=3e11
    )