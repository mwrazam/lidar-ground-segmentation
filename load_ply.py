# Utility functions to load and prepare PLY files for the ground segmentation algorithm
# Author: Muhammad Azam, July 2022
# Note: Most of the code in this file is credited to the authors of the 
#   Sensaturban dataset: https://github.com/QingyongHu/SensatUrban

import os
import numpy as np

def parse_header(plyfile:os.path, ext:str) -> tuple[int, list]:
    # Define PLY types
    ply_dtypes = dict([
        (b'int8', 'i1'),
        (b'char', 'i1'),
        (b'uint8', 'u1'),
        (b'uchar', 'u1'),
        (b'int16', 'i2'),
        (b'short', 'i2'),
        (b'uint16', 'u2'),
        (b'ushort', 'u2'),
        (b'int32', 'i4'),
        (b'int', 'i4'),
        (b'uint32', 'u4'),
        (b'uint', 'u4'),
        (b'float32', 'f4'),
        (b'float', 'f4'),
        (b'float64', 'f8'),
        (b'double', 'f8')
    ])

    # Variables
    line = []
    properties = []
    num_points = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        if b'element' in line:
            line = line.split()
            num_points = int(line[2])

        elif b'property' in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

    return num_points, properties

def get_ply_data(filepath:os.path, subsample:int=None, 
        debug=False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Numpy reader format
    valid_formats = {'ascii': '', 'binary_big_endian': '>',
                    'binary_little_endian': '<'}

    with open(filepath, 'rb') as plyfile:
        if debug: print("Loading PLY file...")
        # Checks to make sure we have a properly formatted ply file
        if b'ply' not in plyfile.readline():
            raise ValueError('The file does not start with the world ply')

        # Get binary_little/big or ascii
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError('The file is not binary')

        # Get extension for building the numpy dtypes
        ext = valid_formats[fmt]
            
        # Parse header
        if debug: print("Parsing header...")
        num_points, properties = parse_header(plyfile, ext)
            
        # Get data
        data = np.fromfile(plyfile, dtype=properties, count=num_points)

        if subsample is not None:
            if debug: print("Subsampling...")
            idx = np.random.choice(data.shape[0], data.shape[0] // subsample)
            data = data[idx]

        # pipe data
        data = np.array([list(data[d]) for d in range(data.shape[0])])
    if debug: print("Done.")

    labels = None
    if data.shape[1] == 7:
        labels = np.int_(data[:,-1])

    return data[:,:3], data[:,3:6]/255.0, labels