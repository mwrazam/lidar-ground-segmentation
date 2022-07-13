# Utility functions to load and prepare LAS files for the ground segmentation algorithm
# Author: Muhammad Azam, July 2022

import os, sys
import laspy
import numpy as np
import open3d as o3d

def load_las_file(filepath, debug:bool=False) -> tuple[np.ndarray, laspy.LasHeader]:
    try:
        las = laspy.read(filepath)
        if debug: print("LAS File successfully loaded!")
        return np.vstack((las.x, las.y, las.z)).transpose(), las.header
    except FileExistsError as fee:
        sys.exit(f"No file found at {filepath}")
    except Exception as e:
        sys.exit(f"Could not open file, {e}")

def preprocess_data(data:np.ndarray, scale:np.double=None, offset:np.ndarray=None, 
        recenter:bool=None, rotate:np.ndarray=None, crop:np.ndarray=None, debug:bool=False) -> np.ndarray:
    d = np.copy(data)
    if debug: print("Preprocessing data...")
    # Note: These are not commutative!
    if scale is not None:
        d *= scale
        if offset is not None:
            d -= offset * scale

    # Apply a rotational matrix
    if rotate is not None:
        # Using o3d as its easier than implementing a rotational matrix
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(d)
        R = pcd.get_rotation_matrix_from_xyz((rotate[0], rotate[1], rotate[2]))
        pcd.rotate(R)
        d = np.asarray(pcd.points)

    # The following operations are commutative
    if recenter:
        d = d - np.average(d, axis=0)

    # Crop last
    if crop is not None:
        # Compute range of data
        mins = np.min(d, axis=0)
        r = np.max(d, axis=0) - mins

        # Create masks for each dimension
        for i in range(data.shape[1]):
            m1 = d[:,i] > mins[i] + (r[i] * crop[i])
            m2 = d[:,i] < mins[i] + (r[i] * crop[i+3])
            mask = np.logical_and(m1, m2)
            d = d[mask]

    if debug: print("Done.")
    return d

def get_las_data(filepath:os.path, preprocess:bool=True, trim:list=[0.0, 1.0, 0.0, 1.0], 
        z_quant:list=[0.00, 1.00], rotate:list=[0.0, 0.0, 0.0], debug=False) -> np.ndarray:

    # Load point and header data from file
    data, las_header = load_las_file(filepath, debug=debug)
    if preprocess:

        # Figure out what scale and offset has been applied to the data, and undo it
        scale = np.array([las_header.x_scale, las_header.y_scale, las_header.z_scale])
        offset = np.array([las_header.x_offset, las_header.y_offset, las_header.z_offset])

        # Optional step to exclude outliers based on quantiles, not necessary on all datasets, especially
        #  if noise has already been removed
        z_min = 0.99*(np.quantile(data, z_quant[0], axis=0)[2] - np.min(data[:,2]))/np.ptp(data[:,2])
        z_max = 1.01*(np.quantile(data, z_quant[1], axis=0)[2] - np.min(data[:,2]))/np.ptp(data[:,2])

        # Trim the dataset down so we only work on a smaller section for faster computation/demo
        crop = np.array([trim[0], trim[2], z_min, trim[1], trim[3], z_max]) # percentages

        data = preprocess_data(data, scale=scale, offset=offset, recenter=True, rotate=rotate, crop=crop, debug=debug)

    return data
