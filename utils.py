# Utility functions for working with point cloud data
# Author: Muhammad Azam, July 2022

import os, sys
import numpy as np
import open3d as o3d

from load_las import get_las_data
from load_ply import get_ply_data

# Load data from either file or intermediate output
def load_data(filepath:os.path, intermediate_output:bool=False, 
        save_to:os.path=None, debug:bool=False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # deduce if we are working with ply or las files
    datadir, fname = os.path.split(filepath)
    name, ft = fname.split(".")

    imd_data = os.path.join(datadir, f"{name}_imd_data.npy")
    imd_colors = os.path.join(datadir, f"{name}_imd_colors.npy")
    imd_labels = os.path.join(datadir, f"{name}_imd_labels.npy")

    data, colors, labels = None, None, None
    if intermediate_output and os.path.exists(imd_data):
        data = np.load(imd_data)
        if debug: print(f"Loaded saved point data from {imd_data}")
        if os.path.exists(imd_colors):
            colors = np.load(imd_colors)
            if debug: print(f"Loaded color from {imd_colors}")
        if os.path.exists(imd_labels):
            labels = np.load(imd_labels)
            if debug: print(f"Loaded labels from {imd_labels}")

    else:
        if ft == "ply":
            # Load point cloud format file
            data, colors, labels = get_ply_data(filepath, subsample=50, debug=debug)
            if debug: print(f"Loaded PLY file with {data.shape}")
            if intermediate_output:
                np.save(imd_data, data)
                if debug: (f"Saved point data to {imd_data}")
                if colors is not None:
                    np.save(imd_colors, colors)
                    if debug: (f"Saved colors to {imd_colors}")
                if labels is not None:
                    np.save(imd_labels, labels)
                    if debug: (f"Saved labels to {imd_labels}")
        elif ft == "las":
            # Load LAS data
            data = get_las_data(filepath, trim=[0.15, 0.40, 0.4, 0.65], z_quant=[0.01, 0.99], debug=debug)
            #data = get_las_data(filepath, debug=debug)
            if debug: print(f"Loaded LAS file with {data.shape}")
            if intermediate_output:
                np.save(imd_data, data)
                if debug: (f"Saved point data to {imd_data}")
        else:
            sys.exit("Error: Unknown file type provided. Only .las and .ply files can be used.")

    return data, colors, labels

# Generate a open3d geometry point cloud object
def generate_point_cloud(data:np.ndarray, colors:np.ndarray=None, 
        estimate_normals:bool=False, visualize:bool=False) -> o3d.geometry:

    # Use o3d to produce point cloud visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    if estimate_normals:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if visualize:
        o3d.visualization.draw_geometries([pcd])
    return pcd

# Convert gradient magnitudes to color
def gradients_to_color(mag:np.ndarray, lower_bound:np.double, upper_bound:np.double) -> np.ndarray:
    c = np.full((mag.shape[0], 3), [0.0, 1.0, 0.0])
    mask = np.where((mag < upper_bound) & (mag > lower_bound))[0]
    c[mask] = np.array([1.0, 0.0, 0.0])
    return c

# Convert ground labels to color
def labels_to_color(cls:np.ndarray) -> np.ndarray:
    c = np.full((cls.shape[0], 3), [0.1, 0.1, 0.1])
    mask = np.where((cls == 1))[0]
    c[mask] = np.array([1.0, 0.0, 0.0])
    return c