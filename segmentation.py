import os
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix

# Generate per point analogous gradients
def compute_point_based_gradient(filepath:os.path, data:np.ndarray, k:int=50, 
        debug:bool=False, intermediate_output=False) -> tuple[np.ndarray, np.ndarray]:
    # Compute the gradient magnitude and direction based on the first and third principal components
    #  of a k nearest neighbors local space for each point
    g_mag = np.zeros((data.shape[0], 1))
    g_dir = np.zeros_like(data)

    datadir, fname = os.path.split(filepath)
    name, ft = fname.split(".")

    imd_gmag = os.path.join(datadir, f"{name}_imd_gradients_mag.npy")
    imd_gdir = os.path.join(datadir, f"{name}_imd_gradients_dir.npy")

    # Use KDTree data structure for fast nearest neighbors search
    d = KDTree(data)

    if intermediate_output and os.path.exists(imd_gmag) and os.path.exists(imd_gdir):
        g_mag = np.load(imd_gmag)
        g_dir = np.load(imd_gdir)
        if debug: print(f"Loaded gradient magnitudes from {imd_gmag} and directions from {imd_gdir}")
    else:
        if debug: print(f"Computing point based gradients...")
        for idx, p in enumerate(data):
            # 1. Find nearest neighbors
            _, nn_i = d.query(p, k=k)
            nn = data[nn_i]
            
            # 2. Remove centroid from each point
            m = np.sum(nn, axis=0) / nn.shape[0]
            nn = nn - m

            # 3. Compute Scatter matrix
            S = np.matmul(nn.T, nn)

            # 4. Compute orthogonality between normal vector vertical unit vector
            eig_vals, eig_vectors = np.linalg.eig(S)
            ev = eig_vectors[:, np.argsort(eig_vals)]
            g_mag[idx] = ev[:, 0].dot([0,0,1])
            g_dir[idx, :] = ev[:, 2]

            if idx % int(data.shape[0]/10) == 0:
                print(f"{idx}/{data.shape[0]}")
        if debug: print("Done.")

        if intermediate_output:
            np.save(imd_gmag, g_mag)
            np.save(imd_gdir, g_dir)
            if debug: (f"Saved gradient magnitudes to {imd_gmag} and directions to {imd_gdir}")

    return g_mag, g_dir

# Perform ground classification
def classify_ground(filepath:os.path, data:np.ndarray, grad:np.ndarray, 
        lower_bound:np.double =-0.9, upper_bound:np.double=0.9, eps=0.5,
        debug:bool=False, intermediate_output=False) -> np.ndarray:

    # classification storage
    m = np.ones((data.shape[0], 2), dtype=int)
    m[:,1] = 0

    datadir, fname = os.path.split(filepath)
    name, ft = fname.split(".")

    imd_cls = os.path.join(datadir, f"{name}_imd_cls.npy")

    if intermediate_output and os.path.exists(imd_cls):
        m = np.load(imd_cls)
        if debug: print(f"Loaded class labels from {imd_cls}")
    else:
        print("Classifying ground points...")

        # Pick out which points have a gradient within range
        p_i = np.where((grad < upper_bound)&(grad > lower_bound))[0]
        m[p_i,0] = 0
        X = data[m[:,0]==1]

        # Cluster points and pick out the largest as the ground
        if ft == 'las':
            eps = 0.01
            # TODO: Finish off the part that automates determining epsilon
            #  based on x-y point density
        Y = DBSCAN(eps=eps).fit_predict(X)
        labels, counts = np.unique(Y[Y>=0], return_counts=True)
        print(labels, counts)
        largest = labels[np.argsort(-counts)[:1]]
        Y2 = np.zeros((Y.shape[0]), dtype=int)
        Y2[Y==largest] = 1

        # Update labels matrix
        m[m[:,0]==1, 1] = Y2
        print("Done.")
        if intermediate_output:
            np.save(imd_cls, m)

    return m[:,1].reshape((m.shape[0], 1))

# Set all non-ground labels to 0, ground labels to 1
def binarize_labels(labels:np.ndarray) -> np.ndarray:
    truth = np.zeros((labels.shape[0]), dtype=int)
    
    # These are the points we are taking as 'ground' in the sensaturban dataset
    truth[labels==0] = 1  # 'Ground'
    truth[labels==5] = 1 # 'Parking'
    truth[labels==6] = 1 # 'Rail'
    truth[labels==7] = 1 # 'traffic Roads'
    truth[labels==10] = 1 # 'Footpath'
    truth[labels==12] = 1 # 'Water'

    # Labels not included as ground are:
    # 1: 'High Vegetation'
    # 2: 'Buildings'
    # 3: 'Walls'
    # 4: 'Bridge'
    # 8: 'Street Furniture'
    # 9: 'Cars'
    # 11: 'Bikes'

    return truth

# Compare predicated labels and compute metrics
def calculate_scores(predicted:np.ndarray, truth:np.ndarray) -> dict:
    cm = confusion_matrix(truth, predicted)
    tp = cm[0,0]
    tn = cm[1,1]
    fp = cm[0,1]
    fn = cm[1,0]
    scores = {
        "accuracy": (tp + tn)/(tp + tn + fp + fn), 
        "misclassed": (fp + fn)/(tp + tn + fp + fn), 
        "precision": tp/(tp + fp), 
        "recall": tp/(tp + fn), 
        "specificity": tn/(tn + fp), 
        "f1": 2*tp/(2*tp + fp + fn), 
        "iou": tp/(tp + fp + fn)}

    return scores