import os
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix

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

def classify_ground(filepath:os.path, data:np.ndarray, grad:np.ndarray, 
        lower_bound:np.double =-0.9, upper_bound:np.double=0.9,
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
        Y = DBSCAN(eps=0.5).fit_predict(X)
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

# Compare predicated labels and compute metrics
def calculate_scores(predicted, truth):
    scores = {"accuracy": None, "precision": None, "recall": None, "f1": None, "iou": None}
    cm = confusion_matrix(truth, predicted)
    print(cm)
    # TODO: Calculate acc and other measures

    pass