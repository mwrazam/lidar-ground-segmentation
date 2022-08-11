# City-Scale Ground Segmentation of Aerial LiDAR by Image-Analogous Gradients

LiDAR Point Cloud segmentation is a key input to downstream tasks such as object recognition and classification, obstacle avoidance, and even 3D reconstruction. A key challenge in the segmentation of large city-scale datasets is uneven distribution of points to specific classes and significant class imbalances. As highly detailed point cloud datasets of urban environments become available, neural networks have shown significant performance in recognizing large well-defined objects. However, data is fed into these networks in chunks and the scheme by which data is presented for training and evaluation can have a significant impact on performance. In this work, we establish a method analogous to gradients in image processing to segment the ground in point clouds, achieving an accuracy of 91.4\% on the Sensaturban dataset. By isolating the ground, we reduce the quantity of classes that need to be segmented from structures in urban LiDAR and improve data partitioning schemes when combined with random/grid down-sampling techniques for neural network inputs.

Below is a visualization of the output of our method on cropped section of a tile from the Vancouver 2018 LiDAR data (left) and a full tile from the Sensaturban dataset (right).

![Ground segmentation output](/images/icmla-fig-output.png)

# Setup

Python 3.x and a number of packages (listed in requirements.txt) are needed to execute this program. It is highly recommended to use a virtual environment. From the root directory of the project, execute the following:

```
python3 -m virutalenv env
```

On OSX you can activate the environment and install dependencies by:

```
source env/bin/activate
pip install -r requirements.txt
```

# Getting the data
Two datasets are used in this project. The [Vancouver 2018 LiDAR dataset](https://opendata.vancouver.ca/explore/dataset/lidar-2018/information/) and the [Sensaturban point cloud dataset](https://github.com/QingyongHu/SensatUrban). They can be obtained by visiting their respective websites.

We generated the visualizations above on a crop of the 4910E_54590N.las tile from Vancouver, and the full cambridge_block_8.ply tile from Sensaturban.

Once you have selected which tile you want to run the program on, download and extract the .las or the .ply file to the /data folder.

# How to run
From the root directory, simply execute by:

```
python main.py --file=LAS/PLY_FILENAME
```

The program will assume by default that the file is contained in the `/data` folder. The `--file` argument is required, but there are also some other options available (with defaults shown):

`--datadir=data` sets location of the data, intermediate outputs will be saved to this same folder

`--debug=False` provides feedback as to which step the program is currently performing

`--intermediate_output=True` produces intermediate `.npy` files along the way, reducing computation time with reruns
