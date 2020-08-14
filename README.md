# RIO10 Documentation

### Dataset Overview

In this work, we adapt [3RScan](https://waldjohannau.github.io/RIO) - a recently introduced indoor RGB-D dataset of changing indoor environments - to create [RIO10](https://waldjohannau.github.io/RIO10), a new long-term camera re-localization benchmark focused on indoor scenes. RIO10 consists of 74 sequences. We provide splits into training, validation (one sequence per scene) and testing sets, leaving us with 10 train, 10 validation and 54 test sequences overall, see corresponding [metadata.json](data/metadata.json).

### Download

If you would like to download the [RIO10](https://waldjohannau.github.io/RIO10) data, please fill out [this form](https://docs.google.com/forms/d/e/1FAIpQLScA-zciAgxMz7r8xirsPQodeQnPk2XA68yBpSxP10167B-M_g/viewform?usp=sf_link). 

### Dataformat

Sequences in RIO10 follow the naming convention `seq<scene_id>_<scan_id>`. The scenes (`seq<scene_id>.zip`) and models (`models<scene_id>.zip`) are stored per scene and consist of multiple sequences e.g.:
    
```
seq02.zip
├── seq02_01
│   ├── frame-000000.color.jpg
│   ├── frame-000000.pose.txt
│   ├── frame-000000.rendered.depth.png
│   ├── frame-000001.color.jpg
│   ├── frame-000001.pose.txt
│   ├── frame-000001.rendered.depth.png
│   └── ...
├── seq02_01
│   ├── frame-000000.color.jpg
│   ├── frame-000000.pose.txt
│   ├── frame-000000.rendered.depth.png
│   └── ...
├── seq02_03
│   ├── frame-000000.color.jpg
│   ├── frame-000000.rendered.depth.png
│   ├── frame-000001.color.jpg
│   ├── frame-000001.rendered.depth.png
│   └── ...
├── seq02_04
└── ...
```

```
models02.zip
├── seq02_01
│   ├── labels.ply
│   ├── mesh.obj
│   ├── mesh.refined_0.png
│   └── mesh.refined.mtl
├── seq02_02
├── seq02_03
├── seq02_04
└── ...
```

Sequence `seq<scene_id>_01` is always the training sequence (with `*.color.jpg`, `*.pose.txt` and `*.rendered.depth.png`), seqXX_02 is the validation sequence (with `*.color.jpg`, `*.pose.txt` and `*.rendered.depth.png`). Please note that we do not provide the ground truth for our hidden test set, all seqXX_02+ are hidden sequences (only `*.color.jpg` and `*.rendered.depth.png`).

Poses `*.pose.txt` of the test (rescan) sequences are aligned with the train (reference) and validation scans. These transform from the camera coordinate system to the world coordinate system. Please use our evaluation service to compute your final performance score on the hidden test set.

The corresponding camera calibration parameters (`fx, fy, cx, cy`) can be found here: [intrinsics.txt](data/intrinsics.txt).

Scene stats are available here: [stats.txt](data/stats.txt) and are structured as follows:

```
frame	NSSD	NCCOEFF	SemanticChange	GeometricChange	VoL	Context	PoseNovelity
seq09_02/frame-000000	0.571	0.531	0.312	296.868	72.369	1.58	316.253
seq09_02/frame-000001	0.571	0.531	0.312	296.882	72.847	1.581	316.264
seq09_02/frame-000002	0.571	0.531	0.312	296.995	71.599	1.583	316.413
seq09_02/frame-000003	0.571	0.531	0.312	297.063	73.363	1.582	316.497
seq09_02/frame-000004	0.571	0.531	0.313	297.125	72.429	1.581	316.543
...
```

### Setup

To run our python code, we recommended to use a virtual environment. Our [`./setup.sh`](setup.sh) script installs dependencies and creates the necessary data sturcture by downloads files (depth images for validation and example predictions).

```
python3 -m venv env
source env/bin/activate

./setup.sh
```

### Evaluation

In this work, we propose a new metric DCRE for evaluating camera re-localization. We also examine in detail how different types of scene change affect the performance of different methods, based on different ways of detecting such changes in a given RGB-D frame (see [stats](data/stats.txt)).

To evaluate your method on the valdiation set, simply save your prediction results in a `.txt` file of the following format:

```
# scene-id/frame-id qx qy qz qw tx ty tz
seq02_03/frame-000000	0.468 -0.841 -0.238 0.123 -2.371 -0.590 0.044
seq02_03/frame-000001	0.461 -0.839 -0.252 0.134 -2.388 -0.586 0.043
seq02_03/frame-000002	0.454 -0.836 -0.267 0.146 -2.408 -0.582 0.042
seq02_03/frame-000003	0.445 -0.833 -0.286 0.157 -2.432 -0.577 0.040
seq02_03/frame-000104	0.438 -0.830 -0.302 0.163 -2.452 -0.573 0.038
```

We provide a simple merge script to combine your predictions if you have multiple prediction files:

```
python [merge_poses.py](src/merge_poses.py) --input_folder=your_predictions --output_file=merged_prediction.txt
```

To compute the pose errors run the code as follows: 

```
DATA=../../../data
METHOD=active_search

cd src/eval/build
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j8

./eval $DATA $DATA/predictions/$METHOD.txt errors/$METHOD.txt
```

This generates an output error file in the following format:

```
# scene-id/frame-id	translation error	rotation error	DCRE
seq01_02/frame-000007	2.0415 174.381 0.605055
seq01_02/frame-000017	1.67518 36.0372 0.339005
seq01_02/frame-000022	0.553614 8.98095 0.104529
seq01_02/frame-000026	0.4914 9.85202 0.123886
seq01_02/frame-000027	0.126553 2.55466 0.0314654
...
```

To run the evaluation for all methods in the `data/predictions` folder simply run `./evaluate.sh`. This will compute and save the errors for each provided pose (in the validation set) and saves it in `data/errors`.

Based on these errors files you can then generate evaluation plots (cumulative DCRE) or see the error with respect to increasing change  `python [plot.py](src/plot.py)`.


The `[plot.py](src/plot.py)` script expects the data folder to be in a predefined structure. The predictions are the methods to plot listed in `config.json`, see `config['change_corr']['methods']` or `config['overview']['methods']`. If nothing is set there; all methods in the `data/predictions` folder are plotted.

```
data
├── metadata.json
├── config.json
├── stats.txt
└── predictions
    ├── active_search.txt
    ├── d2_net.txt
    ├── grove_v1.txt
    └── ...
...
```
