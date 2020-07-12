# Content
This repository's purpose is to present some productions for my exam...

It is based on the [PyTorch-DP framework](https://github.com/facebookresearch/pytorch-dp) (*frozen in the 19th of June, 2020 version. Under [Apache License 2.0](https://github.com/facebookresearch/pytorch-dp/blob/master/LICENSE)*)

1. A **notebook** that presents the budget computation of Differential Privacy with PyTorch-DP and links to properties in several articles for justifications. You can either [read it or download it](https://github.com/jmg-74/exam/blob/master/torchdp/docs/DP_Computation_in_SGM.ipynb) to use on your own server or run it online (quite slow to launch) thanks to [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jmg-74/exam/master?filepath=torchdp%2Fdocs%2FDP_Computation_in_SGM.ipynb "Tip: right clic / open in new tab...")

2. An attempt to **convert "102-flowers" DL model to a DP version**, see `flowers/` directory.
  * First, once and for all : download data.
    * `mkdir data/ && cd data && wget https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz`
    * `tar xvf flower_data.tar.gz && cd ..`

  * **Train model**: `python3 train.py` or `./train.py` (try `-h` to list optional arguments, like `--cpu` if no GPU available,
  `--batch-size`, `--learning-rate`, `epochs`, `--disable-dp`...)

  * Record stats about **accuracy**: `./stats.py`. Parameters to experiment are hard coded, see the docstring. Results are stored
 in experiment_stats/ directory.

  * Record stats about **GPU memory**:  
    * `flowers_mem_monitor.py` may be run directly to add one line in .cvs file.
    * `flowers_mem_stats.sh` launches `flowers_mem_monitor.py` for different parameters (modify code to chose their values),
  results are stored in mem_flowers/

3. An other attempt on **cifar10** dataset.
  * First without DP: `cifar10.py`
    * (net = home-made simple model imported from `my_net.py`)
    * net = fully pre-trained VGG16
    * net = not pre-trained VGG16
    * net =  pre-trained on "features" only, not on "classifier" layers.
  * Then converted to DP version: `dp_cifar10.py`

4. Other stuff
  * Similar functions (than one of 'flowers') about MNIST dataset, 
    see `mnist_train.py`, `mnist_mem_stats.sh`, `mnist_mem_monitor.py` in `mnist/` directory.

  * Calculate directly a privacy budget without launching any training with `torchdp/scripts/compute_dp_sgd_privcacy.py` (*This adaptation to PyTorch-DP I wrote from an equivalent script for TensorFlow is now included in PyTorch-DP repository* :smirk:).

