# Content
This repository's purpose is to present some productions for my exam...

It is based on the [PyTorch-DP framework](https://github.com/facebookresearch/pytorch-dp) (*frozen in the 19th of June, 2020 version. Under [Apache License 2.0](https://github.com/facebookresearch/pytorch-dp/blob/master/LICENSE)*)

1. A notebook that presents the budget computation of Differential Privacy with PyTorch-DP and links to articles for justifications. You can either [read it or download it](https://github.com/jmg-74/exam/blob/master/torchdp/scripts/DP_Computation_in_SGM.ipynb) to use on your own server or run it online (quite slow to launch) thanks to [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jmg-74/exam/master?filepath=torchdp%2Fscripts%2FDP_Computation_in_SGM.ipynb "Tip: right clic / open in new tab...")
1. An attempt to convert "102-flowers" DL model to a DP version:
  * First, once and for all : download data.
    * `mkdir data/ && cd data && wget https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz`
    * `tar xvf flower_data.tar.gz && cd ..`

  * **Train model**: `python3 train.py` or `./train.py` (try `-h` to list optional arguments, like `--cpu` if no GPU available,
  `--batch-size`, `--learning-rate`, `epochs`, `--disable-dp`...)

  * Record stats about **accuracy**: `./stats.py`. Parameters to experiment are hard coded, see docstring. Results are stored
 in experiment_stats/ directory.

  * Record stats about **GPU memory**:  `flowers_mem_monitor.py` may be launched directly to add one line in .cvs file.
  `flowers_mem_stats.sh` launches `flowers_mem_monitor.py` for different parameters (modify code to chose their values),
  results are stored in mem_flowers/

