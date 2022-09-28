# DeepProfilerExperiments
This is a supplementary repository for [DeepProfiler software](https://github.com/cytomining/DeepProfiler/) and
[the related analysis pre-print](https://cytomining.github.io/DeepProfiler-handbook/).

Please see our [DeepProfiler handbook](https://cytomining.github.io/DeepProfiler-handbook/) for documentation.

In the folders `ta-orf`, `bbbc022` and `cdrp` of this repository you can find Jupyter notebooks for downstream 
analysis and configuration files (folder `config`) to reproduce training experiments and run feature extraction afterwards, plus 
configuration files to run feature extraction with [_**Cell Painting CNN**_](https://doi.org/10.5281/zenodo.7114558) or EfficientNet pre-trained on ImageNet dataset.
Ground truth annotations are in `data` folders used in the publication. 

`profiling` folder contains libraries and utils functions for downstream analysis notebooks.

`bbbc021` contains Jupyter notebooks for downstream analysis of the BBBC021 dataset. This is legacy code. 



