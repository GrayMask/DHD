# Deep Hyperspectral-Depth Reconstruction Using Single Color-Dot Projection
This is the official repo for the implementation of the **dataset generateion part** of the CVPR2022 paper: Deep Hyperspectral-Depth Reconstruction Using Single Color-Dot Projection.

## Introduction
Since it is difficult to simultaneously acquire accurate depth and spectral reflectance as a large-scale ground-truth dataset in real-world situations,
we developed a spectral renderer to generate a synthetic dataset with rendered RGB color-dot images, ground-truth disparity maps,
and ground-truth spectral reflectance images by extending the algorithm of a [structured-light renderer](https://github.com/autonomousvision/connecting_the_dots).

## Usage
### Dependencies
The python packages can be installed with `anaconda`:
```
conda install --file requirements.txt
```

### Building
First make sure the correct `CUDA_LIBRARY_PATH` is set in `config.json`.
Afterwards, the renderer can be build by running `make` within the `renderer` directory.

### Running
First, download [ShapeNet V2](https://www.shapenet.org/) and change `SHAPENET_ROOT` in `config.json`.
Then the data can be generated and saved to `DATA_ROOT` in `config.json` by running
```
python create_syn_data.py
```

## Acknowledgement
The code structure and some code snippets (rasterisation, shading, etc.) are borrowed from [Connecting the Dots](https://github.com/autonomousvision/connecting_the_dots).
Thanks for this great project.
