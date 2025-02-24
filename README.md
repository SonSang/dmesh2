# DMesh++: An Efficient Differentiable Mesh for Complex Shapes

DMesh++ is a probabilistic approach for 2D and 3D mesh, which computes existence probability for faces efficiently and thus handles complex shapes effectively. Please refer to our [paper](https://arxiv.org/abs/2412.16776) and [project website](https://sonsang.github.io/dmesh2-project/) for more details.

![Teaser image](<static/teaser.png>)

## Installation

Our code was developed and tested only for Ubuntu environment. Please clone this repository recursively to include all submodules.

```bash
git clone https://github.com/SonSang/dmesh2.git --recursive
```

### Dependencies

We use Python version 3.10, and recommend using [Anaconda](https://anaconda.org/) to manage the environment. 
After creating a new environment, please run following command to install the required python packages.

```bash
pip install -r requirements.txt
```

We also need additional external libraries to run DMesh++. Please install them by following the instructions below.

#### PyTorch (2.4.0)

Please install PyTorch that aligns with your NVIDIA GPU. Currently, our code requires NVIDIA GPU to run, because some major algorithms are written in CUDA. You can find instructions [here](https://pytorch.org/get-started/locally/). 

#### pytorch_scatter (2.1.2)

Please install pytorch_scatter following instruction [here](https://github.com/rusty1s/pytorch_scatter). In short, you can install pytorch_scatter for PyTorch version 2.4.0 with following command,

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+${CUDA}.html
```

where `${CUDA}` should be replaced by either cpu, cu118, cu121, or cu124 depending on the PyTorch installation.

#### pytorch3d (0.7.8)

Please follow detailed instructions [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md). In short, you can install (the latest) pytorch3d by running the following command.

```bash
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

#### CGAL

We use [CGAL](https://github.com/CGAL/cgal) to run the Delaunay Triangulation (DT) algorithm, which is highly related to our algorithm. If you cloned this repository recursively, you should already have the latest CGAL source code in the `external/cgal` directory. Please follow the instructions below to build and install CGAL.

```bash
cd external/cgal
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

You might need to install some additional dependencies to build CGAL. Please refer to the [official documentation](https://doc.cgal.org/latest/Manual/thirdparty.html) and install essential third party libraries, such as Boost, GMP, and MPFR, to build CGAL and CGAL-dependent code successfully. If you are using Ubuntu, you can install GMP and MPFR with following commands.

```bash
sudo apt-get install libgmp3-dev
sudo apt-get install libmpfr-dev
```

#### Nvdiffrast

We use [nvdiffrast](https://github.com/NVlabs/nvdiffrast) for rendering ground truth images for 3D multi-view reconstruction problems.. Therefore, this installation is optional if you do not use our code for generating ground truth images. Please follow the instructions below to build and install nvdiffrast.

```bash
sudo apt-get install libglvnd0 libgl1 libglx0 libegl1 libgles2 libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev
cd external/nvdiffrast
pip install -e .
```

Please see the [official documentation](https://nvlabs.github.io/nvdiffrast/) if you encounter any issues during the installation.

#### DMesh2Renderer

We implemented our own renderers, named [DMesh2Renderer](https://github.com/SonSang/dmesh2_renderer) for 3D multi-view reconstruction. Before installing it, please install [GLM](https://github.com/g-truc/glm) library. If you use Ubuntu, you can install it by running the following command.

```bash
sudo apt-get install libglm-dev
```

Or, you can install it within the conda environment as follows.

```bash
conda install conda-forge::glm
```

Then, please follow the instructions below to build and install DMeshRenderer.

```bash
cd external/dmesh2_renderer
pip install -e .
```

### Build CGAL-dependent code

Run following command to build CGAL-dependent code.

```bash
cd cgal_wrapper
cmake -DCMAKE_BUILD_TYPE=Release .
make
```

You would be able to find `libcgal_diffdt.a` file in `cgal_wrapper/` directory.

## Boost Install

If there is an error coming from Boost version while building CGAL-dependent code, and you are using Conda, please install the latest Boost with Conda.

```bash
conda install conda-forge::boost
```

### Build DMesh++

Finally, run following command to build DMesh++.

```bash
pip install -e .
```

## Dataset

For 2D point cloud reconstruction tasks, we used font datasets from [Google Font](https://fonts.google.com/). In our repository, we provide an example of [Roboto Font](https://fonts.google.com/specimen/Roboto) under `data/2d/font`. We also used several SVG files downloaded from [Adobe Stock](https://stock.adobe.com/vectors) for our experiments in the paper, but in this repository, we provide two example SVG files from [SVG Repo](https://www.svgrepo.com/). You can find those files under `data/2d/svg`.

For 3D multi-view reconstruction tasks, we used 3D mesh models from [Thingi10K](https://ten-thousand-models.appspot.com/) dataset, and several textured models from [Objaverse](https://objaverse.allenai.org/objaverse-1.0/) dataset. We do not provide the 3D models in this repo, but include example input images for 2 different models under `input/3d/mvrecon`.

## Usage

Here we provide how to use DMesh++ for several downstream tasks discussed in the paper. We first generate input data for the particular downstream task (e.g. point cloud, multi-view images) using our input data generation code, and then run optimization code for the given input data. All of the examples use config files in `exp/config` folder. You can modify the config files to change the input/output paths, hyperparameters, etc. By default, all the results are stored in `exp/result`.

### Example 1: 2D Mesh Reconstruction from Point Clouds

#### Input point cloud generation

To reconstruct 2D mesh from given point cloud, we need an input point cloud in 2D.
First, we can run the following command to generate the input point cloud from a font data.
```bash
python input/generate_pcrecon_2d_input_font.py --font-path data/2d/font/Roboto/Roboto-Regular.ttf --font-char A
```

We can also generate the input point cloud from a SVG file (.svg) with following command.
```bash
python input/generate_pcrecon_2d_input_svg.py --input-path data/2d/svg/botanical_1.svg
```

By default, the generated point cloud inputs are stored in `input/2d/pcrecon` in `.npy` format.
See each of the generation code for more details about arguments.

#### Reconstructing 2D mesh from point clouds

Then, we can reconstruct 2D mesh from this input point cloud.
We can use two configuration files to do that.
The first configuration file `exp/config/d2/pcrecon_svg.yaml` produces high precision 2D mesh without using Reinforce-Ball algorithm.
```bash
python pcrecon_2d.py --config exp/config/d2/pcrecon_svg.yaml --input-path input/2d/pcrecon/botanical_1.npy
```

The second configuration file `exp/config/d2/pcrecon_svg.yaml` produces efficient 2D mesh using Reinforce-Ball algorithm.
Note that this configuration is optimized for reconstructing optimal 2D mesh for font dataset.
```bash
python pcrecon_2d.py --config exp/config/d2/pcrecon_font.yaml --input-path input/2d/pcrecon/Roboto-Regular_A.npy
```

Also see the `prcrecon_2d.py` file for more details about arguments.

### Example 2: 3D Mesh Reconstruction from Multi-View Images

#### Input multi-view image generation

To generate the input multi-view images for 3D mesh reconstruction algorithm, please run the following command.
```bash
python input/generate_mvrecon_3d_input.py --input-path data/3d/98576.stl
python input/generate_mvrecon_3d_input.py --input-path data/3d/toad.glb
```

When the input 3D model is in `.glb` format, our code renders the textured mesh and produces colored images.
Otherwise, our code renders non-textured mesh and produces grey colored images.
Our code also renders depth maps and saves camera parameters to use in the reconstruction algorithm.

By default, the generated multi-view images are stored in `input/3d/mvrecon`.
See the generation code for more details about arguments.

#### Reconstructing 3D mesh from multi-view images

Now we can reconstruct 3D mesh from these multi-view images.
As we did for 2D point cloud reconstruction, we can use two configuration files to do that.
The first configuration file `exp/config/d3/mvrecon_thingi10k.yaml` assumes the mesh color is fixed to white, and thus produces non-textured mesh.
Also, it runs for only two epochs of optimization as reported in the paper, and removes non-manifoldness at the last step.
```bash
python mvrecon_3d.py --config exp/config/d3/mvrecon_thingi10k.yaml --input-path input/3d/mvrecon/98576/
```

The second configuration file `exp/config/d3/mvrecon.yaml` produces textured mesh by optimizing vertex-wise colors together.
It runs for four epochs of optimization, and thus produces high-resolution mesh. However, it does not remove non-manifoldness, as it requires much more computational cost for high-resolution mesh.
```bash
python mvrecon_3d.py --config exp/config/d3/mvrecon.yaml --input-path input/3d/mvrecon/toad/
```

Please see the `mvrecon_3d.py` file for more details about arguments.

## Citation

```bibtex
@article{son2024dmesh++,
  title={DMesh++: An Efficient Differentiable Mesh for Complex Shapes},
  author={Son, Sanghyun and Gadelha, Matheus and Zhou, Yang and Fisher, Matthew and Xu, Zexiang and Qiao, Yi-Ling and Lin, Ming C and Zhou, Yi},
  journal={arXiv preprint arXiv:2412.16776},
  year={2024}
}
```

## Acknowledgement

Our code is mainly based on that of [DMesh](https://github.com/SonSang/dmesh). Therefore, we use [CGAL](https://github.com/CGAL/cgal) in our codebase. Also, for implementing 3D multi-view reconstruction code, we brought implementations of [nvdiffrast](https://github.com/NVlabs/nvdiffrast), [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [Continuous Remeshing For Inverse Rendering](https://github.com/Profactor/continuous-remeshing). We appreciate these great works.