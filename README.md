<h2 align="center"><b>INSTA - Instant Volumetric Head Avatars (pytorch)</b></h2>

<h4 align="center"><b><a href="https://zielon.github.io/" target="_blank">Wojciech
Zielonka</a>, <a href="https://sites.google.com/site/bolkartt/" target="_blank">Timo
Bolkart</a>, <a href="https://justusthies.github.io/" target="_blank">Justus Thies</a></b></h4>

<h6 align="center"><i>Max Planck Institute for Intelligent Systems, Tübingen, Germany</i></h6>

<h4 align="center">
<a href="https://youtu.be/HOgaeWTih7Q" target="_blank">Video&nbsp</a>
<a href="https://arxiv.org/pdf/2211.12499.pdf" target="_blank">Paper&nbsp</a>
<a href="https://zielon.github.io/insta/" target="_blank">Project Website&nbsp</a>
<a href="https://keeper.mpdl.mpg.de/d/5ea4d2c300e9444a8b0b/" target="_blank"><b>Dataset&nbsp</b></a>
<a href="https://github.com/Zielon/metrical-tracker" target="_blank">Face Tracker&nbsp</a>
<a href="mailto:&#105;&#110;&#115;&#116;&#97;&#64;&#116;&#117;&#101;&#46;&#109;&#112;&#103;&#46;&#100;&#101;">Email</a>
</h4>

This repository is based on [torch-ngp](https://github.com/ashawkey/torch-ngp) and implements most of the C++ version.
**Please note that the speed of training, rendering, and the quality
are not exactly the same like in the case of the [paper version](https://github.com/Zielon/INSTA) of INSTA. Therefore, for any comparisons please use the C++ version.**

### Installation

Please follow the installation from [torch-ngp](https://github.com/ashawkey/torch-ngp#install). This implementation
requires to use `cuda` ray marching, which is enabled by `--cuda-ray`. Therefore, C++ extension must be
built. Moreover, for a fast nearest neighbor search we are
using [BVH](https://github.com/YuliangXiu/bvh-distance-queries) repository.

First create the environment.

```shell
conda create -n insta-pytorch python=3.9
conda activate insta-pytorch
pip install -r requirements.txt
```

After that use the `install.sh` script to compile all the required libraries and prepare the workbench.

### Usage

The dataset structure is compatible with the C++ version. In order to generate a new sequence input
follow [instructions](https://github.com/Zielon/INSTA#dataset-generation). The hardware requirement are the same as in the case of C++ version.

The [released avatars](https://keeper.mpdl.mpg.de/d/5ea4d2c300e9444a8b0b/) are compatible with training, however, the
checkpoints were generated for the C++ version.

```shell
# Offscreen rendering
python main_insta.py data/nerf/wojtek --workspace workspace/wojtek -O --tcnn

# GUI
python main_insta.py data/nerf/wojtek --workspace workspace/wojtek -O --tcnn --gui
```

Using GUI in the `Menu/Options` you can control the selected mesh.

### Citation

If you use this project in your research please cite INSTA:

```bibtex
@proceedings{INSTA:CVPR2023,
  author = {Zielonka, Wojciech and Bolkart, Timo and Thies, Justus},
  title = {Instant Volumetric Head Avatars},
  journal = {Conference on Computer Vision and Pattern Recognition},
  year = {2023}
}
```
