#!/bin/bash

COLOR='\033[0;32m'

echo -e "\n${COLOR}Installing dependencies..."

pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

bash scripts/install_ext.sh
cd raymarching
python setup.py build_ext --inplace
pip install .

cd ../bvh/
pip install -r requirements.txt
python setup.py install
