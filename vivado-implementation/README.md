# Overview

To run this code script:

```
g++ -o barebones barebones.cpp
./run_barebones.sh xcku115-flvb2104-2-i "<16,8>" 1
```
Arguments are:
   - FPGA
   - Precision
   - Reuse Factor 

# Recipe

```
git clone https://github.com/hls-fpga-machine-learning/analysis-tools.git -b nvt/vivado-resources
cd analysis-tools/vivado-implementation
git clone https://github.com/hls-fpga-machine-learning/hls4ml.git
## Set the model as the 3-layer pruned in hls4ml/keras-to-hls/keras-config.yml
wget https://www.xilinx.com/support/packagefiles/usapackages/usaall.zip
unzip usaall.zip
mv usaall ..
g++ -o barebones barebones.cpp
./run_barebones.sh xcku115-flvb2104-2-i "<16,8>" 1
```
