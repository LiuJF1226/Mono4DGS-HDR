NUMPY_VERSION=1.26.4

which python
which pip

CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
CPP=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
$CC --version
$CXX --version

################################################################################    
pip install numpy==$NUMPY_VERSION
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
# conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install fvcore iopath -c fvcore -c iopath -c conda-forge -y
conda install nvidiacub -c bottler -y
conda install pytorch3d -c pytorch3d -y
conda install xformers -c xformers -y
pip install pyg_lib torch_scatter torch_geometric torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.1+cu118.html
# pip install pyg_lib torch_scatter torch_geometric torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.1+cu118.html
################################################################################

################################################################################
echo "Install other dependencies..."
pip install -r requirements.txt
pip install numpy==$NUMPY_VERSION
################################################################################

################################################################################
echo "Install GS..."
pip install --no-build-isolation lib_render/simple-knn
pip install --no-build-isolation lib_render/diff-gaussian-rasterization-alphadep-add3
pip install --no-build-isolation lib_render/diff-gaussian-rasterization-alphadep
pip install --no-build-isolation lib_render/diff-gaussian-rasterization-alphadep-cam
pip install --no-build-isolation lib_render/diff-gaussian-rasterization-alphadep-cam-add3
################################################################################

################################################################################
# pip install numpy==$NUMPY_VERSION
# pip install -U scikit-learn 
# pip install -U scipy
# pip install opencv-python==4.10.0.84
# pip install mmcv-full==1.7.2
################################################################################

################################################################################
# echo "Install JAX for evaluating DyCheck"
# pip install -r jax_requirements.txt
################################################################################
