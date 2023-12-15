install anaconda
conda create --name=rl_soccer-twos_testing python=3.8
conda activate rl_soccer-twos_testing
pip install setuptools==65.5.0 wheel==0.38.4
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
<cd to this dir>
pip install -r requirements.txt
pip install protobuf==3.19.0


usage:
python testing.py <pytorch model file 1> <pytorch model file 2>