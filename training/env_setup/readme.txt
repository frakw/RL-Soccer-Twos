install anaconda
install https://git-lfs.com/
conda create --name rl_train_ma-poca python=3.10.12
conda activate rl_train_ma-poca
<cd to this dir>
pip install -e ./ml-agents-envs
pip install -e ./ml-agents