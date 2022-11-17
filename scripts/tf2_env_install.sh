#!/bin/bash 
CONDA_ENV_NAME=py37_tf26_cpu

#bash脚本中嵌套expect脚本
echo -e "\n -----开始安装Tensorflow 2.6.0 CPU 环境------------ \n" 

. /data/miniconda3/etc/profile.d/conda.sh

echo -e "\n  + 更新conda 默认配置 \n" 
{
/usr/bin/expect <<EOF  
set timeout  300
spawn conda update -n base -c defaults conda
expect {
"*Proceed*" {send "y\r"}
} 
expect eof
EOF
}

echo -e "\n  + 先删除conda环境 $CONDA_ENV_NAME \n" 
conda env remove -n py37_tf26_cpu 

echo -e "\n  + 创建conda环境 $CONDA_ENV_NAME \n" 
{
/usr/bin/expect <<EOF  
set timeout  3000
spawn conda create -n py37_tf26_cpu python=3.7
expect {
"*Proceed*" {send "y\r"}
} 
expect eof
EOF
}

echo -e "\n  + 进入conda环境 $CONDA_ENV_NAME\n" 
conda activate $CONDA_ENV_NAME 

echo -e "\n  + $CONDA_ENV_NAME 下安装相关python工具包 \n" 
pip install tensorflow==2.6
pip install numpy sklearn pandas keras==2.6
pip install matplotlib 
pip install pytest 
pip install networkx
pip install easydict

pip install psutil
pip install setproctitle
pip install tdqm

{
/usr/bin/expect <<EOF  
set timeout  300000
spawn conda install scipy scikit-learn
expect {
"*Proceed*" {send "y\r"}
} 
expect eof
EOF
}


echo -e "\n  + $CONDA_ENV_NAME 下注册ipykernel环境\n" 
pip install ipykernel
python -m ipykernel install --user --name=$CONDA_ENV_NAME --display-name "$CONDA_ENV_NAME"

# 设置bashrc环境
cp custom.bashrc ~/custom.bashrc 
source ~/.bashrc




