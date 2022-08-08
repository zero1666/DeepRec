#!/bin/bash 

#bash脚本中嵌套expect脚本
echo -e "\n -----开始安装Pytorch环境------------ \n" 

. /data/miniconda3/etc/profile.d/conda.sh

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

echo -e "\n  + 创建conda环境 py39_torch_cpu \n" 
{
/usr/bin/expect <<EOF  
set timeout  3000
spawn conda create -n py39_torch_cpu python=3.9
expect {
"*Proceed*" {send "y\r"}
} 
expect eof
EOF
}

echo -e "\n  + 进入conda环境 py39_torch_cpu \n" 
conda activate py39_torch_cpu

echo -e "\n  + py39_torch_cpu 下安装相关python工具包 \n" 
{
/usr/bin/expect <<EOF  
set timeout -1 
spawn conda install pytorch torchvision torchaudio cpuonly -c pytorch
expect {
"*Proceed*" {send "y\r"}
} 
expect eof
EOF
}

{
/usr/bin/expect <<EOF  
set timeout  30000
spawn conda install scikit-learn-intelex
expect {
"*Proceed*" {send "y\r"}
} 
expect eof
EOF
}

{
/usr/bin/expect <<EOF  
set timeout  300000
spawn conda install pandas matplotlib jupyter notebook scipy scikit-learn
expect {
"*Proceed*" {send "y\r"}
} 
expect eof
EOF
}

echo -e "\n  + py39_torch_cpu 下pip安装第三方python工具包 \n" 
pip install tensorboard   tensorboardX visdom opencv-python  pillow
pip install --upgrade pip
pip install psutil
pip install setproctitle

echo -e "\n  + py39_torch_cpu 下注册ipykernel环境\n" 
pip install ipykernel
python -m ipykernel install --user --name=py39_torch_cpu --display-name "py39_torch_cpu"



