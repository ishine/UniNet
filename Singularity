Bootstrap: docker
From: pytorch/pytorch:latest
 
%labels
  Author Zhou Xiao
  Version v1.0.0
  build_date 2019 July 22

%post
  apt-get update
  apt-get upgrade -y
  apt-get install -y tmux htop ranger tree ncdu wget zip unzip nano
  apt-get autoclean
  
  /opt/conda/bin/pip install tensorflow==1.13.1 tensorboardX==1.1 numpy inflect librosa==0.6.3 pillow scipy matplotlib jupyter tqdm
