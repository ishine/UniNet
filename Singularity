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
  
  /opt/conda/bin/pip install tensorflow numpy inflect==0.2.5 librosa==0.6.0 scipy==1.0.0 tensorboardX==1.1 pillow scipy matplotlib Unidecode==1.0.22 jupyter tqdm
