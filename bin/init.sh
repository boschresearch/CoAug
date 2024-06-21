#  Copyright (c) 2023 Robert Bosch GmbH
#  SPDX-License-Identifier: AGPL-3.0
#
#

python -m venv env
source env/bin/activate
pip install --upgrade pip
pip install spacy
python -m spacy download en_core_web_sm
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
mkdir pretrained_models
cd pretrained_models
wget https://dl.fbaipublicfiles.com/quip/quip.tar.gz
tar -xvzf quip.tar.gz
wget https://dl.fbaipublicfiles.com/quip/quip-hf.tar.gz
tar -xvzf quip-hf.tar.gz
rm quip-hf.tar.gz quip.tar.gz
cd ..
pip install gpustat