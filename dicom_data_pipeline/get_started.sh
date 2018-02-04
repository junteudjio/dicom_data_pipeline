#!/usr/bin/env bash

CODE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"

export PYTHONPATH=$PYTHONPATH:$CODE_DIR
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR

pip install -r $CODE_DIR/requirements.txt
pip install http://download.pytorch.org/whl/torch-0.3.0.post4-cp27-none-macosx_10_6_x86_64.whl
pip install torchvision

# create all the necessary directories
DATA_DIR=_data
DOWNLOAD_DIR=_download
PLOTS_DIR=_plots
LOGS_DIR=_logs
#MODEL_CKPTS_DIR=model_ckpts
TEST_LOGS_DIR=../tests/logs

mkdir -p $DATA_DIR
#mkdir -p $DOWNLOAD_DIR
mkdir -p $PLOTS_DIR
mkdir -p $LOGS_DIR
#mkdir -p $MODEL_CKPTS_DIR
mkdir -p $TEST_LOGS_DIR
