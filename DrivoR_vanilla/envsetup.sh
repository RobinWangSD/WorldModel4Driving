#!/bin/bash

set -e

cd /hugsim-storage/DrivoR/nuplan-devkit
pip install -e .
cd /hugsim-storage/DrivoR
pip install -e .
apt-get update && apt-get install -y ninja-build gcc-12 g++-12
pip install nvidia-cudnn-cu12==8.9.7.29
export LD_LIBRARY_PATH=$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0] + '/lib')"):$LD_LIBRARY_PATH
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/avl-west/nuplan/maps"
export NAVSIM_EXP_ROOT="/closed-loop-e2e/drivor-exp"
export NAVSIM_DEVKIT_ROOT="/hugsim-storage/DrivoR"
export OPENSCENE_DATA_ROOT="/avl-west/navsim"
pip install torch torchvision pytorch-lightning wandb
pip install timm

export HYDRA_FULL_ERROR=1 \
EXPERIMENT=training_drivoR_Nav1_traj_long_25epochs
AGENT=drivoR
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_full.py  \
    agent=$AGENT \
    experiment_name=$EXPERIMENT \
    train_test_split=navtrain \
    navsim_log_path=/avl-west/navsim/trainval_navsim_logs/trainval \
    train_test_split.data_split=trainval \
    +train_test_split.log_splits=null \
    cache_path=/closed-loop-e2e/drivor-exp/navsim_cache_nommcv \
    use_cache_without_dataset=true \
    sensor_blobs_path=/avl-west/navsim/trainval_sensor_blobs/trainval \
    trainer.params.max_epochs=25 \
    dataloader.params.prefetch_factor=2 \
    dataloader.params.batch_size=8 \
    agent.lr_args.name=AdamW \
    agent.lr_args.base_lr=0.0002 \
    agent.num_gpus=8 \
    agent.progress_bar=false \
    agent.config.refiner_ls_values=0.0 \
    agent.config.image_backbone.focus_front_cam=false \
    agent.config.one_token_per_traj=true \
    agent.config.refiner_num_heads=1 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.area_pred=false \
    agent.config.agent_pred=false \
    agent.config.ref_num=4 \
    agent.loss.prev_weight=0.0 \
    agent.config.long_trajectory_additional_poses=2 \
    seed=2