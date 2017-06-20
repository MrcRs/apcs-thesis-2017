#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the kaggle dataset
# 2. Fine-tunes an Inception_ResNet_V2 model on the kaggle training set.
# 3. Evaluates the model on the kaggle validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_inception_resnet_v2_on_kaggle.sh

# Where the pre-trained Inception_ResNet_V2 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=output/checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=output/kaggle-models/inception_resnet_v2

# Where the dataset is saved to.
DATASET_DIR=../dataset/kaggle

# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/inception_resnet_v2_2016_08_30.ckpt ]; then
  wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
  tar -xvf inception_resnet_v2_2016_08_30.tar.gz
  mv inception_resnet_v2_2016_08_30.ckpt ${PRETRAINED_CHECKPOINT_DIR}/inception_resnet_v2_2016_08_30.ckpt
  rm inception_resnet_v2_2016_08_30.tar.gz
fi

# Download the dataset
python download_and_convert_data.py \
  --dataset_name=kaggle \
  --dataset_dir=${DATASET_DIR}

# Fine-tune only the new layers for 1000 steps.
# python train_image_classifier.py \
#   --train_dir=${TRAIN_DIR} \
#   --dataset_name=kaggle \
#   --dataset_split_name=train \
#   --dataset_dir=${DATASET_DIR} \
#   --model_name=inception_resnet_v2 \
#   --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_resnet_v2_2016_08_30.ckpt \
#   --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits  \
#   --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits  \
#   --max_number_of_steps=60000 
  # --learning_rate=0.01 \
  # --learning_rate_decay_type=fixed \
  # --optimizer=rmsprop \
  # --weight_decay=0.00004
  # --batch_size=16 \
  # --save_interval_secs=60 \
  # --save_summaries_secs=60 \
  # --log_every_n_steps=100 \

# Run evaluation tensorflow/core/kernels/logging_ops.cc:79] eval/Accuracy[0.8633]
# python eval_image_classifier.py \
#   --checkpoint_path=${TRAIN_DIR} \
#   --eval_dir=${TRAIN_DIR} \
#   --dataset_name=kaggle \
#   --dataset_split_name=test \
#   --dataset_dir=${DATASET_DIR} \
#   --model_name=inception_resnet_v2

# Fine-tune all the new layers for 2000 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=kaggle \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_resnet_v2 \
  --checkpoint_path=${TRAIN_DIR} \
  --max_number_of_steps=10000 \
  --batch_size=8 \
  --learning_rate=0.001 
  # --learning_rate_decay_type=fixed \
  # --save_interval_secs=60 \
  # --save_summaries_secs=60 \
  # --log_every_n_steps=10 \
  # --optimizer=rmsprop \
  # --weight_decay=0.00004

# Run evaluation tensorflow/core/kernels/logging_ops.cc:79] eval/Accuracy[0.9525]
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=kaggle \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_resnet_v2
