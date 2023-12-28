#!/bin/bash

# Without LSTM (DOGE):
NUM_ROUNDS_WITH_GRAD=1
USE_LSTM_VAR=False
USE_REPLAY_BUFFER=True

# # With LSTM (DOGE-M):
# NUM_ROUNDS_WITH_GRAD=3
# USE_LSTM_VAR=True
# USE_REPLAY_BUFFER=False # Using replay buffer with lstm gives worse results as lstm states are out-of-date.

# --test-non-learned # (Use this flag to testing FastDOG during evaluation.)
python train_doge.py --config-file configs/config_qaplib.py \
    OUT_REL_DIR QAPLIB_${USE_LSTM_VAR}_${NUM_ROUNDS_WITH_GRAD}/ \
    MODEL.USE_LSTM_VAR ${USE_LSTM_VAR} \
    TRAIN.NUM_ROUNDS_WITH_GRAD ${NUM_ROUNDS_WITH_GRAD} \
    TRAIN.USE_REPLAY_BUFFER ${USE_REPLAY_BUFFER}


exit 0
