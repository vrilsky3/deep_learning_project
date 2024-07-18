#!/usr/bin/env bash

export PROJECT_DIR="/llmft"
source $PROJECT_DIR/scripts/misc/setup.sh

# args: task_name, max_train_samples, bsz, num_gpus, model_name_or_path, port
# args: task_name, max_train_samples, epochs, warmup_ratio, bsz, num_gpus, learning_rate, model_name_or_path, port

bash $PROJECT_DIR/scripts/vanilla_ft/mnli/run.sh rte 64 1 0.5 32 1 1e-4 facebook/opt-125m 60000