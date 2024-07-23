#!/usr/bin/env bash

export PROJECT_DIR=/llmft
source $PROJECT_DIR/scripts/misc/setup.sh

# args: task_name, max_train_samples, epochs, warmup_ratio, bsz, num_gpus, learning_rate, model_name_or_path, adapter_dim, lora_alpha, port
bash $PROJECT_DIR/scripts/pattern_verbalizer_ft/mnli/lora.sh rte 200 1 0.25 4 1 2e-4 facebook/opt-125m 8 -1 60000

#bash $PROJECT_DIR/scripts/vanilla_ft/mnli/run.sh              rte 64 1 0.5 32 1 1e-4 facebook/opt-125m 60000