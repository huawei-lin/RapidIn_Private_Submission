# RapidIn

RapidIn is a framework for estimating the influence of each training data on a given test generation for Large Language models (LLMs).

# Quick Start

Create a new env and install the requirements.
```
conda create -n RapidIn python=3.10
conda activate RapidIn
pip install -r requirements.txt
```

We provide a 52K alpaca dataset and a test generation data in `./data` dir.


1. Create a config file.
```
data:
  train_data_path: (required, str) the path to your training dataset.
  test_data_path: (required, str) the path to your test generation.

influence:
  outdir: (required, str) the path to program standard output.
  seed: (optional, int, default: 42) the random seed.
  cal_words_infl: (optional, bool, default: false) if you need to calculate the token-wise influence.
  grads_path: (optional, str) the path to save the full gradient vectors or RapidGrads.
  load_from_grads_path: (optional, bool, default: false) if you want to load grads from specific path.
  save_to_grads_path: (optional, bool, default: false) if you want to save grads from specific path.
  n_threads: (optional, int, default: 1) the number of threads for each GPU.
  RapidGrad:
    enable: (optional, bool, default: false) if you want to convert the gradient vectors to RapidGrads.
    K: (optional, int, default: 65536) expected dimensionality.
    n_perm: (optional, int, default: 20) the number of shuffles.
  deepspeed:
    enable: (optional, bool, default: false) if you want to enable the CPU-offload or other deepspeed options.
    config_path: (optional, str, default: None) the path to deepspeed config.
  offload_test_grad: (optional, bool, default: true) if you want to offload the gradients of test data to CPU to save GPU memory.
  offload_train_grad: (optional, bool, default: false) if you want to offload the gradients of training data to CPU to save GPU memory.
  top_k: (optional, int, default: 1000) output top-# influential data.

model:
  model_path: (required, str) the path to model.
  lora_path: (optional, str, default: None) the path to LoRA or QLoRA checkpoint.
  max_length: (optional, int, default: 512) the max length of the model.
  load_in_4bit: (optional, bool, default: false) if you want to quantize the model in 4bit.
```
We provide an example of config in `example/config.json`



2. Run the program on GPU 0
```
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 0 ./MP_main.py --config='./config.json'
```



