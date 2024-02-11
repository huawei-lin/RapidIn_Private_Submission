#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 0 ../MP_main.py --config='./config.json'
