#!/bin/bash

docker run --rm -d --gpus all -v $1:/data/ -v $2:/results/ tmsg:0.0.1