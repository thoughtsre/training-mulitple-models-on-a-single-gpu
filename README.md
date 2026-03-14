> **This repo has been migrated to [GitHub](https://github.com/thoughtsre/training-mulitple-models-on-a-single-gpu).**

# Training multiple models on a single GPU
This is an experiment to see how training multiple models concurrently on a single GPU migh affect training times.

## Experiment process
1. Create simple [model training script](./src/tmsg/main.py) to finetune [Resnet50](https://pytorch.org/vision/main/models/resnet.html)
2. Build training script into a small python library 
    - This is so that we can install it in Docker container
    - The library has a console script `train`. See `[project.scripts]` section in [`pyproject.toml`](./pyproject.toml)
    - `hatch build`
3. Build [Docker container](./Dockerfile.train)
    - Run [`build_train_docker.sh`](./build_train_docker.sh)
    - Or if you are using hatch: `hatch run expt:build`
4. Run concurrent model training using [`run_expt.py`](./run_expt.py)

## Reults analysis
The results are analysed in `analysis.ipynb`.
