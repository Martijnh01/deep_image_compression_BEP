# Deep Image Compression

## Setup the dev environment
0. Download miniconda & install: https://docs.conda.io/projects/miniconda/en/latest/
1. Create an env: ```conda create -n fuda python=3.10 && conda activate fuda```
2. Install the needed packages: ```pip install -r requirements.txt```

## Make a wandb account and an API key
0. https://wandb.ai/
1. Generate an API key

## Running trainings:
0. ```python main.py {fit,validate,test,predict} -c config.yaml --root /path/to/folder/with/datasets```

python main.py fit -c vanilla_autoencoder_cifar10.yaml --root C:/Users/20202182/Documents/GitHub/deep_image_compression_BEP