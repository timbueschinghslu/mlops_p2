# MLOps P2
MLOps Course Project 2 @ HSLU

This repository lets you train the distilbert-base-uncased model on the GLUE MRPC dataset using the Hugging Face Transformers library. The training script is containerized using Docker. The training run can be tracked using Weights and Biases.

## Getting Started

### Prerequisites
- [Docker](https://www.docker.com/) installed on your local machine
- Weight and Biases account with API key


### Clone the Repository
```sh
git clone https://github.com/timbueschinghslu/mlops_p2.git
cd p2_mlops
```

Before building the dockerfile: Create a `.env` file in the root directory with the following content:
```WANDB_API_KEY=<your_wandb_api_key>```

### Building the Docker Image
Build the Docker image using the provided Dockerfile:
```sh
docker build -t mlops_p2 .
```


### Running the Docker Container
Run the Docker container to start a training run:
```sh
docker run --rm --env-file .env mlops_p2
```
This command executes the training script with default hyperparameters as specified in the `CMD` instruction of the Dockerfile. The best hyperparameters from the project 1 are set as default values. The .env file is used to pass the Weights and Biases API key to the container.

Important: 
- The `--rm` flag removes the container after the training run is finished. If you want to keep the container for debugging purposes, you can remove this flag.
- To successfully run the docker container in GitHub Codespaces choose the machine with 4 Cores and 16GB of RAM during the setup.

### Passing Custom Hyperparameters
You can override the default hyperparameters by specifying them in the `docker run` command:
```sh
docker run --rm mlops_p2 python train.py --checkpoint_dir /app/models --lr 1e-4 --warmup_steps 13 --weight_decay 0.02 --train_batch_size 128

```