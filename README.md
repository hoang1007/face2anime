# Anime Style Transfer
This is the final project of the Computer Vision course at Vingroup AI Engineer Training Program 2023. In this project, we propose a novel method for style transfer between anime and real images using Cycle GAN.

## Setup
To setup the project, run the following command:
```
pip install -v -e .
```

## Data preparation
Download datasets from the following links and put them in the `data` folder:
- [Shinkai Makoto's scenery dataset](https://www.kaggle.com/datasets/hoang1808/shinkai-landscape-hoangvh)


Note: For other datasets described in the report, you can download out preprocessed data from [here](https://www.kaggle.com/datasets/hoang1808/human-anime-faces)

## Training
To train the model, run the following command:
```
python train.py --config CONFIG_FILE_PATH
```
where `CONFIG_FILE_PATH` is the path to the config file. For example, to train the model with the default config, run:
```
python train.py --config configs/default.yaml
```
when the training is done, the model will be saved in the `checkpoints` folder.

## Inference
To generate images using the trained model, run the following command:
```
python scripts/inference.py --checkpoint CHECKPOINT_PATH --image_path IMAGE_PATH
```
where `CHECKPOINT_PATH` is the path to the trained model, `IMAGE_PATH` is the path to the input image. For example, to generate an image using the default config, run:
```
python scripts/inference.py --checkpoint checkpoints/lastest.ckpt --image_path data/landscape/landscape.jpg
```
The generated image will be saved as `output.jpg` in the current directory.

## Experiments
You can track our experiments via [Weights & Biases](https://wandb.ai/hoang1007/face2anime) 🥰.

## Using the pre-trained models
You can download our pre-trained models with following links:
- [Anime face style transfer](https://github.com/hoang1007/face2anime/releases/download/face/epoch.389-step.195000.ckpt)
- [Anime scenery style transfer](https://github.com/hoang1007/face2anime/releases/download/scenery/epoch.199-step.498400.ckpt)
