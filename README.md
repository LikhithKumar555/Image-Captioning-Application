# Image Captioning Application

## Overview
This project automatically generates descriptive captions for images by combining computer vision and natural language processing. A ResNet50 encoder extracts visual features and an LSTM decoder turns them into a sentence. It is useful for digital media, education, social platforms and accessibility tools for the visually impaired.

## Features
- Generates captions for any input image
- ResNet50 encoder + LSTM decoder
- Custom vocabulary built from the training data
- Full training pipeline and inference script

## Tech Stack
Python, PyTorch, TorchVision, Pillow, scikit-learn, tqdm

## Dataset
[Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k): 8,000 images with 5 captions each. Place the images and `captions.txt` in the project folder.

## Setup
```bash
git clone https://github.com/yourusername/image-captioning-app.git
cd image-captioning-app
pip install -r requirements.txt
```
Update the hardcoded file paths in `a.py` and `b.py` for your system.

## Usage
```bash
python a.py   # train the model (saves trained_model.pth)
python b.py   # generate a caption for an image
```
To caption your own image, change `image_path` in `b.py`.

## Model Settings
Embedding 256, hidden size 512, 1 LSTM layer, Adam (lr 0.001), batch size 32, 5 epochs.

## Future Work
Transformer decoder, attention and beam search, BLEU/CIDEr scoring, web interface.

## Acknowledgements
Anurva Advanced Techserve Pvt Ltd (AAT Groups) and Apoorva for guidance and support.
