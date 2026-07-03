# Image Captioning Application using Transformer Networks

An automated image captioning system that generates descriptive captions for images using deep learning techniques, combining computer vision and natural language processing.

##  Overview

This project implements an image captioning application that leverages Transformer networks to analyze visual content and generate accurate, context-aware captions. The application is designed to serve various sectors including digital media, education, accessibility services for the visually impaired, and social media platforms.

##  Features

- **Automated Image Captioning**: Generate descriptive captions for any input image
- **Deep Learning Architecture**: Uses ResNet50 as encoder and LSTM as decoder
- **Custom Vocabulary Building**: Creates a vocabulary from the training dataset
- **Model Training Pipeline**: Complete training workflow with validation
- **Image Preprocessing**: Standardized image preprocessing pipeline
- **Caption Generation**: Generate captions for new images using trained model

##  Technologies Used

- **Python**: Core programming language
- **PyTorch**: Deep learning framework
- **TorchVision**: Image preprocessing and pre-trained models
- **PIL**: Image processing
- **scikit-learn**: Data splitting and preprocessing
- **tqdm**: Progress bars for training

##  Project Structure

```
Image-Captioning-Application/
│
├── a.py                          # Training script
├── b.py                          # Testing and inference script
├── captions.txt                  # Dataset captions file
├── trained_model.pth             # Saved model weights
├── requirements.txt              # Project dependencies
├── README.md                     # Project documentation
│
├── image/                        # Directory for images
│   └── [image files]
│
└── Flickr8k_Dataset/             # Dataset directory
    └── Flicker8k_Dataset/
        └── [image files]
```

##  Dataset

The project uses the **Flickr8k Dataset** which contains:
- 8,000 images
- 5 captions per image
- Diverse scenes and objects

### Dataset Format
The `captions.txt` file should have the following format:
```
image_name.jpg\tcaption text
image_name.jpg\tcaption text
```

##  Installation

### Prerequisites

```bash
pip install torch torchvision Pillow scikit-learn tqdm matplotlib
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/image-captioning-app.git
cd image-captioning-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Flickr8k dataset and place it in the appropriate directory structure.

##  Usage

### Training the Model

Run the training script:
```bash
python a.py
```

The script will:
1. Load and preprocess the dataset
2. Build vocabulary
3. Train the model for 5 epochs (configurable)
4. Save the trained model as `trained_model.pth`

### Generating Captions

Run the inference script:
```bash
python b.py
```

The script will:
1. Load the trained model
2. Display the input image
3. Generate and print the caption

### Custom Usage

To generate captions for your own images:

```python
# In b.py, modify the image_path variable
image_path = 'path/to/your/image.jpg'
caption = generate_caption(image_path, model, transform, vocab)
print(f'Generated Caption: {caption}')
```

##  Model Architecture

### Encoder (ResNet50)
- Pre-trained ResNet50 as feature extractor
- Removes final classification layer
- Adds embedding layer to project features to desired dimension

### Decoder (LSTM)
- Embedding layer for word representations
- LSTM layers for sequence generation
- Linear layer for vocabulary output

### Training Configuration
- **Embedding Dimension**: 256
- **Hidden Dimension**: 512
- **Number of Layers**: 1
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: CrossEntropyLoss (ignoring padding)
- **Batch Size**: 32
- **Epochs**: 5

##  Results

The model generates captions that are:
- **Accurate**: Correctly identifies objects and scenes
- **Contextually Relevant**: Understands relationships between objects
- **Grammatically Coherent**: Produces natural language descriptions
- **Fluent**: Mimics human-like descriptions

### Sample Output
```
Image: Oceans35.jpg
Generated Caption: "A beautiful beach with blue ocean waves and sandy shore"
```

##  Future Improvements

1. **Transformer Implementation**: Replace LSTM with Transformer decoder
2. **Data Augmentation**: Expand dataset with more diverse images
3. **Model Enhancement**: Implement attention mechanisms and beam search
4. **Performance Metrics**: Add BLEU, CIDEr, and METEOR scoring
5. **Web Interface**: Create a user-friendly web application
6. **Real-time Processing**: Optimize for faster inference

##  Known Issues & Solutions

### File Path Issues
The code contains hardcoded file paths. Update these paths according to your system:
- `captions_path` in both files
- `image_path` patterns in dataset class
- `model_path` for loading/saving models

### GPU vs CPU
The code automatically detects CUDA. To force CPU usage:
```python
device = torch.device('cpu')
```

##  License

This project is part of an internship assignment at Anurva Advanced Techserve Pvt Ltd (AAT Groups).


##  Acknowledgements

- Anurva Advanced Techserve Pvt Ltd (AAT Groups)
- Apoorva for guidance and support
- Vidyavardhaka College of Engineering

##  References

1. Vaswani, A., et al. "Attention is All You Need"
2. Gupta, R., et al. "Image Captioning with Transformers: A Review"
3. Patel, S., et al. "Image Captioning Application using Transformer Networks"
4. Jain, N., et al. "Improving Image Captioning with Pre-trained Transformers"
5. Singh, M., et al. "End-to-End Image Captioning with Transformers"

