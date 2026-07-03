import os
import re
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence   

# Define the image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
  
# Load the captions from the dataset
def load_captions(file_path):
    captions = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split('\t')
            image_id, caption = parts[0], parts[1]
            if image_id not in captions:
                captions[image_id] = []
            captions[image_id].append(caption)
    return captions

captions_path = 'C:\internship project\Image Captioning Application\captions.txt'
image_captions = load_captions(captions_path)

# Split the dataset into train and test sets
image_ids = list(image_captions.keys())
train_ids, test_ids = train_test_split(image_ids, test_size=0.2, random_state=42)

# Define the vocabulary
all_captions = [caption for captions in image_captions.values() for caption in captions]
words = [word for caption in all_captions for word in caption.split()]
word_freq = {}
for word in words:
    if word not in word_freq:
        word_freq[word] = 0
    word_freq[word] += 1
vocab = {
    '<pad>': 0,
    '<start>': 1,
    '<end>': 2,
    '<unk>': 3
}
idx = 4
for word, freq in word_freq.items():
    if freq >= 5:  # Minimum word frequency threshold
        vocab[word] = idx
        idx += 1

# Custom dataset class for loading images and captions
class ImageCaptionDataset(Dataset):
    def __init__(self, image_ids, image_captions, transform=None, vocab=None):
        self.image_ids = image_ids
        self.image_captions = image_captions
        self.transform = transform
        self.vocab = vocab

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        pattern_removed = re.sub(r'#\d+', '', image_id)
        processed_image_id = pattern_removed + '_processed'
        image_path = f'C:\\internship project\\Image Captioning Application\\Flickr8k_Dataset\\Flicker8k_Dataset\\C:\internship project\Image Captioning Application\image\\{pattern_removed}'
        if not os.path.exists(image_path):
        # Image not found, return a placeholder image and caption
            placeholder_image = torch.zeros(3, 256, 256)
            placeholder_caption = torch.tensor([self.vocab['<pad>']])
            return placeholder_image, placeholder_caption
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        captions = self.image_captions[image_id]
        caption_tokens = [self.vocab.get(token, self.vocab['<unk>']) for caption in captions for token in caption.split()]
        caption_tokens = [self.vocab['<start>']] + caption_tokens + [self.vocab['<end>']]
        
        return image, torch.tensor(caption_tokens)

# Define the dataset and dataloaders
train_dataset = ImageCaptionDataset(train_ids, image_captions, transform=transform, vocab=vocab)
test_dataset = ImageCaptionDataset(test_ids, image_captions, transform=transform, vocab=vocab)

def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, 0)
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=vocab['<pad>'])
    lengths = [len(caption) for caption in captions]
    return images, captions_padded, lengths


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


# Define the ResNet encoder
# Define the ResNet encoder
class ResNetEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(ResNetEncoder, self).__init__()
        resnet = models.resnet50(weights="ResNet50_Weights.DEFAULT")
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.embedding = nn.Linear(resnet.fc.in_features, embedding_dim)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return x


# Define the LSTM-based decoder
class LSTMDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers, dropout):
        super(LSTMDecoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.fc(hiddens)
        return outputs

# Define the Image Captioning model
class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

# Initialize the encoder, decoder, and model
embedding_dim = 256  # Define your embedding dimension
hidden_dim = 512     # Define your hidden dimension
num_layers = 1       # Define your number of layers
dropout = 0          # Set dropout to 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = ResNetEncoder(embedding_dim).to(device)
decoder = LSTMDecoder(embedding_dim, hidden_dim, len(vocab), num_layers, dropout).to(device)

model = ImageCaptioningModel(encoder, decoder, device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for images, captions, lengths in tqdm(train_loader):
        images = images.to(device)
        captions = captions.to(device)
        optimizer.zero_grad()
        outputs = model(images, captions)
        # Calculate the loss based on the lengths of the sequences
        loss = 0
        for i in range(len(lengths)):
            loss += criterion(outputs[i, :lengths[i]], captions[i, :lengths[i]])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(train_loader):.4f}')
# Function to save the trained model
def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)

# Example usage
custom_model_path = 'C:\internship project\Image Captioning Application\ trained_model.pth'
save_model(model, custom_model_path)   