import torch 
from torchvision import datasets, transforms

print(f"\nWorking with Image Datasets (page 74)\n")

# Define a basic transformation
transform = transforms.Compose([transforms.ToTensor()])

# Download and load the training data
train_data = datasets.MNIST(
    root = 'data',
    train = True,
    download = True,
    transform = transform
)

print(f"Number of training images: {len(train_data)}")
print(f"Image shape: {train_data[0][0].shape}") #[1,28,28]

print(f"\nImage Transformations\n")

