# DCGAN Human Face Generation Project

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) for generating human-like face images. Using the CelebA dataset, we aim to train a model that can create realistic face images from random noise vectors. Below is a breakdown of each code snippet in the project.

---

## Table of Contents
1. [Dataset Download](#dataset-download)
2. [Hyperparameters and Device Setup](#hyperparameters-and-device-setup)
3. [Image Visualization Function](#image-visualization-function)
4. [Weight Initialization](#weight-initialization)
5. [Generator Model](#generator-model)
6. [Discriminator Model](#discriminator-model)
7. [Model Creation and Initialization](#model-creation-and-initialization)
8. [Loss Function and Optimizers](#loss-function-and-optimizers)
9. [Training Step Function](#training-step-function)
10. [Recursive Training Function](#recursive-training-function)
11. [Visualizing Generated Images](#visualizing-generated-images)
12. [Batch Display of Generated Images](#batch-display-of-generated-images)
13. [Single Image Generation](#single-image-generation)

---

### 1. Dataset Download

```python
import kagglehub

# Download CelebA dataset using kagglehub
path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
dataset_path = path
print("Path to dataset files:", dataset_path)
```

**Explanation**: This snippet uses `kagglehub` to download the CelebA dataset from Kaggle, which contains labeled face images. The dataset path is printed for verification.

---

### 2. Hyperparameters and Device Setup

```python
# Hyperparameters
image_size = 64
batch_size = 128
nz = 100         # Latent vector size
lr = 0.0002      # Learning rate
beta1 = 0.5      # Beta1 for Adam optimizer
num_epochs = 50

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset loading
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

dataset = dsets.ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
```

**Explanation**: Here, we set up hyperparameters for the model, define the image transformations, and load the dataset into a `DataLoader`. We also choose a computation device (CPU or GPU).

---

### 3. Image Visualization Function

```python
import matplotlib.pyplot as plt
import numpy as np
import torchvision

def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Display a batch of training images
dataiter = iter(dataloader)
images, _ = next(dataiter)
imshow(torchvision.utils.make_grid(images[:8]))
```

**Explanation**: Defines a function to display images by un-normalizing and plotting them using Matplotlib. A batch of training images is visualized in a grid.

---

### 4. Weight Initialization

```python
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
```

**Explanation**: This function initializes the weights for convolutional, transposed convolutional, and batch normalization layers with a normal distribution to improve model training stability.

---

### 5. Generator Model

```python
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
```

**Explanation**: The Generator model is designed to take a noise vector and produce a realistic face image using a series of transposed convolutional layers with ReLU activations and batch normalization. The final layer uses a `Tanh` activation to produce pixel values between -1 and 1.

---

### 6. Discriminator Model

```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
```

**Explanation**: The Discriminator model uses convolutional layers and LeakyReLU activations to classify images as real or fake. It ends with a `Sigmoid` function to output probabilities.

---

### 7. Model Creation and Initialization

```python
netG = Generator().to(device)
netG.apply(weights_init)
netD = Discriminator().to(device)
netD.apply(weights_init)
```

**Explanation**: Instantiates the Generator and Discriminator models, moves them to the selected device, and applies the weight initialization function.

---

### 8. Loss Function and Optimizers

```python
criterion = nn.BCELoss()
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
```

**Explanation**: Sets up the binary cross-entropy loss for training, generates fixed noise for consistent sample generation, and initializes the optimizers for both the Generator and Discriminator models.

---

### 9. Training Step Function

```python
def train_step(data):
    real_images = data[0].to(device)
    label_real = torch.full((real_images.size(0),), 1., device=device)
    label_fake = torch.full((real_images.size(0),), 0., device=device)

    # Update Discriminator
    netD.zero_grad()
    output_real = netD(real_images).view(-1)
    lossD_real = criterion(output_real, label_real)
    lossD_real.backward()
    
    noise = torch.randn(real_images.size(0), nz, 1, 1, device=device)
    fake_images = netG(noise)
    output_fake = netD(fake_images.detach()).view(-1)
    lossD_fake = criterion(output_fake, label_fake)
    lossD_fake.backward()
    optimizerD.step()

    # Update Generator
    netG.zero_grad()
    output = netD(fake_images).view(-1)
    lossG = criterion(output, label_real)
    lossG.backward()
    optimizerG.step()

    return lossD_real + lossD_fake, lossG, fake_images
```

**Explanation**: Defines a single training step. It updates the Discriminator to classify real and fake images accurately and then updates the Generator to produce more realistic fake images.

---

### 10. Recursive Training Function

```python
def train_epoch(epoch, num_epochs):
    if epoch >= num_epochs:
        print("Training completed.")
        return
    dataiter = iter(dataloader)
    lossD, lossG, fake_images = 0, 0, None
    try:
        data = next(dataiter)
        lossD, lossG, fake_images = train_step(data)
    except StopIteration:
        pass
    
    vutils.save_image(fake_images.detach(), f"fake_samples_epoch_{epoch}.png", normalize=True)
    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss_D: {lossD:.4f} Loss_G: {lossG:.4f}")
    
    train

_epoch(epoch + 1, num_epochs)
    
train_epoch(0, num_epochs)
```

**Explanation**: A recursive training loop that saves generated images after each epoch. The loop continues until the specified number of epochs is reached.

---

### 11. Visualizing Generated Images

```python
import torchvision.utils as vutils
img_list = []
for epoch in range(num_epochs):
    fake_images = netG(fixed_noise).detach().cpu()
    img_list.append(vutils.make_grid(fake_images, padding=2, normalize=True))
```

**Explanation**: Collects generated images at each epoch for later visualization.

---

### 12. Batch Display of Generated Images

```python
import matplotlib.animation as animation

fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
plt.show()
```

**Explanation**: Creates an animated visualization of the generated images across epochs.

---

### 13. Single Image Generation

```python
noise = torch.randn(1, nz, 1, 1, device=device)
generated_image = netG(noise).cpu()
imshow(generated_image)
```

**Explanation**: Generates and displays a single new image using the trained Generator.

---

