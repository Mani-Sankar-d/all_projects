import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch.nn as nn
# ==== DEVICE ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== TRANSFORM ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

class Sampler(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, means, stds):
        noise = torch.randn_like(stds)
        return means + stds*noise

class Encoder(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3,1,1)
        self.norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,3,1,1)
        self.norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128,3,1,1)
        self.norm3 = nn.BatchNorm2d(128)
        self.means = nn.Linear(28*28*128, 256)
        self.stds = nn.Linear(28*28*128, 256)
        self.relu = nn.ReLU()
        self.max = nn.MaxPool2d(2,2)
        self.flat = nn.Flatten(1)
    def forward(self,x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.norm1(x)
        x=self.max(x)
        
        x=self.conv2(x)
        x=self.relu(x)
        x=self.norm2(x)
        x=self.max(x)
        
        x=self.conv3(x)
        x=self.relu(x)
        x=self.norm3(x)
        x=self.max(x)
        x=self.flat(x)
    
        return self.means(x), torch.nn.functional.softplus(self.stds(x))

class Decoder(nn.Module):
    def __init__(self,img_dim):
        super().__init__()
        self.fcc1 = nn.Linear(256,28*28*128)
        self.relu = nn.ReLU()
        self.conv1 = nn.ConvTranspose2d(128,64,4,2,1)
        self.norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.ConvTranspose2d(64,32,4,2,1)
        self.norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.ConvTranspose2d(32,3,4,2,1)
        self.tanh = nn.Tanh()
    def forward(self,samples):
        x = self.fcc1(samples)
        x = self.relu(x)
        x = x.view(-1, 128, 28, 28)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.conv3(x)
        return self.tanh(x)

class Model(nn.Module):
    def __init__(self,encoder,decoder,sampler):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler
    def forward(self, img):
        means, stds = self.encoder(img)
        samples = self.sampler(means,stds)
        new_img = self.decoder(samples)
        return new_img,means,stds

  

# ==== LOAD IMAGE ====
img_path = "E:\\ml_projects\\VOCdevkit\\VOCdevkit\\VOC2012\\JPEGImages\\2007_000170.jpg"
img = Image.open(img_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)  # shape: (1, 3, 224, 224)

# ==== LOAD MODEL ====
encoder = Encoder(224)
decoder = Decoder(224)
sampler = Sampler()
model = Model(encoder, decoder, sampler).to(device)
model.load_state_dict(torch.load("vae_model.pth", map_location=device))
model.eval()

# ==== RECONSTRUCT ====
with torch.no_grad():
    recon_img, _, _ = model(img_tensor)

# ==== DE-NORMALIZE ====
recon_img = (recon_img * 0.5 + 0.5).clamp(0, 1).squeeze().permute(1, 2, 0).cpu()
orig_img = (img_tensor * 0.5 + 0.5).clamp(0, 1).squeeze().permute(1, 2, 0).cpu()

# ==== VISUALIZE ====
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(orig_img)
ax[0].set_title("Original")
ax[0].axis("off")

ax[1].imshow(recon_img)
ax[1].set_title("Reconstructed")
ax[1].axis("off")

plt.suptitle("VAE Image Reconstruction")
plt.tight_layout()
plt.show()
