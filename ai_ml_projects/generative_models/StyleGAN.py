import torch
from torch import nn as nn
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
tranformer = transforms.Compose(
    [transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)
dataset = CIFAR10('./root',train=True, transform=tranformer,download=True)
dataloader = DataLoader(dataset,8,shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.z_to_w = nn.Sequential(
            nn.Linear(n_in, n_in),
            nn.ReLU(),
            nn.Linear(n_in, n_in),
            nn.ReLU(),
            nn.Linear(n_in, n_in),
            nn.ReLU(),
            nn.Linear(n_in, n_in),
            nn.ReLU(),
            nn.Linear(n_in, n_in),
            nn.ReLU(),
            nn.Linear(n_in, n_in),
            nn.ReLU(),
            nn.Linear(n_in, n_in),
            nn.ReLU(),
            nn.Linear(n_in, n_in)
        )
    def forward(self, z):
        return self.z_to_w(z)
    
class Block(nn.Module):
    def __init__(self,w_dim,in_channels):
        super().__init__()
        self.affine1 = nn.Linear(w_dim, in_channels)
        self.affine2 = nn.Linear(w_dim, in_channels)
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels,4,2,1)
        self.noise_weight = nn.Parameter(torch.zeros(1,in_channels//2, 1, 1)).to(device)
        self.conv_1 = nn.Conv2d(in_channels, in_channels//2, 3, 1, 1)
        self.conv_2 = nn.Conv2d(in_channels//2,in_channels//2, 3, 1, 1)
        self.n_channels = in_channels
        self.relu = nn.ReLU()

    def forward(self, x, w):
        height = x.shape[2]*2
        width = x.shape[3]*2
        x = self.upsample(x)


        x = self.conv_1(x)
        x = self.relu(x)
        affine1 = self.affine1(w).to(device)
        bias = affine1[:, :self.n_channels//2]
        bias = bias[:,:,None,None]
        scale = affine1[:, (self.n_channels//2):]
        scale = scale[:,:,None, None]
        noise1 = torch.randn(1,1,height,width).to(device)
        # print(x.shape)
        # print(noise1.shape)
        # print(self.noise_weight.shape)
        x = x + self.noise_weight*noise1
        
        mean = x.mean(dim=[2,3])
        std = x.std(dim=[2,3], unbiased=False)
        x = (x-mean[:,:,None,None])/(std[:,:,None,None]+1e-8)
        x = x*scale +bias
        


        affine2 = self.affine2(w)
        # print(affine2.shape)
        bias = affine2[:,:self.n_channels//2]
        scale = affine2[:,self.n_channels//2:]
        scale = scale[:,:,None, None]
        bias = bias[:,:,None,None]
        noise2 = torch.randn(1,1,height,width).to(device)
        # print("x shape ",x.shape)
        # print("reqd shape ",self.noise_weight.shape)
        x = x + self.noise_weight*noise2
        x = self.conv_2(x)
        x = self.relu(x)
        mean = x.mean(dim=[2,3])
        std = x.std(dim=[2,3], unbiased=False)
        x = (x-mean[:,:,None,None])/(std[:,:,None,None]+1e-8)
        x = x*scale + bias
        return x

class CreatorBlock(nn.Module):
    def __init__(self,w_dim,in_channels):
        super().__init__()
        self.affine1 = nn.Linear(w_dim, 6)
        self.affine2 = nn.Linear(w_dim, 6)
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels,4,2,1)
        self.noise_weight = nn.Parameter(torch.zeros(1,3, 1, 1)).to(device)
        self.conv_1 = nn.Conv2d(in_channels, 3, 3, 1, 1)
        self.conv_2 = nn.Conv2d(3,3, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x, w):
        height = x.shape[2]*2
        width = x.shape[3]*2
        x = self.upsample(x)


        x = self.conv_1(x)
        x = self.relu(x)
        affine1 = self.affine1(w).to(device)
        bias = affine1[:, :3]
        bias = bias[:,:,None,None]
        scale = affine1[:,3:]
        scale = scale[:,:,None, None]
        noise1 = torch.randn(1,1,height,width).to(device)
        # print(x.shape)
        # print(noise1.shape)
        # print(self.noise_weight.shape)
        x = x + self.noise_weight*noise1
        
        mean = x.mean(dim=[2,3])
        std = x.std(dim=[2,3], unbiased=False)
        x = (x-mean[:,:,None,None])/(std[:,:,None,None]+1e-8)
        x = x*scale +bias
        


        affine2 = self.affine2(w)
        # print(affine2.shape)
        bias = affine2[:,:3]
        scale = affine2[:,3:]
        scale = scale[:,:,None, None]
        bias = bias[:,:,None,None]
        noise2 = torch.randn(1,1,height,width).to(device)
        # print("x shape ",x.shape)
        # print("reqd shape ",self.noise_weight.shape)
        x = x + self.noise_weight*noise2
        x = self.conv_2(x)
        x = self.relu(x)
        mean = x.mean(dim=[2,3])
        std = x.std(dim=[2,3], unbiased=False)
        x = (x-mean[:,:,None,None])/(std[:,:,None,None]+1e-8)
        x = x*scale + bias
        return x

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.classify = nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(256,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(512, 1, img_dim//16, 1,1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.classify(x).view(x.size(0),-1)

class Generator(nn.Module):
    def __init__(self,b1,b2,b3,b4,b5,b6,main,mlp):
        super().__init__()
        self.mlp=mlp
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.b4 = b4
        self.b5 = b5
        self.b6 = b6
        self.main = main
        self.const_input = nn.Parameter(torch.randn(1, 512, 4, 4)).to(device)
    def forward(self,w):
        batch_size = 8
        w=self.mlp(w)
        x = self.const_input.expand(batch_size, -1, -1, -1)
        x=self.b1(x,w)
        x=self.b2(x,w)
        x=self.b3(x,w)
        x=self.b4(x,w)
        x=self.b5(x,w)
        x=self.b6(x,w)
        x=self.main(x,w)
        return x


z=100
img_dim=64
lr=2e-4
mlp=MLP(z)
b1 = Block(z,512).to(device)
b2 = Block(z,256).to(device)
b3 = Block(z,128).to(device)
b4 = Block(z,64).to(device)
b5 = Block(z,32).to(device)
b6 = Block(z,16).to(device)
main = CreatorBlock(z,8).to(device)
generator = Generator(b1,b2,b3,b4,b5,b6,main,mlp)
discriminator = Discriminator(img_dim=img_dim).to(device)
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)
generator.apply(weights_init)
discriminator.apply(weights_init)


discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),lr=lr)
generator_optimizer = torch.optim.Adam(generator.parameters(),lr=lr)

criterion = nn.BCELoss()
def train(epochs):
    for epoch in range(epochs):
        L_G = 0
        L_D = 0
        for real_imgs,_ in dataloader:
            real_imgs = real_imgs.to(device)
            b_size = real_imgs.shape[0]
            vectors = torch.randn(b_size, z).to(device)
            fake_imgs = generator(vectors).detach()
            pred_label_real = discriminator(real_imgs)
            pred_label_fake = discriminator(fake_imgs)
            T_real = torch.ones_like(pred_label_real)
            T_fake = torch.zeros_like(pred_label_fake)
            # print("D(real):", pred_label_real.min().item(), pred_label_real.max().item())
            # print("D(fake):", pred_label_fake.min().item(), pred_label_fake.max().item())

            loss_D = criterion(pred_label_real, T_real) + criterion(pred_label_fake,T_fake)
            discriminator_optimizer.zero_grad()
            loss_D.backward()
            discriminator_optimizer.step()
            L_D+=loss_D.item()
         ######################################################

            vectors = torch.randn(b_size, z).to(device)
            fake_imgs = generator(vectors).to(device)
            
            pred_label_fake = discriminator(fake_imgs)
            T_fake = torch.ones_like(pred_label_fake)
            loss_G = criterion(pred_label_fake, T_fake)
            generator_optimizer.zero_grad()
            loss_G.backward()
            generator_optimizer.step()
            L_G+=loss_G.item()
        print(f"Epoch {epoch} loss_G: {L_G}  loss_D: {L_D}")
    torch.save(generator.state_dict(),"generator.pth")
        
train(10)