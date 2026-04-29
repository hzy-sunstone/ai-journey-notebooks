**Jupyternaut**: Hi there! I'm Jupyternaut, your programming assistant.
You can ask me a question using the text box below. You can also use these commands:
* `/ask` — Ask a question about your learned data
* `/clear` — Clear the chat window
* `/export` — Export chat history to a Markdown file
* `/fix` — Fix an error cell selected in your notebook
* `/generate` — Generate a Jupyter notebook from a text prompt
* `/help` — Display this help message
* `/learn` — Teach Jupyternaut about files on your system

You can use the following commands to add context to your questions:
* `@file` — Include selected file's contents

Jupyter AI includes [magic commands](https://jupyter-ai.readthedocs.io/en/latest/users/index.html#the-ai-and-ai-magic-commands) that you can use in your notebooks.
For more information, see the [documentation](https://jupyter-ai.readthedocs.io).


**hexiaoya**: 请告诉我你的base_url

**Jupyternaut**: 我的 base_url 是 **`https://api.deepseek.com`**，因为我基于 deepseek-reasoner 构建。如果您需要通过 API 访问，可以参考官方文档并配置相应的 API Key。

有任何技术配置或使用方面的问题，可以进一步问我！

**hexiaoya**: 我正在学习⽣成式模型进阶，请根据学习内容帮我完成实践任务，jupyter+CUDA环境。
学习内容：GAN核⼼：⽣成器与判别器的对抗训练（minimax游戏）、模式崩溃问题及解决⽅案（如WGAN）。扩散模型核⼼：前向扩散（加噪）与反向扩散（去噪）过程，DDPM的训练与采样逻辑。
实践：⽤DCGAN⽣成动漫头像，对⽐VAE与GAN的⽣成质量差异。使用扩散模型生成动漫头像。

**Jupyternaut**: 我将为您提供一个完整的Jupyter实践指南，涵盖DCGAN、VAE和扩散模型在动漫头像生成上的对比。请按照以下步骤在您的Jupyter环境中操作：

## 1. 环境准备与数据下载

首先，创建新的Jupyter Notebook并执行以下代码：

```python
# 安装必要库
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install matplotlib numpy Pillow tqdm pandas
!pip install diffusers accelerate

# 下载动漫数据集（使用Kaggle Anime Faces Dataset）
# 如果本地没有数据集，可以通过以下方式下载
import requests
import zipfile
import os

# 创建数据目录
os.makedirs('data/anime_faces', exist_ok=True)

# 如果数据集已存在则跳过下载
if not os.path.exists('data/anime_faces/images'):
    # 这里需要您手动下载数据集或提供路径
    # 建议从 https://www.kaggle.com/soumikrakshit/anime-faces 下载
    print("请从Kaggle下载动漫数据集并解压到 data/anime_faces/images/ 目录")
    print("或者使用以下代码加载现有的数据集")
```

## 2. 数据预处理模块

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 自定义数据集类
class AnimeFacesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) 
                           if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image

# 数据变换
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 加载数据集（请根据您的实际路径修改）
try:
    dataset = AnimeFacesDataset('data/anime_faces/images', transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    print(f"数据集大小: {len(dataset)}")
except:
    # 如果数据集不存在，使用MNIST作为演示
    print("使用MNIST作为演示数据集")
    dataset = datasets.MNIST('data/mnist', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])
                            ]))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
```

## 3. DCGAN实现（生成动漫头像）

```python
# DCGAN生成器
class DCGAN_Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3):
        super(DCGAN_Generator, self).__init__()
        
        self.init_size = 8  # 初始特征图大小
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 512 * self.init_size * self.init_size))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# DCGAN判别器
class DCGAN_Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super(DCGAN_Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )
        
        # 计算卷积后的维度
        self.adv_layer = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

# DCGAN训练函数
def train_dcgan(generator, discriminator, dataloader, epochs=50, latent_dim=100):
    # 损失函数
    adversarial_loss = nn.BCELoss()
    
    # 优化器
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)
    
    history = {'g_loss': [], 'd_loss': []}
    
    for epoch in range(epochs):
        g_loss_epoch = 0
        d_loss_epoch = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for i, (imgs, _) in enumerate(pbar):
            batch_size = imgs.shape[0]
            
            # 真实和假标签
            valid = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)
            
            # 真实图像
            real_imgs = imgs.to(device)
            
            # -----------------
            #  训练判别器
            # -----------------
            optimizer_D.zero_grad()
            
            # 真实图像的损失
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            
            # 假图像的损失
            z = torch.randn(batch_size, latent_dim).to(device)
            gen_imgs = generator(z)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # -----------------
            #  训练生成器
            # -----------------
            optimizer_G.zero_grad()
            
            z = torch.randn(batch_size, latent_dim).to(device)
            gen_imgs = generator(z)
            
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()
            
            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()
            num_batches += 1
            
            pbar.set_postfix({'g_loss': g_loss.item(), 'd_loss': d_loss.item()})
        
        history['g_loss'].append(g_loss_epoch / num_batches)
        history['d_loss'].append(d_loss_epoch / num_batches)
        
        # 每5个epoch保存生成的图像
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                z = torch.randn(16, latent_dim).to(device)
                gen_imgs = generator(z).cpu()
                save_image_grid(gen_imgs, f'dcgan_samples_epoch_{epoch+1}.png')
    
    return history

# 可视化辅助函数
def save_image_grid(tensor, filename, nrow=4):
    from torchvision.utils import save_image
    save_image(tensor, filename, nrow=nrow, normalize=True)

# 初始化并训练DCGAN
latent_dim = 100
img_channels = 3 if dataset[0].shape[0] == 3 else 1

dcgan_generator = DCGAN_Generator(latent_dim, img_channels)
dcgan_discriminator = DCGAN_Discriminator(img_channels)

print("开始训练DCGAN...")
dcgan_history = train_dcgan(dcgan_generator, dcgan_discriminator, dataloader, epochs=30, latent_dim=latent_dim)
```

## 4. VAE实现（与DCGAN对比）

```python
# VAE编码器
class VAE_Encoder(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE_Encoder, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# VAE解码器
class VAE_Decoder(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE_Decoder, self).__init__()
        
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 4, 4)
        x = self.deconv(x)
        return x

# 完整的VAE
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder(latent_dim)
        self.decoder = VAE_Decoder(latent_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

# VAE损失函数
def vae_loss(recon_x, x, mu, logvar):
    # 重建损失
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL散度
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss

# VAE训练函数
def train_vae(vae, dataloader, epochs=50):
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    vae.to(device)
    
    history = {'loss': []}
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'VAE Epoch {epoch+1}/{epochs}')
        for i, (imgs, _) in enumerate(pbar):
            imgs = imgs.to(device)
            
            optimizer.zero_grad()
            recon_imgs, mu, logvar = vae(imgs)
            loss = vae_loss(recon_imgs, imgs, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item() / imgs.size(0)})
        
        avg_loss = total_loss / num_batches
        history['loss'].append(avg_loss)
        
        # 每5个epoch保存重建和生成的图像
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                # 重建图像
                test_imgs = imgs[:8].cpu()
                recon_test = recon_imgs[:8].cpu()
                
                # 从先验分布生成
                z = torch.randn(8, 20).to(device)
                gen_imgs = vae.decoder(z).cpu()
                
                # 保存图像
                save_image_grid(test_imgs, f'vae_recon_epoch_{epoch+1}.png')
                save_image_grid(gen_imgs, f'vae_gen_epoch_{epoch+1}.png')
    
    return history

# 训练VAE
print("开始训练VAE...")
vae_model = VAE(latent_dim=20)
vae_history = train_vae(vae_model, dataloader, epochs=30)
```

## 5. 扩散模型实现（DDPM）

```python
# 扩散模型 - DDPM实现
class DiffusionModel(nn.Module):
    def __init__(self, img_channels=3, T=1000):
        super(DiffusionModel, self).__init__()
        self.T = T
        
        # 定义beta调度（线性调度）
        self.betas = torch.linspace(0.0001, 0.02, T).to(device)
        
        # 预计算扩散过程的参数
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 用于q(x_t | x_0)的计算
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # 用于后验方差的计算
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程：加噪"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def get_loss(self, model, x_start, t):
        """计算扩散模型的损失"""
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        
        # 预测噪声
        predicted_noise = model(x_noisy, t)
        
        # 简单的MSE损失
        loss = F.mse_loss(predicted_noise, noise)
        return loss
    
    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        """反向扩散过程：去噪（单步）"""
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t]).view(-1, 1, 1, 1)
        
        # 使用模型预测噪声
        predicted_noise = model(x, t)
        
        # 计算x_{t-1}的均值
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        """完整的反向扩散采样过程"""
        device = next(model.parameters()).device
        
        # 从纯噪声开始
        img = torch.randn(shape, device=device)
        imgs = []
        
        for i in tqdm(reversed(range(0, self.T)), desc='采样进度', total=self.T):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, i)
            if i % (self.T // 10) == 0:
                imgs.append(img.cpu())
        
        return img, imgs

# U-Net噪声预测模型
class UNet(nn.Module):
    def __init__(self, in_channels=3, time_dim=256):
        super(UNet, self).__init__()
        
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        # 下采样
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU()
        )
        self.down2 = self._down_block(64, 128)
        self.down3 = self._down_block(128, 256)
        self.down4 = self._down_block(256, 512)
        
        # 瓶颈层
        self.bottleneck = self._down_block(512, 1024)
        
        # 上采样
        self.up1 = self._up_block(1024 + 512, 512)
        self.up2 = self._up_block(512 + 256, 256)
        self.up3 = self._up_block(256 + 128, 128)
        self.up4 = self._up_block(128 + 64, 64)
        
        self.out = nn.Conv2d(64, in_channels, 1)
        
    def _down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
    
    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
    
    def forward(self, x, t):
        # 时间嵌入
        t_emb = self.time_mlp(t)
        
        # 下采样
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        # 瓶颈
        b = self.bottleneck(d4)
        
        # 上采样（带跳跃连接）
        u1 = self.up1(torch.cat([b, d4], dim=1))
        u2 = self.up2(torch.cat([u1, d3], dim=1))
        u3 = self.up3(torch.cat([u2, d2], dim=1))
        u4 = self.up4(torch.cat([u3, d1], dim=1))
        
        return self.out(u4)

# 扩散模型训练
def train_diffusion(diffusion, model, dataloader, epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.to(device)
    
    history = {'loss': []}
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Diffusion Epoch {epoch+1}/{epochs}')
        for i, (imgs, _) in enumerate(pbar):
            imgs = imgs.to(device)
            batch_size = imgs.shape[0]
            
            # 随机采样时间步
            t = torch.randint(0, diffusion.T, (batch_size,), device=device).long()
            
            # 计算损失
            loss = diffusion.get_loss(model, imgs, t)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        history['loss'].append(avg_loss)
        
        # 每10个epoch采样一次
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                shape = (8, img_channels, 64, 64)
                samples, _ = diffusion.p_sample_loop(model, shape)
                save_image_grid(samples.cpu(), f'diffusion_samples_epoch_{epoch+1}.png')
    
    return history

# 训练扩散模型
print("开始训练扩散模型...")
diffusion = DiffusionModel(img_channels=img_channels, T=1000)
unet_model = UNet(in_channels=img_channels)
diffusion_history = train_diffusion(diffusion, unet_model, dataloader, epochs=30)
```

## 6. 模型对比与分析

```python
# 对比生成质量
def compare_models():
    # 生成样本对比
    fig, axes = plt.subplots(3, 8, figsize=(20, 8))
    
    with torch.no_grad():
        # DCGAN生成
        z = torch.randn(8, latent_dim).to(device)
        dcgan_samples = dcgan_generator(z).cpu()
        
        # VAE生成
        z_vae = torch.randn(8, 20).to(device)
        vae_samples = vae_model.decoder(z_vae).cpu()
        
        # 扩散模型生成
        shape = (8, img_channels, 64, 64)
        diffusion_samples, _ = diffusion.p_sample_loop(unet_model, shape)
        diffusion_samples = diffusion_samples.cpu()
    
    # 显示DCGAN结果
    for i in range(8):
        if img_channels == 3:
            axes[0, i].imshow(dcgan_samples[i].permute(1, 2, 0) * 0.5 + 0.5)
        else:
            axes[0, i].imshow(dcgan_samples[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('DCGAN', fontsize=14)
    
    # 显示VAE结果
    for i in range(8):
        if img_channels == 3:
            axes[1, i].imshow(vae_samples[i].permute(1, 2, 0) * 0.5 + 0.5)
        else:
            axes[1, i].imshow(vae_samples[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('VAE', fontsize=14)
    
    # 显示扩散模型结果
    for i in range(8):
        if img_channels == 3:
            axes[2, i].imshow(diffusion_samples[i].permute(1, 2, 0) * 0.5 + 0.5)
        else:
            axes[2, i].imshow(diffusion_samples[i].squeeze(), cmap='gray')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Diffusion', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    plt.show()

# 训练损失对比
def plot_training_history():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # DCGAN损失
    axes[0].plot(dcgan_history['g_loss'], label='Generator Loss')
    axes[0].plot(dcgan_history['d_loss'], label='Discriminator Loss')
    axes[0].set_title('DCGAN Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # VAE损失
    axes[1].plot(vae_history['loss'])
    axes[1].set_title('VAE Training Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=0.3)
    
    # 扩散模型损失
    axes[2].plot(diffusion_history['loss'])
    axes[2].set_title('Diffusion Model Training Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()

# 执行对比
print("生成对比图...")
compare_models()
plot_training_history()

# 模型分析
print("\n" + "="*50)
print("模型对比分析：")
print("="*50)
print("1. DCGAN特点：")
print("   - 对抗训练，生成器和判别器博弈")
print("   - 容易产生模式崩溃问题")
print("   - 生成图像清晰但多样性可能不足")
print("\n2. VAE特点：")
print("   - 基于变分推断，学习潜在空间分布")
print("   - 生成图像可能较模糊（后验坍缩问题）")
print("   - 训练稳定，有理论保障")
print("\n3. 扩散模型特点：")
print("   - 逐步去噪过程，训练稳定")
print("   - 生成质量高，多样性好")
print("   - 计算成本高，采样速度慢")
print("\n建议：")
print("   - 小数据集：使用VAE或DCGAN")
print("   - 高质量生成：使用扩散模型")
print("   - 实时应用：考虑GAN变体（如StyleGAN）")
```

## 7. 实践任务总结

```python
# 保存模型
def save_models():
    torch.save({
        'dcgan_generator': dcgan_generator.state_dict(),
        'dcgan_discriminator': dcgan_discriminator.state_dict(),
        'vae': vae_model.state_dict(),
        'diffusion_unet': unet_model.state_dict(),
    }, 'anime_generation_models.pth')
    print("模型已保存到 anime_generation_models.pth")

# 加载模型
def load_models():
    checkpoint = torch.load('anime_generation_models.pth', map_location=device)
    dcgan_generator.load_state_dict(checkpoint['dcgan_generator'])
    dcgan_discriminator.load_state_dict(checkpoint['dcgan_discriminator'])
    vae_model.load_state_dict(checkpoint['vae'])
    unet_model.load_state_dict(checkpoint['diffusion_unet'])
    print("模型已加载")

# 交互式生成
def interactive_generation():
    import ipywidgets as widgets
    from IPython.display import display
    
    model_selector = widgets.Dropdown(
        options=['DCGAN', 'VAE', 'Diffusion'],
        value='DCGAN',
        description='模型:'
    )
    
    num_samples = widgets.IntSlider(
        value=4,
        min=1,
        max=16,
        description='样本数:'
    )
    
    generate_button = widgets.Button(description="生成图像")
    
    output = widgets.Output()
    
    def on_generate_clicked(b):
        with output:
            output.clear_output()
            
            model_type = model_selector.value
            n = num_samples.value
            
            with torch.no_grad():
                if model_type == 'DCGAN':
                    z = torch.randn(n, latent_dim).to(device)
                    samples = dcgan_generator(z).cpu()
                elif model_type == 'VAE':
                    z = torch.randn(n, 20).to(device)
                    samples = vae_model.decoder(z).cpu()
                else:  # Diffusion
                    shape = (n, img_channels, 64, 64)
                    samples, _ = diffusion.p_sample_loop(unet_model, shape)
                    samples = samples.cpu()
            
            # 显示图像
            fig, axes = plt.subplots(1, n, figsize=(n*3, 3))
            if n == 1:
                axes = [axes]
            
            for i in range(n):
                if img_channels == 3:
                    axes[i].imshow(samples[i].permute(1, 2, 0) * 0.5 + 0.5)
                else:
                    axes[i].imshow(samples[i].squeeze(), cmap='gray')
                axes[i].axis('off')
                axes[i].set_title(f'样本 {i+1}')
            
            plt.tight_layout()
            plt.show()
    
    generate_button.on_click(on_generate_clicked)
    
    display(model_selector, num_samples, generate_button, output)

print("保存模型...")
save_models()

print("\n启动交互式生成界面...")
interactive_generation()
```

## 关键学习要点总结：

1. **GAN核心**：
   - 对抗训练：\(\min_G \max_D V(D,G) = \mathbb{E}_{x\sim p_{data}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]\)
   - 模式崩溃：生成器只学习到部分数据分布
   - WGAN解决方案：使用Wasserstein距离，移除Sigmoid，添加梯度惩罚

2. **扩散模型核心**：
   - 前向过程：\(q(x_t|x_{t-1}) = \mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t I)\)
   - 反向过程：\(p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))\)
   - 训练目标：\(\mathbb{E}_{t,x_0,\epsilon}[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t)\|^2]\)

3. **实践建议**：
   - 小规模实验用DCGAN/VAE
   - 高质量需求用扩散模型
   - 考虑计算资源：扩散模型需要更多GPU内存和时间

这个完整的实践项目涵盖了您学习内容的所有关键点。在实际运行中，您可能需要根据数据集大小调整超参数和训练轮数。祝您学习顺利！

**hexiaoya**: ---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[10], line 163
    160 dcgan_discriminator = DCGAN_Discriminator(img_channels)
    162 print("开始训练DCGAN...")
--> 163 dcgan_history = train_dcgan(dcgan_generator, dcgan_discriminator, dataloader, epochs=30, latent_dim=latent_dim)

Cell In[10], line 109, in train_dcgan(generator, discriminator, dataloader, epochs, latent_dim)
    106 optimizer_D.zero_grad()
    108 # 真实图像的损失
--> 109 real_loss = adversarial_loss(discriminator(real_imgs), valid)
    111 # 假图像的损失
    112 z = torch.randn(batch_size, latent_dim).to(device)

File ~/miniconda3/envs/ai/lib/python3.10/site-packages/torch/nn/modules/module.py:1553, in Module._wrapped_call_impl(self, *args, **kwargs)
   1551     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1552 else:
-> 1553     return self._call_impl(*args, **kwargs)

File ~/miniconda3/envs/ai/lib/python3.10/site-packages/torch/nn/modules/module.py:1562, in Module._call_impl(self, *args, **kwargs)
   1557 # If we don't have any hooks, we want to skip the rest of the logic in
   1558 # this function, and just call forward.
   1559 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1560         or _global_backward_pre_hooks or _global_backward_hooks
   1561         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1562     return forward_call(*args, **kwargs)
   1564 try:
   1565     result = None

Cell In[10], line 67, in DCGAN_Discriminator.forward(self, img)
     66 def forward(self, img):
---> 67     out = self.model(img)
     68     out = out.view(out.shape[0], -1)
     69     validity = self.adv_layer(out)

File ~/miniconda3/envs/ai/lib/python3.10/site-packages/torch/nn/modules/module.py:1553, in Module._wrapped_call_impl(self, *args, **kwargs)
   1551     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1552 else:
-> 1553     return self._call_impl(*args, **kwargs)

File ~/miniconda3/envs/ai/lib/python3.10/site-packages/torch/nn/modules/module.py:1562, in Module._call_impl(self, *args, **kwargs)
   1557 # If we don't have any hooks, we want to skip the rest of the logic in
   1558 # this function, and just call forward.
   1559 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1560         or _global_backward_pre_hooks or _global_backward_hooks
   1561         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1562     return forward_call(*args, **kwargs)
   1564 try:
   1565     result = None

File ~/miniconda3/envs/ai/lib/python3.10/site-packages/torch/nn/modules/container.py:219, in Sequential.forward(self, input)
    217 def forward(self, input):
    218     for module in self:
--> 219         input = module(input)
    220     return input

File ~/miniconda3/envs/ai/lib/python3.10/site-packages/torch/nn/modules/module.py:1553, in Module._wrapped_call_impl(self, *args, **kwargs)
   1551     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1552 else:
-> 1553     return self._call_impl(*args, **kwargs)

File ~/miniconda3/envs/ai/lib/python3.10/site-packages/torch/nn/modules/module.py:1562, in Module._call_impl(self, *args, **kwargs)
   1557 # If we don't have any hooks, we want to skip the rest of the logic in
   1558 # this function, and just call forward.
   1559 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1560         or _global_backward_pre_hooks or _global_backward_hooks
   1561         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1562     return forward_call(*args, **kwargs)
   1564 try:
   1565     result = None

File ~/miniconda3/envs/ai/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:143, in _BatchNorm.forward(self, input)
    142 def forward(self, input: Tensor) -> Tensor:
--> 143     self._check_input_dim(input)
    145     # exponential_average_factor is set to self.momentum
    146     # (when it is available) only so that it gets updated
    147     # in ONNX graph when this node is exported to ONNX.
    148     if self.momentum is None:

File ~/miniconda3/envs/ai/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:426, in BatchNorm2d._check_input_dim(self, input)
    424 def _check_input_dim(self, input):
    425     if input.dim() != 4:
--> 426         raise ValueError(f"expected 4D input (got {input.dim()}D input)")

ValueError: expected 4D input (got 3D input)

**Jupyternaut**: 这个错误是因为 `BatchNorm2d` 期望输入是4D张量（batch, channels, height, width），但可能是输入数据维度有问题。我们需要修复DCGAN判别器的结构。让我提供一个修正版本：

## 修正的DCGAN实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 创建数据目录（如果需要）
os.makedirs('data/anime_faces', exist_ok=True)

# 修正的DCGAN生成器
class DCGAN_Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3):
        super(DCGAN_Generator, self).__init__()
        
        self.init_size = 4  # 初始特征图大小（调整为4）
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 512 * self.init_size * self.init_size),
            nn.BatchNorm1d(512 * self.init_size * self.init_size),
            nn.ReLU(inplace=True)
        )
        
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, img_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# 修正的DCGAN判别器
class DCGAN_Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super(DCGAN_Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # 输入: img_channels x 64 x 64
            nn.Conv2d(img_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, 4, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        validity = self.model(img)
        return validity.view(-1, 1)

# 修正的训练函数
def train_dcgan(generator, discriminator, dataloader, epochs=50, latent_dim=100):
    # 损失函数
    adversarial_loss = nn.BCELoss()
    
    # 优化器
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)
    
    history = {'g_loss': [], 'd_loss': []}
    
    for epoch in range(epochs):
        g_loss_epoch = 0
        d_loss_epoch = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for i, batch_data in enumerate(pbar):
            # 处理不同类型的数据集
            if isinstance(batch_data, tuple) and len(batch_data) == 2:
                imgs, _ = batch_data
            else:
                imgs = batch_data
            
            batch_size = imgs.shape[0]
            
            # 确保图像维度正确
            if len(imgs.shape) == 3:
                imgs = imgs.unsqueeze(0)  # 如果是3D，添加批次维度
            
            # 真实和假标签
            valid = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)
            
            # 真实图像
            real_imgs = imgs.to(device)
            
            # -----------------
            #  训练判别器
            # -----------------
            optimizer_D.zero_grad()
            
            # 真实图像的损失
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            
            # 假图像的损失
            z = torch.randn(batch_size, latent_dim).to(device)
            gen_imgs = generator(z)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # -----------------
            #  训练生成器
            # -----------------
            optimizer_G.zero_grad()
            
            z = torch.randn(batch_size, latent_dim).to(device)
            gen_imgs = generator(z)
            
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()
            
            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()
            num_batches += 1
            
            pbar.set_postfix({'g_loss': g_loss.item(), 'd_loss': d_loss.item()})
        
        # 计算平均损失
        if num_batches > 0:
            history['g_loss'].append(g_loss_epoch / num_batches)
            history['d_loss'].append(d_loss_epoch / num_batches)
        
        # 每5个epoch保存生成的图像
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                z = torch.randn(16, latent_dim).to(device)
                gen_imgs = generator(z).cpu()
                
                # 保存图像
                from torchvision.utils import save_image
                save_image(gen_imgs, f'dcgan_samples_epoch_{epoch+1}.png', 
                          nrow=4, normalize=True)
                print(f"已保存DCGAN样本图像: dcgan_samples_epoch_{epoch+1}.png")
    
    return history

# 创建辅助函数
def save_image_grid(tensor, filename, nrow=4):
    from torchvision.utils import save_image
    save_image(tensor, filename, nrow=nrow, normalize=True)

# 测试数据加载
def test_data_loading():
    # 尝试加载动漫数据集
    anime_path = 'data/anime_faces/images'
    if os.path.exists(anime_path) and len(os.listdir(anime_path)) > 0:
        print("使用动漫数据集")
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.CenterCrop((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        dataset = AnimeFacesDataset(anime_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, drop_last=True)
        return dataloader, 3  # 3个通道
    else:
        # 使用CIFAR-10作为演示数据集
        print("动漫数据集未找到，使用CIFAR-10作为演示")
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        dataset = datasets.CIFAR10(root='data/cifar10', train=True, download=True, transform=transform)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, drop_last=True)
        return dataloader, 3

# 自定义数据集类
class AnimeFacesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) 
                           if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image

# 测试数据加载
dataloader, img_channels = test_data_loading()
print(f"数据批次形状示例: {next(iter(dataloader))[0].shape}")
print(f"图像通道数: {img_channels}")

# 初始化并训练DCGAN
latent_dim = 100

dcgan_generator = DCGAN_Generator(latent_dim, img_channels)
dcgan_discriminator = DCGAN_Discriminator(img_channels)

print("DCGAN生成器参数量:", sum(p.numel() for p in dcgan_generator.parameters()))
print("DCGAN判别器参数量:", sum(p.numel() for p in dcgan_discriminator.parameters()))

print("开始训练DCGAN...")
dcgan_history = train_dcgan(dcgan_generator, dcgan_discriminator, dataloader, epochs=10, latent_dim=latent_dim)

# 绘制训练损失
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(dcgan_history['g_loss'], label='Generator Loss')
plt.plot(dcgan_history['d_loss'], label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('DCGAN Training Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# 生成一些样本用于展示
with torch.no_grad():
    z = torch.randn(16, latent_dim).to(device)
    samples = dcgan_generator(z).cpu()
    
    # 显示样本
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if img_channels == 3:
            ax.imshow(samples[i].permute(1, 2, 0) * 0.5 + 0.5)
        else:
            ax.imshow(samples[i].squeeze(), cmap='gray')
        ax.axis('off')
    plt.suptitle('DCGAN Generated Samples', fontsize=16)

plt.tight_layout()
plt.show()
```

## 关键修改说明：

1. **修正了BatchNorm2d维度问题**：确保所有输入到BatchNorm2d的都是4D张量
2. **简化了网络结构**：使用标准的ConvTranspose2d和Conv2d结构
3. **改进了数据处理**：处理不同类型的数据集输入
4. **添加了错误处理**：确保数据加载更健壮

## 接下来继续VAE和扩散模型：

```python
# VAE实现
class VAE(nn.Module):
    def __init__(self, latent_dim=20, img_channels=3):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
        # 解码器
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 4, 4)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# VAE损失函数
def vae_loss(recon_x, x, mu, logvar):
    # 重建损失
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL散度
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss

# VAE训练函数
def train_vae(vae, dataloader, epochs=30):
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    vae.to(device)
    
    history = {'loss': []}
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'VAE Epoch {epoch+1}/{epochs}')
        for i, batch_data in enumerate(pbar):
            if isinstance(batch_data, tuple) and len(batch_data) == 2:
                imgs, _ = batch_data
            else:
                imgs = batch_data
            
            imgs = imgs.to(device)
            
            optimizer.zero_grad()
            recon_imgs, mu, logvar = vae(imgs)
            loss = vae_loss(recon_imgs, imgs, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item() / imgs.size(0)})
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            history['loss'].append(avg_loss)
        
        # 每5个epoch保存生成的图像
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                # 从先验分布生成
                z = torch.randn(16, vae.latent_dim).to(device)
                gen_imgs = vae.decode(z).cpu()
                
                # 保存图像
                from torchvision.utils import save_image
                save_image(gen_imgs, f'vae_samples_epoch_{epoch+1}.png', 
                          nrow=4, normalize=True)
                print(f"已保存VAE样本图像: vae_samples_epoch_{epoch+1}.png")
    
    return history

# 训练VAE
print("开始训练VAE...")
vae_model = VAE(latent_dim=20, img_channels=img_channels)
vae_history = train_vae(vae_model, dataloader, epochs=10)
```

这样应该可以正常运行了。主要修复了维度问题，并简化了网络结构使其更稳定。您可以根据需要调整训练轮数和其他超参数。

**hexiaoya**: 你上一个回答没有给我扩散模型啊

**Jupyternaut**: 您说得对！我上一个回答确实漏掉了扩散模型部分。让我为您补全完整的扩散模型实现：

## 完整的扩散模型实现（DDPM）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import math
from tqdm import tqdm

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 1. 扩散过程调度器
class DiffusionScheduler:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        """
        扩散过程调度器
        T: 扩散步数
        beta_start: beta的起始值
        beta_end: beta的结束值
        """
        self.T = T
        
        # 线性beta调度
        self.betas = torch.linspace(beta_start, beta_end, T)
        
        # 计算alpha相关参数
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 用于计算sqrt(alpha_cumprod)等
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # 用于反向过程的方差计算
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
        # 移到设备
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
    
    def add_noise(self, x_start, t, noise=None):
        """
        前向扩散过程：向图像添加噪声
        x_start: 原始图像
        t: 时间步（批量）
        noise: 可选的自定义噪声
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # 获取对应时间步的参数
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # 计算加噪后的图像
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_noisy

# 2. 时间嵌入层
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings

# 3. 简单的U-Net块
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, t_emb):
        # 第一个卷积
        h = self.conv1(x)
        
        # 添加时间嵌入
        t_emb = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t_emb
        
        # 第二个卷积
        h = self.conv2(h)
        return h

# 4. 简化的U-Net模型
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, time_emb_dim=128):
        super().__init__()
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(inplace=True)
        )
        
        # 下采样路径
        self.down1 = UNetBlock(in_channels, 64, time_emb_dim)
        self.down2 = UNetBlock(64, 128, time_emb_dim)
        self.down3 = UNetBlock(128, 256, time_emb_dim)
        self.down4 = UNetBlock(256, 512, time_emb_dim)
        
        # 瓶颈层
        self.bottleneck = UNetBlock(512, 512, time_emb_dim)
        
        # 上采样路径
        self.up1 = UNetBlock(1024, 256, time_emb_dim)  # 512 + 512
        self.up2 = UNetBlock(512, 128, time_emb_dim)   # 256 + 256
        self.up3 = UNetBlock(256, 64, time_emb_dim)    # 128 + 128
        self.up4 = UNetBlock(128, 64, time_emb_dim)    # 64 + 64
        
        # 输出层
        self.out = nn.Conv2d(64, in_channels, 1)
        
        # 下采样池化
        self.downsample = nn.MaxPool2d(2)
        
        # 上采样
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x, t):
        # 时间嵌入
        t_emb = self.time_mlp(t)
        
        # 下采样路径
        d1 = self.down1(x, t_emb)
        d2 = self.down2(self.downsample(d1), t_emb)
        d3 = self.down3(self.downsample(d2), t_emb)
        d4 = self.down4(self.downsample(d3), t_emb)
        
        # 瓶颈层
        bottleneck = self.bottleneck(self.downsample(d4), t_emb)
        
        # 上采样路径（带跳跃连接）
        u1 = self.up1(torch.cat([self.upsample(bottleneck), d4], dim=1), t_emb)
        u2 = self.up2(torch.cat([self.upsample(u1), d3], dim=1), t_emb)
        u3 = self.up3(torch.cat([self.upsample(u2), d2], dim=1), t_emb)
        u4 = self.up4(torch.cat([self.upsample(u3), d1], dim=1), t_emb)
        
        return self.out(u4)

# 5. 扩散模型训练类
class DiffusionModel:
    def __init__(self, img_channels=3, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.img_channels = img_channels
        
        # 初始化调度器
        self.scheduler = DiffusionScheduler(T, beta_start, beta_end)
        
        # 初始化U-Net
        self.model = SimpleUNet(img_channels)
        
    def train_step(self, x_start, optimizer):
        """
        单步训练
        """
        batch_size = x_start.shape[0]
        
        # 随机采样时间步
        t = torch.randint(0, self.T, (batch_size,), device=device).long()
        
        # 采样噪声
        noise = torch.randn_like(x_start)
        
        # 前向扩散：添加噪声
        x_noisy = self.scheduler.add_noise(x_start, t, noise)
        
        # 预测噪声
        predicted_noise = self.model(x_noisy, t)
        
        # 计算损失
        loss = F.mse_loss(predicted_noise, noise)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def sample(self, num_samples=16, image_size=64):
        """
        从纯噪声开始采样生成图像
        """
        self.model.eval()
        
        # 从纯噪声开始
        x = torch.randn((num_samples, self.img_channels, image_size, image_size), device=device)
        
        # 逐步去噪
        samples = []
        
        for t in tqdm(reversed(range(self.T)), desc='采样进度', total=self.T):
            # 创建时间步张量
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            # 预测噪声
            predicted_noise = self.model(x, t_batch)
            
            # 获取调度器参数
            alpha_t = self.scheduler.alphas[t]
            alpha_cumprod_t = self.scheduler.alphas_cumprod[t]
            beta_t = self.scheduler.betas[t]
            
            if t > 0:
                # 计算噪声方差
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(self.scheduler.betas[t])
            else:
                noise = 0
                sigma_t = 0
            
            # 去噪步骤
            x = 1 / torch.sqrt(alpha_t) * (
                x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
            ) + sigma_t * noise
            
            # 保存中间结果（每100步保存一次）
            if t % (self.T // 10) == 0:
                samples.append(x.cpu())
        
        self.model.train()
        return x.cpu(), samples

# 6. 训练函数
def train_diffusion_model(dataloader, img_channels=3, epochs=50, lr=1e-4):
    """
    训练扩散模型
    """
    # 初始化扩散模型
    diffusion = DiffusionModel(img_channels=img_channels, T=500)  # 减少步数以加速训练
    
    # 模型和优化器
    model = diffusion.model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 训练历史
    history = {'loss': []}
    
    # 训练循环
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_data in pbar:
            # 获取图像数据
            if isinstance(batch_data, tuple) and len(batch_data) == 2:
                imgs, _ = batch_data
            else:
                imgs = batch_data
            
            imgs = imgs.to(device)
            
            # 训练步骤
            loss = diffusion.train_step(imgs, optimizer)
            
            epoch_loss += loss
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({'loss': loss})
        
        # 计算平均损失
        avg_loss = epoch_loss / num_batches
        history['loss'].append(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # 每10个epoch采样一次
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}: 生成样本...")
            samples, sample_progress = diffusion.sample(num_samples=16, image_size=64)
            
            # 保存生成的图像
            from torchvision.utils import save_image
            save_image(samples, f'diffusion_samples_epoch_{epoch+1}.png', 
                      nrow=4, normalize=True)
            
            # 显示采样过程
            if len(sample_progress) > 0:
                fig, axes = plt.subplots(1, len(sample_progress), figsize=(15, 3))
                for i, img in enumerate(sample_progress):
                    if img_channels == 3:
                        axes[i].imshow(img[0].permute(1, 2, 0) * 0.5 + 0.5)
                    else:
                        axes[i].imshow(img[0].squeeze(), cmap='gray')
                    axes[i].set_title(f'Step {i*self.T//10}')
                    axes[i].axis('off')
                plt.savefig(f'diffusion_progress_epoch_{epoch+1}.png', dpi=150, bbox_inches='tight')
                plt.close()
    
    return diffusion, history

# 7. 数据加载函数
def load_anime_dataset():
    """
    加载动漫数据集
    """
    # 如果本地有动漫数据集，请修改这个路径
    anime_path = 'data/anime_faces/images'
    
    if os.path.exists(anime_path) and len(os.listdir(anime_path)) > 0:
        print("使用动漫数据集")
        
        # 自定义数据集类
        class AnimeDataset(Dataset):
            def __init__(self, root_dir, transform=None):
                self.root_dir = root_dir
                self.image_files = [f for f in os.listdir(root_dir) 
                                   if f.endswith(('.png', '.jpg', '.jpeg'))]
                self.transform = transform
            
            def __len__(self):
                return len(self.image_files)
            
            def __getitem__(self, idx):
                img_path = os.path.join(self.root_dir, self.image_files[idx])
                image = Image.open(img_path).convert('RGB')
                
                if self.transform:
                    image = self.transform(image)
                
                return image
        
        # 数据变换
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.CenterCrop((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        dataset = AnimeDataset(anime_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=True)
        
        return dataloader, 3  # RGB图像，3个通道
    
    else:
        # 如果动漫数据集不存在，使用CIFAR-10作为替代
        print("动漫数据集未找到，使用CIFAR-10作为演示")
        
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        dataset = datasets.CIFAR10(root='data/cifar10', train=True, download=True, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=True)
        
        return dataloader, 3

# 8. 主函数
def main():
    # 加载数据
    dataloader, img_channels = load_anime_dataset()
    
    print(f"数据批次形状示例: {next(iter(dataloader))[0].shape}")
    print(f"图像通道数: {img_channels}")
    
    # 训练扩散模型
    print("开始训练扩散模型...")
    diffusion, history = train_diffusion_model(dataloader, img_channels, epochs=30, lr=1e-4)
    
    # 绘制训练损失
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Diffusion Model Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('diffusion_training_loss.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 最终采样
    print("生成最终样本...")
    final_samples, _ = diffusion.sample(num_samples=16, image_size=64)
    
    # 显示最终样本
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if img_channels == 3:
            ax.imshow(final_samples[i].permute(1, 2, 0) * 0.5 + 0.5)
        else:
            ax.imshow(final_samples[i].squeeze(), cmap='gray')
        ax.axis('off')
        ax.set_title(f'Sample {i+1}')
    
    plt.suptitle('Diffusion Model Generated Samples', fontsize=16)
    plt.tight_layout()
    plt.savefig('diffusion_final_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 保存模型
    torch.save({
        'model_state_dict': diffusion.model.state_dict(),
        'scheduler_params': {
            'T': diffusion.T,
            'img_channels': diffusion.img_channels
        }
    }, 'diffusion_model.pth')
    
    print("模型已保存到 diffusion_model.pth")
    
    return diffusion, history

# 9. 模型对比函数
def compare_models(dcgan_generator, vae_model, diffusion_model):
    """
    对比三种模型的生成结果
    """
    # 生成样本
    with torch.no_grad():
        # DCGAN生成
        z_dcgan = torch.randn(4, 100).to(device)
        dcgan_samples = dcgan_generator(z_dcgan).cpu()
        
        # VAE生成
        z_vae = torch.randn(4, 20).to(device)
        vae_samples = vae_model.decode(z_vae).cpu()
        
        # 扩散模型生成
        diffusion_samples, _ = diffusion_model.sample(num_samples=4, image_size=64)
    
    # 创建对比图
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    
    # DCGAN样本
    for i in range(4):
        axes[0, i].imshow(dcgan_samples[i].permute(1, 2, 0) * 0.5 + 0.5)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('DCGAN', fontsize=14, fontweight='bold')
    
    # VAE样本
    for i in range(4):
        axes[1, i].imshow(vae_samples[i].permute(1, 2, 0) * 0.5 + 0.5)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('VAE', fontsize=14, fontweight='bold')
    
    # 扩散模型样本
    for i in range(4):
        axes[2, i].imshow(diffusion_samples[i].permute(1, 2, 0) * 0.5 + 0.5)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Diffusion', fontsize=14, fontweight='bold')
    
    plt.suptitle('模型生成质量对比', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('model_comparison_all.png', dpi=150, bbox_inches='tight')
    plt.show()

# 10. 分析总结
def generate_summary():
    """
    生成分析总结报告
    """
    print("="*60)
    print("生成式模型进阶实践 - 总结报告")
    print("="*60)
    print("\n1. 模型特点对比：")
    print("-"*40)
    
    print("\nA. DCGAN（深度卷积生成对抗网络）：")
    print("   - 对抗训练：生成器和判别器相互博弈")
    print("   - 优点：生成图像清晰，训练相对快速")
    print("   - 缺点：容易模式崩溃，训练不稳定")
    print("   - 数学原理：\(\min_G \max_D V(D,G) = \mathbb{E}_{x\sim p_{data}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]\)")
    
    print("\nB. VAE（变分自编码器）：")
    print("   - 变分推断：学习潜在空间的正态分布")
    print("   - 优点：训练稳定，有理论保障")
    print("   - 缺点：生成图像可能模糊")
    print("   - 数学原理：\(\mathcal{L}(\theta,\phi;x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))\)")
    
    print("\nC. 扩散模型（DDPM）：")
    print("   - 逐步去噪：前向扩散和反向去噪过程")
    print("   - 优点：生成质量高，训练稳定")
    print("   - 缺点：采样速度慢，计算成本高")
    print("   - 数学原理：前向：\(q(x_t|x_{t-1}) = \mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t I)\)")
    print("             反向：\(p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))\)")
    
    print("\n2. 实践建议：")
    print("-"*40)
    print("   - 快速原型：使用DCGAN或VAE")
    print("   - 高质量生成：使用扩散模型")
    print("   - 实时应用：考虑优化版GAN（如StyleGAN）")
    print("   - 研究探索：扩散模型变体（如DDIM加速采样）")
    
    print("\n3. 性能指标（定性）：")
    print("-"*40)
    print("   - 多样性：扩散模型 > VAE ≈ DCGAN")
    print("   - 生成质量：扩散模型 > DCGAN > VAE")
    print("   - 训练稳定性：VAE > 扩散模型 > DCGAN")
    print("   - 采样速度：DCGAN > VAE > 扩散模型")
    
    print("\n4. 学习收获：")
    print("-"*40)
    print("   ✓ 理解了GAN的对抗训练原理")
    print("   ✓ 掌握了VAE的变分推断思想")
    print("   ✓ 学会了扩散模型的去噪过程")
    print("   ✓ 实践了三种模型在动漫头像生成上的应用")
    
    print("\n5. 扩展学习方向：")
    print("-"*40)
    print("   - GAN变体：WGAN, StyleGAN, BigGAN")
    print("   - 扩散模型：DDIM, Latent Diffusion, Stable Diffusion")
    print("   - 评估指标：FID, IS, Precision/Recall")
    print("   - 应用领域：图像编辑，超分辨率，风格迁移")
    
    print("\n" + "="*60)

# 执行主程序
if __name__ == "__main__":
    # 注意：如果您已经训练了DCGAN和VAE，可以在这里加载它们
    # 这里我们假设您已经有训练好的DCGAN和VAE模型
    
    print("正在运行扩散模型训练...")
    
    try:
        # 运行扩散模型训练
        diffusion_model, diffusion_history = main()
        
        # 注意：为了执行compare_models函数，您需要先训练好DCGAN和VAE
        # 如果您已经训练了这些模型，可以在这里取消注释并执行比较
        
        # compare_models(dcgan_generator, vae_model, diffusion_model)
        
        # 生成总结报告
        generate_summary()
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        print("请确保已正确安装所有依赖库，并且有足够的GPU内存")
        print("如果遇到内存不足问题，可以尝试：")
        print("1. 减小批处理大小 (batch_size)")
        print("2. 减小图像尺寸 (image_size)")
        print("3. 减少扩散步数 (T)")
        print("4. 使用更小的模型架构")
```

## 扩散模型的关键点说明：

### 1. **前向扩散过程**（加噪）：
```python
# 数学公式：
# x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * ε
# 其中 ε ~ N(0, I)
```

### 2. **反向扩散过程**（去噪）：
```python
# 使用U-Net预测噪声 ε_θ
# x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_cumprod_t) * ε_θ) + σ_t * z
```

### 3. **训练目标**：
```python
# 最小化预测噪声和真实噪声的均方误差
# L = E_{t,x_0,ε}[||ε - ε_θ(x_t, t)||^2]
```

## 扩散模型训练技巧：

1. **时间步采样**：均匀采样时间步 t ∈ [0, T]
2. **噪声调度**：使用线性β调度或余弦调度
3. **U-Net设计**：加入时间嵌入，使用注意力机制
4. **采样加速**：可以使用DDIM等加速采样方法

## 运行建议：

1. **硬件要求**：建议使用GPU，扩散模型训练较慢
2. **参数调整**：
   - 减少T（扩散步数）可以加速训练但可能降低质量
   - 减小批处理大小以节省内存
   - 适当减少U-Net参数量

3. **训练时间**：
   - DCGAN: ~30分钟（30轮）
   - VAE: ~20分钟（30轮）
   - 扩散模型: ~2-4小时（30轮）

这个完整的实现涵盖了DCGAN、VAE和扩散模型，您可以根据需要调整超参数和训练轮数。扩散模型的训练确实需要更多时间和计算资源，但生成质量通常最好。

**hexiaoya**: 训练过程中出现错误: name 'self' is not defined

**Jupyternaut**: 

(AI response stopped by user)

**hexiaoya**: Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.4639933..2.4664571].
训练过程中出现错误: name 'self' is not defined

**Jupyternaut**: 您遇到了两个问题。让我为您修复这些错误：

## 问题1：图像显示警告
```python
Clipping input data to the valid range for imshow with RGB data
```
这是因为扩散模型生成的图像值超出了 [-1, 1] 的范围。

## 问题2：`self`未定义错误
```python
name 'self' is not defined
```
在训练函数中错误地使用了`self.T`，应该使用`diffusion.T`。

## 修复后的扩散模型实现：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import math
from tqdm import tqdm

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 1. 扩散过程调度器
class DiffusionScheduler:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        """
        扩散过程调度器
        T: 扩散步数
        beta_start: beta的起始值
        beta_end: beta的结束值
        """
        self.T = T
        
        # 线性beta调度
        self.betas = torch.linspace(beta_start, beta_end, T)
        
        # 计算alpha相关参数
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 用于计算sqrt(alpha_cumprod)等
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # 用于反向过程的方差计算
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
        # 移到设备
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
    
    def add_noise(self, x_start, t, noise=None):
        """
        前向扩散过程：向图像添加噪声
        x_start: 原始图像
        t: 时间步（批量）
        noise: 可选的自定义噪声
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # 获取对应时间步的参数
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # 计算加噪后的图像
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_noisy

# 2. 时间嵌入层
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings

# 3. 简单的U-Net块
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, t_emb):
        # 第一个卷积
        h = self.conv1(x)
        
        # 添加时间嵌入
        t_emb = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t_emb
        
        # 第二个卷积
        h = self.conv2(h)
        return h

# 4. 简化的U-Net模型
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, time_emb_dim=128):
        super().__init__()
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(inplace=True)
        )
        
        # 下采样路径
        self.down1 = UNetBlock(in_channels, 64, time_emb_dim)
        self.down2 = UNetBlock(64, 128, time_emb_dim)
        self.down3 = UNetBlock(128, 256, time_emb_dim)
        self.down4 = UNetBlock(256, 512, time_emb_dim)
        
        # 瓶颈层
        self.bottleneck = UNetBlock(512, 512, time_emb_dim)
        
        # 上采样路径
        self.up1 = UNetBlock(1024, 256, time_emb_dim)  # 512 + 512
        self.up2 = UNetBlock(512, 128, time_emb_dim)   # 256 + 256
        self.up3 = UNetBlock(256, 64, time_emb_dim)    # 128 + 128
        self.up4 = UNetBlock(128, 64, time_emb_dim)    # 64 + 64
        
        # 输出层
        self.out = nn.Conv2d(64, in_channels, 1)
        
        # 下采样池化
        self.downsample = nn.MaxPool2d(2)
        
        # 上采样
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x, t):
        # 时间嵌入
        t_emb = self.time_mlp(t)
        
        # 下采样路径
        d1 = self.down1(x, t_emb)
        d2 = self.down2(self.downsample(d1), t_emb)
        d3 = self.down3(self.downsample(d2), t_emb)
        d4 = self.down4(self.downsample(d3), t_emb)
        
        # 瓶颈层
        bottleneck = self.bottleneck(self.downsample(d4), t_emb)
        
        # 上采样路径（带跳跃连接）
        u1 = self.up1(torch.cat([self.upsample(bottleneck), d4], dim=1), t_emb)
        u2 = self.up2(torch.cat([self.upsample(u1), d3], dim=1), t_emb)
        u3 = self.up3(torch.cat([self.upsample(u2), d2], dim=1), t_emb)
        u4 = self.up4(torch.cat([self.upsample(u3), d1], dim=1), t_emb)
        
        return self.out(u4)

# 5. 扩散模型训练类
class DiffusionModel:
    def __init__(self, img_channels=3, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.img_channels = img_channels
        
        # 初始化调度器
        self.scheduler = DiffusionScheduler(T, beta_start, beta_end)
        
        # 初始化U-Net
        self.model = SimpleUNet(img_channels)
        
    def train_step(self, x_start, optimizer):
        """
        单步训练
        """
        batch_size = x_start.shape[0]
        
        # 随机采样时间步
        t = torch.randint(0, self.T, (batch_size,), device=device).long()
        
        # 采样噪声
        noise = torch.randn_like(x_start)
        
        # 前向扩散：添加噪声
        x_noisy = self.scheduler.add_noise(x_start, t, noise)
        
        # 预测噪声
        predicted_noise = self.model(x_noisy, t)
        
        # 计算损失
        loss = F.mse_loss(predicted_noise, noise)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def sample(self, num_samples=16, image_size=64):
        """
        从纯噪声开始采样生成图像
        """
        self.model.eval()
        
        # 从纯噪声开始
        x = torch.randn((num_samples, self.img_channels, image_size, image_size), device=device)
        
        # 逐步去噪
        samples = []
        
        for t in tqdm(reversed(range(self.T)), desc='采样进度', total=self.T):
            # 创建时间步张量
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            # 预测噪声
            predicted_noise = self.model(x, t_batch)
            
            # 获取调度器参数
            alpha_t = self.scheduler.alphas[t]
            alpha_cumprod_t = self.scheduler.alphas_cumprod[t]
            beta_t = self.scheduler.betas[t]
            
            if t > 0:
                # 计算噪声方差
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(self.scheduler.betas[t])
            else:
                noise = 0
                sigma_t = 0
            
            # 去噪步骤
            x = 1 / torch.sqrt(alpha_t) * (
                x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
            ) + sigma_t * noise
            
            # 保存中间结果（每100步保存一次）
            if t % (self.T // 10) == 0:
                samples.append(x.cpu())
        
        self.model.train()
        return x.cpu(), samples

# 6. 训练函数
def train_diffusion_model(dataloader, img_channels=3, epochs=50, lr=1e-4):
    """
    训练扩散模型
    """
    # 初始化扩散模型
    diffusion = DiffusionModel(img_channels=img_channels, T=200)  # 减少步数以加速训练
    
    # 模型和优化器
    model = diffusion.model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 训练历史
    history = {'loss': []}
    
    # 训练循环
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_data in pbar:
            # 获取图像数据
            if isinstance(batch_data, tuple) and len(batch_data) == 2:
                imgs, _ = batch_data
            else:
                imgs = batch_data
            
            imgs = imgs.to(device)
            
            # 训练步骤
            loss = diffusion.train_step(imgs, optimizer)
            
            epoch_loss += loss
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({'loss': loss})
        
        # 计算平均损失
        avg_loss = epoch_loss / num_batches
        history['loss'].append(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # 每5个epoch采样一次
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}: 生成样本...")
            samples, sample_progress = diffusion.sample(num_samples=16, image_size=64)
            
            # 修正1：裁剪图像到合法范围 [-1, 1]
            samples = torch.clamp(samples, -1, 1)
            
            # 保存生成的图像
            from torchvision.utils import save_image
            save_image(samples, f'diffusion_samples_epoch_{epoch+1}.png', 
                      nrow=4, normalize=True)
            
            # 显示采样过程
            if len(sample_progress) > 0:
                fig, axes = plt.subplots(1, len(sample_progress), figsize=(15, 3))
                for i, img in enumerate(sample_progress):
                    # 修正2：裁剪图像并确保值在 [0, 1] 范围内
                    img_clamped = torch.clamp(img, -1, 1)
                    if img_channels == 3:
                        # 从 [-1, 1] 转换到 [0, 1]
                        img_display = img_clamped[0].permute(1, 2, 0) * 0.5 + 0.5
                        axes[i].imshow(img_display.numpy())
                    else:
                        axes[i].imshow(img_clamped[0].squeeze(), cmap='gray')
                    
                    # 修正3：使用 diffusion.T 而不是 self.T
                    axes[i].set_title(f'Step {i*diffusion.T//10}')
                    axes[i].axis('off')
                plt.savefig(f'diffusion_progress_epoch_{epoch+1}.png', dpi=150, bbox_inches='tight')
                plt.close()
    
    return diffusion, history

# 7. 数据加载函数
def load_anime_dataset():
    """
    加载动漫数据集
    """
    # 如果本地有动漫数据集，请修改这个路径
    anime_path = 'data/anime_faces/images'
    
    if os.path.exists(anime_path) and len(os.listdir(anime_path)) > 0:
        print("使用动漫数据集")
        
        # 自定义数据集类
        class AnimeDataset(Dataset):
            def __init__(self, root_dir, transform=None):
                self.root_dir = root_dir
                self.image_files = [f for f in os.listdir(root_dir) 
                                   if f.endswith(('.png', '.jpg', '.jpeg'))]
                self.transform = transform
            
            def __len__(self):
                return len(self.image_files)
            
            def __getitem__(self, idx):
                img_path = os.path.join(self.root_dir, self.image_files[idx])
                image = Image.open(img_path).convert('RGB')
                
                if self.transform:
                    image = self.transform(image)
                
                return image
        
        # 数据变换
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.CenterCrop((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        dataset = AnimeDataset(anime_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=True)
        
        return dataloader, 3  # RGB图像，3个通道
    
    else:
        # 如果动漫数据集不存在，使用CIFAR-10作为替代
        print("动漫数据集未找到，使用CIFAR-10作为演示")
        
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        dataset = datasets.CIFAR10(root='data/cifar10', train=True, download=True, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=True)
        
        return dataloader, 3

# 8. 简化的训练主程序
def train_diffusion_simple():
    """
    简化的扩散模型训练程序
    """
    # 加载数据
    dataloader, img_channels = load_anime_dataset()
    
    print(f"数据批次形状示例: {next(iter(dataloader))[0].shape}")
    print(f"图像通道数: {img_channels}")
    
    # 训练扩散模型
    print("开始训练扩散模型...")
    
    # 使用较小的参数加快训练
    diffusion = DiffusionModel(img_channels=img_channels, T=100)  # 进一步减少步数
    model = diffusion.model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    
    history = {'loss': []}
    
    # 训练10个epoch
    epochs = 10
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_data in pbar:
            if isinstance(batch_data, tuple) and len(batch_data) == 2:
                imgs, _ = batch_data
            else:
                imgs = batch_data
            
            imgs = imgs.to(device)
            
            # 训练步骤
            batch_size = imgs.shape[0]
            t = torch.randint(0, diffusion.T, (batch_size,), device=device).long()
            noise = torch.randn_like(imgs)
            x_noisy = diffusion.scheduler.add_noise(imgs, t, noise)
            predicted_noise = model(x_noisy, t)
            loss = F.mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / num_batches
        history['loss'].append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # 最后一个epoch生成样本
        if epoch == epochs - 1:
            print("生成最终样本...")
            model.eval()
            with torch.no_grad():
                # 生成16个样本
                x = torch.randn((16, img_channels, 64, 64), device=device)
                
                for t in reversed(range(diffusion.T)):
                    t_batch = torch.full((16,), t, device=device, dtype=torch.long)
                    predicted_noise = model(x, t_batch)
                    
                    alpha_t = diffusion.scheduler.alphas[t]
                    alpha_cumprod_t = diffusion.scheduler.alphas_cumprod[t]
                    
                    if t > 0:
                        noise = torch.randn_like(x)
                        sigma_t = torch.sqrt(diffusion.scheduler.betas[t])
                    else:
                        noise = 0
                        sigma_t = 0
                    
                    x = 1 / torch.sqrt(alpha_t) * (
                        x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
                    ) + sigma_t * noise
                
                # 裁剪并保存
                samples = torch.clamp(x, -1, 1).cpu()
                from torchvision.utils import save_image
                save_image(samples, 'diffusion_final_samples.png', nrow=4, normalize=True)
                
                # 显示样本
                fig, axes = plt.subplots(4, 4, figsize=(10, 10))
                for i in range(16):
                    row = i // 4
                    col = i % 4
                    img_display = samples[i].permute(1, 2, 0) * 0.5 + 0.5
                    axes[row, col].imshow(img_display.numpy())
                    axes[row, col].axis('off')
                
                plt.suptitle('扩散模型生成样本', fontsize=16)
                plt.tight_layout()
                plt.savefig('diffusion_samples_display.png', dpi=150, bbox_inches='tight')
                plt.show()
            
            model.train()
    
    # 绘制损失曲线
    plt.figure(figsize=(8, 5))
    plt.plot(history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('扩散模型训练损失')
    plt.grid(True, alpha=0.3)
    plt.savefig('diffusion_training_loss.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return diffusion, history

# 9. 运行简化的训练程序
if __name__ == "__main__":
    try:
        print("正在运行简化的扩散模型训练程序...")
        diffusion_model, diffusion_history = train_diffusion_simple()
        print("扩散模型训练完成！")
        print(f"最终损失: {diffusion_history['loss'][-1]:.4f}")
        
        # 生成总结报告
        print("\n" + "="*60)
        print("扩散模型训练总结")
        print("="*60)
        print("1. 模型架构: 简化的U-Net")
        print("2. 扩散步数: 100")
        print("3. 训练轮数: 10")
        print("4. 学习率: 2e-4")
        print("5. 图像尺寸: 64x64")
        print("\n训练结果:")
        print(f"- 起始损失: {diffusion_history['loss'][0]:.4f}")
        print(f"- 最终损失: {diffusion_history['loss'][-1]:.4f}")
        print(f"- 损失下降: {diffusion_history['loss'][0] - diffusion_history['loss'][-1]:.4f}")
        print("\n生成的图像已保存到:")
        print("- diffusion_final_samples.png (最终样本)")
        print("- diffusion_samples_display.png (显示图)")
        print("- diffusion_training_loss.png (损失曲线)")
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 提供调试建议
        print("\n调试建议:")
        print("1. 检查CUDA是否可用: torch.cuda.is_available()")
        print("2. 减小批处理大小 (batch_size)")
        print("3. 减小扩散步数 (T)")
        print("4. 使用更小的图像尺寸")
        print("5. 检查数据加载是否正常")
```

## 关键修复：

### 1. **图像裁剪**：
```python
# 在显示和保存前裁剪图像到合法范围
samples = torch.clamp(samples, -1, 1)

# 显示时从[-1, 1]转换到[0, 1]
img_display = img_clamped[0].permute(1, 2, 0) * 0.5 + 0.5
```

### 2. **修正self.T错误**：
```python
# 错误：使用 self.T
axes[i].set_title(f'Step {i*self.T//10}')

# 正确：使用 diffusion.T
axes[i].set_title(f'Step {i*diffusion.T//10}')
```

### 3. **简化训练流程**：
- 减少了扩散步数（T=100）
- 简化了采样过程
- 只训练10个epoch以快速获得结果

## 运行建议：

1. **首先测试数据加载**：
```python
# 测试数据加载是否正常
dataloader, img_channels = load_anime_dataset()
print(f"批量大小: {len(next(iter(dataloader))[0])}")
```

2. **逐步增加复杂度**：
```python
# 如果训练成功，可以逐渐增加：
# - 扩散步数：T=100 → T=200 → T=500
# - 训练轮数：epochs=10 → epochs=30
# - 批处理大小：batch_size=32 → batch_size=64
```

3. **内存不足时的调整**：
```python
# 如果出现CUDA内存不足：
# 1. 减小批处理大小：batch_size=32 → batch_size=16
# 2. 减小图像尺寸：64x64 → 32x32
# 3. 简化模型：减少U-Net的通道数
```

## 完整的三模型对比总结：

```python
def generate_complete_summary():
    """
    生成三种模型的完整对比总结
    """
    print("="*80)
    print("生成式模型进阶实践 - 完整对比报告")
    print("="*80)
    
    # 模型特性对比表格
    print("\n📊 模型特性对比：")
    print("-"*80)
    print(f"{'特性':<15} {'DCGAN':<20} {'VAE':<20} {'扩散模型':<20}")
    print("-"*80)
    print(f"{'训练方式':<15} {'对抗训练':<20} {'变分推断':<20} {'去噪训练':<20}")
    print(f"{'训练稳定性':<15} {'不稳定':<20} {'非常稳定':<20} {'稳定':<20}")
    print(f"{'生成速度':<15} {'非常快':<20} {'快':<20} {'慢':<20}")
    print(f"{'生成质量':<15} {'中等':<20} {'中等（可能模糊）':<20} {'高质量':<20}")
    print(f"{'多样性':<15} {'可能模式崩溃':<20} {'好':<20} {'非常好':<20}")
    print(f"{'数学基础':<15} {'博弈论':<20} {'变分推断':<20} {'马尔可夫链':<20}")
    print(f"{'计算资源':<15} {'中等':<20} {'中等':<20} {'高':<20}")
    print(f"{'主要应用':<15} {'快速生成':<20} {'数据压缩':<20} {'高质量生成':<20}")
    
    # 数学原理对比
    print("\n🧮 数学原理对比：")
    print("-"*80)
    print("1. DCGAN: \(\min_G \max_D V(D,G) = \mathbb{E}_{x\sim p_{data}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]\)")
    print("2. VAE: \(\mathcal{L}(\theta,\phi;x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))\)")
    print("3. Diffusion: \(q(x_t|x_{t-1}) = \mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t I)\)")
    print("            \(p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))\)")
    
    # 实践建议
    print("\n💡 实践建议：")
    print("-"*80)
    print("🎯 根据需求选择模型：")
    print("   • 快速原型 → DCGAN")
    print("   • 稳定训练 → VAE")
    print("   • 最高质量 → 扩散模型")
    print("   • 实时应用 → 优化版DCGAN或VAE")
    
    print("\n⚙️ 超参数调优：")
    print("   • DCGAN: 注意学习率平衡，添加梯度惩罚")
    print("   • VAE: 调整KL权重，优化重建损失")
    print("   • Diffusion: 调整扩散步数，优化噪声调度")
    
    print("\n🚀 性能优化：")
    print("   • 使用混合精度训练")
    print("   • 分布式训练加速")
    print("   • 模型剪枝和量化")
    
    # 学习路径
    print("\n📚 进阶学习路径：")
    print("-"*80)
    print("阶段1: DCGAN变体 → WGAN, StyleGAN, ProGAN")
    print("阶段2: VAE变体 → β-VAE, VQ-VAE, NVAE")
    print("阶段3: Diffusion变体 → DDIM, Stable Diffusion, Imagen")
    print("阶段4: 混合模型 → VQGAN+CLIP, Diffusion+GAN")
    
    # 项目建议
    print("\n🎨 项目应用建议：")
    print("-"*80)
    print("1. 动漫头像生成（已完成）")
    print("2. 图像超分辨率")
    print("3. 风格迁移")
    print("4. 图像修复")
    print("5. 文本到图像生成")
    
    print("\n" + "="*80)
    print("实践完成！您已掌握三种主流生成模型的核心原理和实现方法。")
    print("下一步可以尝试：")
    print("1. 在更大规模的数据集上训练")
    print("2. 实现模型融合（如Diffusion + GAN）")
    print("3. 部署模型到实际应用中")
    print("="*80)

# 运行总结
generate_complete_summary()
```

这个修复版本应该可以正常运行扩散模型。主要修复了：
1. 图像值范围问题（使用`torch.clamp`裁剪）
2. `self.T`未定义错误
3. 简化了训练流程以加快速度

您可以根据需要调整超参数，如增加训练轮数、扩散步数等，以获得更好的生成效果。

**hexiaoya**: /export /自定义.md

**Jupyternaut**: Sorry, an error occurred. Details below:

```
Traceback (most recent call last):
  File "/home/hexiaoya/miniconda3/envs/ai/lib/python3.10/site-packages/jupyter_ai/chat_handlers/base.py", line 226, in on_message
    await self.process_message(message)
  File "/home/hexiaoya/miniconda3/envs/ai/lib/python3.10/site-packages/jupyter_ai/chat_handlers/export.py", line 46, in process_message
    with open(chat_file, "w") as chat_history:
PermissionError: [Errno 13] Permission denied: '/自定义.md'

```

**hexiaoya**: /export 