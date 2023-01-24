from datetime import datetime
import os

import torch
from tqdm import tqdm
import numpy as np

from data_factory.data_loader import ImageLoader
from model.gan import PePe_GAN
from torchvision.utils import save_image

def train(config):
    EXP_NAME = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    os.makedirs(f"result/{EXP_NAME}")
    os.makedirs(f"result/{EXP_NAME}/train_image")
    os.makedirs(f"result/{EXP_NAME}/test_image")

    data_loader = ImageLoader(config['data_path'], config['batch_size'], config['img_size'])
    model = PePe_GAN(config).to(config['device'])
    g_loss_hist = []
    d_loss_hist = []
    min_g_loss = np.inf
    min_d_loss = np.inf

    for epoch in tqdm(range(config['epoch']), desc='EPOCH'):
        gloss = []
        dloss = []
        for batch, (real_image, label) in enumerate(data_loader):
            for idx, img in enumerate(real_image):
                save_image(img, f'result/{EXP_NAME}/train_image/{epoch}_{batch}_{idx}.png')

            inputs = real_image.view(-1, config['img_size'] * config['img_size'] * 3).to(config['device'])

            g_loss = model.update_generator(config['batch_size'])
            d_loss = model.update_discriminator(inputs, config['batch_size'])
            g_loss_hist.append(g_loss.detach().item())
            d_loss_hist.append(d_loss.detach().item())
            gloss.append(g_loss.detach().item())
            dloss.append(d_loss.detach().item())
            mean_gloss = np.mean(gloss)
            mean_dloss = np.mean(dloss)

            if min_g_loss > mean_gloss:
                min_g_loss = mean_gloss
                torch.save(model.state_dict(), f'result/{EXP_NAME}/checkpoint.pth')
    torch.save(model.state_dict(), f'result/{EXP_NAME}/last_checkpoint.pth')

    model.load_state_dict(torch.load(f'result/{EXP_NAME}/checkpoint.pth'))
    for i in range(10):
        img = model.generate_image()
        img = img.detach().view(3, config['img_size'], config['img_size'])
        save_image(img, f'result/{EXP_NAME}/test_image/{i}.png')

    model.load_state_dict(torch.load(f'result/{EXP_NAME}/last_checkpoint.pth'))
    for i in range(10):
        img = model.generate_image()
        img = img.detach().view(3, config['img_size'], config['img_size'])
        save_image(img, f'result/{EXP_NAME}/test_image/last_{i}.png')

    import matplotlib.pyplot as plt
    f, ax = plt.subplots(1, 1, figsize = (20, 4))
    ax.plot(g_loss_hist, color='blue')
    ax2 = ax.twinx()
    ax2.plot(d_loss_hist, color='red')
    plt.savefig(f'result/{EXP_NAME}/loss.png')

if __name__ == '__main__':
    config = {}
    config['latent_dim'] = 32
    config['hidden_dim'] = 64
    config['img_size'] = 128

    config['data_path'] = 'dataset'

    config['batch_size'] = 4
    config['learning_rate'] = 0.001
    config['epoch'] = 100

    config['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train(config)