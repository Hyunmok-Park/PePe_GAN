from datetime import datetime
import os

import torch
from tqdm import tqdm
import numpy as np

from data_factory.data_loader import load_dataset
from model.gan import PePe_GAN
from torchvision.utils import save_image

def train(config, verbose):
    EXP_NAME = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    os.makedirs(f"result/{config['dataset']}/{EXP_NAME}")
    os.makedirs(f"result/{config['dataset']}/{EXP_NAME}/train_image")
    os.makedirs(f"result/{config['dataset']}/{EXP_NAME}/test_image")

    data_loader = load_dataset(config['data_path'], config['dataset'], config['batch_size'], config['img_size'])
    model = PePe_GAN(config).to(config['device'])
    g_loss_hist = []
    d_loss_hist = []
    min_g_loss = np.inf

    for epoch in tqdm(range(config['epoch']), desc='EPOCH'):
        model.train()
        gloss = []
        dloss = []
        for batch, (real_image, label) in enumerate(data_loader):
            for idx, img in enumerate(real_image):
                if epoch == 0 and verbose:
                    save_image(img, f'result/{config["dataset"]}/{EXP_NAME}/train_image/{epoch}_{batch}_{idx}.png')

            inputs = real_image.view(-1, config['img_size'] * config['img_size'] * config['num_channel']).to(config['device'])

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
                torch.save(model.state_dict(), f'result/{config["dataset"]}/{EXP_NAME}/checkpoint.pth')

        if epoch % 10 == 0:
            model.update_learning_rate()

        model.eval()
        img = model.generate_image()
        img = img.detach().view(config['num_channel'], config['img_size'], config['img_size'])
        save_image(img, f'result/{config["dataset"]}/{EXP_NAME}/test_image/{epoch}.png')

    torch.save(model.state_dict(), f'result/{config["dataset"]}/{EXP_NAME}/last_checkpoint.pth')

    # model.load_state_dict(torch.load(f'result/{config["dataset"]}/{EXP_NAME}/checkpoint.pth'))
    # for i in range(10):
    #     img = model.generate_image()
    #     img = img.detach().view(config['num_channel'], config['img_size'], config['img_size'])
    #     save_image(img, f'result/{config["dataset"]}/{EXP_NAME}/test_image/best_{i}.png')

    # model.load_state_dict(torch.load(f'result/{config["dataset"]}/{EXP_NAME}/last_checkpoint.pth'))
    # for i in range(10):
    #     img = model.generate_image()
    #     img = img.detach().view(config['num_channel'], config['img_size'], config['img_size'])
    #     save_image(img, f'result/{config["dataset"]}/{EXP_NAME}/test_image/last_{i}.png')

    import matplotlib.pyplot as plt
    f, ax = plt.subplots(1, 1, figsize=(20, 4))
    ax.plot(g_loss_hist, color='blue')
    ax.set_title(f"Loss : {mean_gloss} / {mean_dloss}")
    ax2 = ax.twinx()
    ax2.plot(d_loss_hist, color='red')
    plt.savefig(f'result/{config["dataset"]}/{EXP_NAME}/loss.png')

    from PIL import Image
    from IPython.display import Image as Img
    from IPython.display import display
    from glob import glob
    images = [Image.open(x) for x in sorted(glob(f'result/{config["dataset"]}/{EXP_NAME}/test_image/*.png'))]
    im = images[0]
    im.save(f'result/{config["dataset"]}/{EXP_NAME}/result.gif', save_all=True, append_images=images[1:], loop=0xff, duration=100)

if __name__ == '__main__':
    config = {}
    config['latent_dim'] = 32
    config['hidden_dim1'] = 64
    config['hidden_dim2'] = 128
    config['hidden_dim3'] = 256

    config['img_size'] = 128
    config['num_channel'] = 3

    config['data_path'] = 'dataset'
    config['dataset'] = 'pepe'
    config['num_sample'] = 1

    config['batch_size'] = 16
    config['learning_rate'] = 0.0002
    config['epoch'] = 500

    config['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train(config, True)