import torch
from torch import nn
from torch import optim

from net import Discriminator
from net import Generator

class PePe_GAN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.img_size = config['img_size']
        self.channel = config['num_channel']
        self.hidden_dim1 = config['hidden_dim1']
        self.hidden_dim2 = config['hidden_dim2']
        self.hidden_dim3 = config['hidden_dim3']
        self.latent_dim = config['latent_dim']
        self.learning_rate = config['learning_rate']
        self.device = config['device']

        self.generator = Generator(img_size=self.img_size,
                                   channel=self.channel,
                                   hidden_dim1=self.hidden_dim1,
                                   hidden_dim2=self.hidden_dim2,
                                   hidden_dim3=self.hidden_dim3,
                                   latent_dim=self.latent_dim
                                   )

        self.discriminator = Discriminator(img_size=self.img_size,
                                           channel=self.channel,
                                           hidden_dim1=self.hidden_dim1,
                                           hidden_dim2=self.hidden_dim2,
                                           hidden_dim3=self.hidden_dim3
                                           )

        self.criterion = nn.BCELoss()
        self.generator_optimizer = optim.Adam(params=self.generator.parameters(), lr=self.learning_rate)
        self.discriminatorr_optimizer = optim.Adam(params=self.discriminator.parameters(), lr=self.learning_rate)

        self.g_scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.generator_optimizer,
                                                lr_lambda=lambda epoch: 0.99 ** epoch,
                                                last_epoch=-1,
                                                verbose=False)

        self.d_scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.discriminatorr_optimizer,
                                                     lr_lambda=lambda epoch: 0.99 ** epoch,
                                                     last_epoch=-1,
                                                     verbose=False)
    def update_learning_rate(self):
        self.g_scheduler.step()
        self.d_scheduler.step()


    def update_generator(self, batch_size):
        self.generator_optimizer.zero_grad()
        self.discriminatorr_optimizer.zero_grad()

        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_images = self.generator(z)
        all_true_labels = torch.ones([batch_size, 1], dtype=torch.float32).to(self.device)

        generator_loss = self.criterion(self.discriminator(fake_images), all_true_labels)
        generator_loss.backward()
        self.generator_optimizer.step()
        return generator_loss

    def update_discriminator(self, real_images, batch_size):
        self.generator_optimizer.zero_grad()
        self.discriminatorr_optimizer.zero_grad()

        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_images = self.generator(z)
        all_true_labels = torch.ones([real_images.size(0), 1], dtype=torch.float32).to(self.device)
        all_false_labels = torch.zeros([fake_images.size(0), 1], dtype=torch.float32).to(self.device)

        real_images_loss = self.criterion(self.discriminator(real_images), all_true_labels)
        fake_images_loss = self.criterion(self.discriminator(fake_images), all_false_labels)
        discriminator_loss = (real_images_loss + fake_images_loss) / 2
        discriminator_loss.backward()
        self.discriminatorr_optimizer.step()
        return discriminator_loss

    def generate_image(self):
        z = torch.randn(1, self.latent_dim).to(self.device)
        fake_images = self.generator(z)
        return fake_images