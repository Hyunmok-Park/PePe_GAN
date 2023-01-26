from torch import nn

class Generator(nn.Module):
    def __init__(self, img_size=128, channel=1, hidden_dim1=256, hidden_dim2=512, hidden_dim3=1024, latent_dim=64):
        super().__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.channel = channel
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = hidden_dim3

        self.generator = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim1),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim2, self.hidden_dim3),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim3, self.img_size * self.img_size * self.channel),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.generator(z)
        return x