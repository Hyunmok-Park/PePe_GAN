from torch import nn

class Generator(nn.Module):
    def __init__(self, img_size=128 * 128 * 3, hidden_dim=512, latent_dim=64):
        super().__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.hidden_dim = hidden_dim

        self.generator = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.img_size * self.img_size * 3),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.generator(z)
        return x