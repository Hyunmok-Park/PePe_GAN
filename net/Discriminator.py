from torch import nn

class Discriminator(nn.Module):
    def __init__(self, img_size=128 * 128 * 3, hidden_dim=512):
        super().__init__()

        self.img_size = img_size
        self.hidden_dim = hidden_dim

        self.discriminator = nn.Sequential(
            nn.Linear(self.img_size * self.img_size * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

        # self.discriminator_1 = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3),
        #     nn.ReLU(),
        #     nn.MaxPool2d(3, 3),
        #     nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3),
        #     nn.ReLU(),
        #     nn.MaxPool2d(3, 3)
        # )
        #
        # self.discriminator_2 = nn.Sequential(
        #     nn.Linear(12 * 13 * 13, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, 1),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        # x = self.discriminator_1(x).view(x.size(0), -1)
        # x = self.discriminator_2(x)
        x = self.discriminator(x)
        return x