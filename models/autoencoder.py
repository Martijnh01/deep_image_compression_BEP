import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(
            self,
            img_size: int = 32,

    ):
        super().__init__()
        self.img_size = img_size
        self.latent_dim = 64
        self.num_input_channels = 3
        self.c_hid = 32

        self.encoder = nn.Sequential(
            nn.Conv2d(self.num_input_channels, self.c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            nn.ReLU(),
            nn.Conv2d(self.c_hid, self.c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.c_hid, 2 * self.c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
            nn.ReLU(),
            nn.Conv2d(2 * self.c_hid, 2 * self.c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * self.c_hid, 2 * self.c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            nn.ReLU(),
            nn.Conv2d(2 * self.c_hid, 2 * self.c_hid, kernel_size=3, padding=1, stride=1),
            nn.Flatten(),
            nn.Linear(2 * 16 * self.c_hid, self.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, 2 * self.c_hid, kernel_size=4, padding=0, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                2 * self.c_hid, 2 * self.c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 4x4 => 8x8
            nn.ReLU(),
            nn.Conv2d(2 * self.c_hid, 2 * self.c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * self.c_hid, self.c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 8x8 => 16x16
            nn.ReLU(),
            nn.Conv2d(self.c_hid, self.c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.c_hid, self.num_input_channels, kernel_size=3, output_padding=1, padding=1,
                               stride=2),  # 16x16 => 32x32
            nn.Tanh(),  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, img):
        # b, c, h, w = img.shape
        z = self.encoder(img)
        y = self.decoder(z.unsqueeze(-1).unsqueeze(-1))
        return y
