import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            )
    
            self.bn1 = nn.BatchNorm2d(
                num_features=out_channels
            )
    
            self.conv2 = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            )
    
            self.bn2 = nn.BatchNorm2d(
                num_features=out_channels
            )
    
            if in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                    ),
                    nn.BatchNorm2d(
                        num_features=out_channels
                    )
                )
            else:
                self.shortcut = nn.Identity()

            self.activation = nn.LeakyReLU()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.activation(out)

        return out

class EncoderModule(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),

            ResidualBlock(32, 32),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),

            ResidualBlock(64, 64),

            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.LeakyReLU()
        )

        self.mu_head = nn.Linear(128, latent_dim)
        self.log_var_head = nn.Linear(128, latent_dim)

    def forward(self, imgs):
        features = self.features(imgs)

        mu = self.mu_head(features)
        log_var = self.log_var_head(features)

        return mu, log_var


class DecoderModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            ResidualBlock(64, 64),

            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            ),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            ResidualBlock(32, 32),

            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            ),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            ResidualBlock(16, 16),

            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z_in):
        z_out = self.net(z_in)
        return z_out