import torch
import torch.nn as nn


class GeneratorNet(nn.Module):
    def __init__(
            self,
            noise_channel: int = 128
    ):
        super().__init__()
        n_channel = 64
        self.noise_channel = noise_channel
        self.__net = nn.Sequential(
            nn.ConvTranspose2d(noise_channel, n_channel*8, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(n_channel*8),
            nn.ReLU(),

            nn.ConvTranspose2d(n_channel*8, n_channel * 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(n_channel * 4),
            nn.ReLU(),

            nn.ConvTranspose2d(n_channel * 4, n_channel * 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(n_channel * 2),
            nn.ReLU(),

            nn.ConvTranspose2d(n_channel * 2, n_channel * 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(n_channel * 1),
            nn.ReLU(),

            nn.ConvTranspose2d(n_channel * 1, 3, kernel_size=(5, 5), stride=(3, 3), padding=(1, 1)),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):

        return self.__net(x.view(-1, self.noise_channel, 1, 1))


class DiscriminatorNet(nn.Module):
    def __init__(self):
        super().__init__()
        n_channel = 64
        self.__net = nn.Sequential(
            nn.Conv2d(3, n_channel, kernel_size=(5, 5), stride=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(n_channel),
            nn.LeakyReLU(0.2),
            # 32*32*n_channel

            nn.Conv2d(n_channel, n_channel * 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(n_channel * 2),
            nn.LeakyReLU(0.2),
            # 16*16*(n_channel*2)

            nn.Conv2d(n_channel * 2, n_channel * 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(n_channel * 4),
            nn.LeakyReLU(0.2),
            # 8*8*(n_channel*4)

            nn.Conv2d(n_channel * 4, n_channel * 8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(n_channel * 8),
            nn.LeakyReLU(0.2),
            # 4*4*(n_channel*8)

            nn.Conv2d(n_channel * 8, 1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0)),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        out = self.__net(x)  # type:torch.Tensor
        return out.view(-1)


if __name__ == '__main__':
    noise = torch.rand(size=(128, 100, 1, 1))
    g_net = GeneratorNet(100)
    d_net = DiscriminatorNet()
    fake_x = g_net(noise)
    fake_y = d_net(fake_x)
    print(fake_x.shape)
    print(fake_y.shape)
