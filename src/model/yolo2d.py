import torch


class Conv2dBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=(1, 3), s=(1, 1), p=(0, 1), g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03)
        self.act = activation

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Residual2d(torch.nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = Conv2dBlock(ch, ch, torch.nn.SiLU(), k=(1, 3), p=(0, 1))
        self.conv2 = Conv2dBlock(ch, ch, torch.nn.SiLU(), k=(1, 3), p=(0, 1))

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class EEG2DHead(torch.nn.Module):
    def __init__(self, in_ch, S=200, num_classes=3):
        super().__init__()
        self.S = int(S)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, self.S))
        self.proj = torch.nn.Conv2d(in_ch, 2 + num_classes, kernel_size=1)

    def forward(self, x):
        x = self.pool(x)
        x = self.proj(x)
        return x.squeeze(2).transpose(1, 2)


class EEG2DYOLO(torch.nn.Module):
    def __init__(self, in_channels=1, input_height=18, S=200, num_classes=3):
        super().__init__()
        self.input_height = int(input_height)
        self.stem = Conv2dBlock(
            in_channels,
            32,
            torch.nn.SiLU(),
            k=(self.input_height, 7),
            s=(1, 2),
            p=(0, 3),
        )
        self.stage1 = torch.nn.Sequential(
            Conv2dBlock(32, 64, torch.nn.SiLU(), k=(1, 5), s=(1, 2), p=(0, 2)),
            Residual2d(64),
        )
        self.stage2 = torch.nn.Sequential(
            Conv2dBlock(64, 128, torch.nn.SiLU(), k=(1, 5), s=(1, 2), p=(0, 2)),
            Residual2d(128),
        )
        self.stage3 = torch.nn.Sequential(
            Conv2dBlock(128, 192, torch.nn.SiLU(), k=(1, 5), s=(1, 2), p=(0, 2)),
            Residual2d(192),
        )
        self.stage4 = torch.nn.Sequential(
            Conv2dBlock(192, 256, torch.nn.SiLU(), k=(1, 3), s=(1, 1), p=(0, 1)),
            Residual2d(256),
        )
        self.head = EEG2DHead(256, S=S, num_classes=num_classes)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input [B, C, H, W] but received shape {tuple(x.shape)}")
        if x.shape[2] != self.input_height:
            raise ValueError(
                f"Expected EEG height {self.input_height}, got {x.shape[2]}. "
                "Make sure the dataset emits [B, 1, 18, T] for the 2D path."
            )

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


def yolo_2d_v11_n(in_channels: int = 1, input_height: int = 18, S: int = 200, num_classes: int = 3):
    return EEG2DYOLO(in_channels=in_channels, input_height=input_height, S=S, num_classes=num_classes)