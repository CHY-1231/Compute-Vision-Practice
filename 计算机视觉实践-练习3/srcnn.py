import mindspore.nn as nn

class SRCNN(nn.Cell):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2, pad_mode='pad')
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2, pad_mode='pad')
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2, pad_mode='pad')
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
