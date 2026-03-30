import torch.nn as nn
import torch
class ResLayer(nn.Module):
    def __init__(self, in_channel, out_channel,stride = 1):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.convNet  = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=3, stride=stride,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel,kernel_size=3,stride = 1,padding=1),
            nn.BatchNorm2d(out_channel))
        self.shortcut = nn.Identity()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, stride=stride, kernel_size=1),
                nn.BatchNorm2d(out_channel)
            )
    def forward(self,x):
        out = self.convNet(x)
        out += self.shortcut(x)
        return nn.functional.relu(out)


class FaceNet(nn.Module):
    def __init__(self, in_channel ,dim:int = 200):
        super().__init__()
        self.res1 = ResLayer(in_channel=in_channel, out_channel=16,stride=2)
        self.res2 = ResLayer(in_channel=16, out_channel=32, stride = 2)
        self.res3 = ResLayer(in_channel=32, out_channel=64)
        self.res4 = ResLayer(in_channel=64,out_channel=128, stride = 2)
        self.res5 = ResLayer(in_channel=128, out_channel=128, stride = 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()

        self.age_fc = nn.Sequential(
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128,1),
        )
        self.gender_fc = nn.Sequential(
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128,1),
        )
        # self.ethinicity_fc = nn.Sequential(
        #     nn.Linear(128,256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128,1),
        #     nn.ReLU()
        # )
    def forward(self,x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.pool(x)
        x = self.flat(x)
        
        return self.age_fc(x), self.gender_fc(x)#, self.ethinicity_fc(x)


if __name__ == "__main__":
    model = FaceNet(in_channel=3, dim=200)
    print(model(torch.rand(1,3,200,200)))