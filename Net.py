import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()

        #Encoder
        self.encoder_layer1 = nn.Sequential(
            # 512*512*3
            nn.Conv2d(3 , 64,4,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 256*256*64
        )
        self.encoder_layer2 = nn.Sequential(
            # 256*256*64
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128*128*128
        )

        self.encoder_layer3 = nn.Sequential(
            # 128*128*128
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 64*64*256
        )
        self.encoder_layer4 = nn.Sequential(
            # 64*64*256
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 32*32*512
        )
        self.encoder_layer5 = nn.Sequential(
            # 32*32*512
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 16*16*512

        )
        self.encoder_layer6 = nn.Sequential(
            # 16*16*512
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 8*8*512
        )
        self.encoder_layer7 = nn.Sequential(
            # 8*8*512
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 4*4*512
        )
        self.encoder_layer8 = nn.Sequential(
            # 4*4*512
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 2*2*512
        )
        self.encoder_layer9 = nn.Sequential(
            # 2*2*512
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 1*1*512
        )

        #Decoder
        self.decoder_layer9 = nn.Sequential(
            # 1*1*512
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
            # 2*2*512
        )
        self.decoder_layer8 = nn.Sequential(
            # 2*2*512
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
            # 4*4*512
        )
        self.decoder_layer7 = nn.Sequential(
            # 4*4*512
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
            # 8*8*512
        )
        self.decoder_layer6 = nn.Sequential(
            # 8*8*512
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
            # 16*16*512
        )
        self.decoder_layer5 = nn.Sequential(
            # 16*16*512
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
            # 32*32*512
        )
        self.decoder_layer4 = nn.Sequential(
            # 32*32*512
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
            # 64*64*256
        )
        self.decoder_layer3 = nn.Sequential(
            # 64*64*256
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
            # 128*128*128
        )
        self.decoder_layer2 = nn.Sequential(
            # 128*128*128
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
            # 256*256*64
        )
        self.decoder_layer1 = nn.Sequential(
            # 256*256*64
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(3),
            # 512*512*3
        )

        self.conv1 = nn.Conv2d(1024, 512,3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512,256,3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(256,128,3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(128,64,3,stride=1,padding=1)

    def forward(self,x):

        #Downsample
        x1 = self.encoder_layer1(x)
        x2 = self.encoder_layer2(x1)
        x3 = self.encoder_layer3(x2)
        x4 = self.encoder_layer4(x3)
        x5 = self.encoder_layer5(x4)
        x6 = self.encoder_layer6(x5)
        x7 = self.encoder_layer7(x6)
        x8 = self.encoder_layer8(x7)
        x9 = self.encoder_layer9(x8)

        #Upsample
        x10 = self.decoder_layer9(x9)
        x10 = self.conv2(x10)
        x8 = self.conv2(x8)
        x10 = torch.cat((x8, x10), dim=1)

        x11 = self.decoder_layer8(x10)
        x11 = self.conv2(x11)
        x7 = self.conv2(x7)
        x11 = torch.cat((x7, x11), dim=1)

        x12 = self.decoder_layer7(x11)
        x12 = self.conv2(x12)
        x6 = self.conv2(x6)
        x12 = torch.cat((x6, x12), dim=1)

        x13 = self.decoder_layer6(x12)
        x13 = self.conv2(x13)
        x5 = self.conv2(x5)
        x13 = torch.cat((x5, x13), dim=1)

        x14 = self.decoder_layer5(x13)
        x14 = self.conv2(x14)
        x4 = self.conv2(x4)
        x14 = torch.cat((x4, x14), dim=1)

        x15 = self.decoder_layer4(x14)
        x15 = self.conv3(x15)
        x3 = self.conv3(x3)
        x15 = torch.cat((x3, x15),dim=1)

        x16 = self.decoder_layer3(x15)
        x16 = self.conv4(x16)
        x2 = self.conv4(x2)
        x16 = torch.cat((x2,x16),dim=1)

        x17 = self.decoder_layer2(x16)
        x18 = self.decoder_layer1(x17)
        output = F.tanh(x18)
        return output














