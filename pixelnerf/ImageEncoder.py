import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision.models import resnet34


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.resnet = resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        feats1 = self.resnet.relu(x)

        feats2 = self.resnet.layer1(self.resnet.maxpool(feats1))
        feats3 = self.resnet.layer2(feats2)
        feats4 = self.resnet.layer3(feats3)

        latents = [feats1, feats2, feats3, feats4]
        latent_sz = latents[0].shape[-2:]
        for i in range(len(latents)):
            latents[i] = F.interpolate(
                latents[i], latent_sz, mode="bilinear", align_corners=True
            )

        latents = torch.cat(latents, dim=1)
        return latents


# 调整最后的特征通道数
class ModifiedPixelNeRF(nn.Module):
    def __init__(self):
        super(ModifiedPixelNeRF, self).__init__()
        # 添加卷积层以生成 128 特征通道
        self.feature_conv = nn.Conv2d(512, 128, kernel_size=1)

    def forward(self, feature, target_rgb):
        # 将特征降维至 128 通道
        feature_map = self.feature_conv(feature)
        # 将512*512降采样到64*64
        downs_feature = F.interpolate(feature_map, size=(64, 64), mode='bilinear', align_corners=False)
        # 将输出的目标位姿下的图像rgb下采样到64*64
        rgb_map = F.interpolate(target_rgb, size=(64, 64), mode='bilinear', align_corners=False)
        # 拼接 RGB 和特征通道
        output = torch.cat((rgb_map, downs_feature), dim=1)
        return output