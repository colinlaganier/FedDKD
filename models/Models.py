import torch
from models import lenet, resnet, mobilenet, vgg, convnet
from torchvision.models import mobilenet_v3_small, shufflenet_v2_x0_5

class Models:

    @staticmethod
    def Lenet5():
        return lenet.Lenet5

    #############
    @staticmethod
    def ResNet34():
        return resnet.ResNet34

    #############
    @staticmethod
    def ResNet18():
        return resnet.ResNet18

    #############
    @staticmethod
    def MobileNetV3():
        # return mobilenet.MobileNetV3
        return mobilenet_v3_small
    
    #############
    @staticmethod
    def ShuffleNetV2():
        return shufflenet_v2_x0_5

    #############
    @staticmethod
    def VGG():
        return vgg.VGG19
    
    #############
    @staticmethod    
    def CNN_small():
        return convnet.CNNSmall
    
    #############
    @staticmethod
    def CNN_medium():
        return convnet.CNNMedium
    
    #############
    @staticmethod
    def CNN_large():
        return convnet.CNNLarge

    available = {"lenet5" : Lenet5, 
                    "resnet34" : ResNet34,
                    "resnet18" : ResNet18,
                    "mobilenetv3" : MobileNetV3,
                    "shufflenetv2" : ShuffleNetV2,
                    "vgg" : VGG,
                    "cnn_small" : CNN_small,
                    "cnn_medium" : CNN_medium,
                    "cnn_large" : CNN_large}    
    

