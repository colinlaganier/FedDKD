import torch
from models import lenet, resnet, mobilenet, vgg

class Models:

    @staticmethod
    def Lenet5():
        return lenet.Lenet5

    #############
    @staticmethod
    def ResNet32():
        return resnet.ResNet32

    #############
    @staticmethod
    def ResNet18():
        return resnet.ResNet18

    #############
    @staticmethod
    def MobileNetV3():
        return mobilenet.MobileNetV3

    #############
    @staticmethod
    def VGG():
        return vgg.VGG19
    
    available = {"lenet5" : Lenet5, 
                 "resnet32" : ResNet32,
                 "resnet18" : ResNet18,
                 "mobilenetv3" : MobileNetV3,
                 "vgg" : VGG}    

