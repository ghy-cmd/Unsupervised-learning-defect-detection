import timm
import PIL.Image
import torch
from torchvision import transforms
import torchvision.transforms as transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FeatureExtractor:
    def __init__(self):
        self.features = {}

    def __call__(self, module, input, output):
        self.features[module] = output


if __name__ == "__main__":
    # transform_img = transforms.Compose([
    #     transforms.Resize(256),  # 256
    #     transforms.CenterCrop(224),  # 224
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    # ])
    # image = transform_img(PIL.Image.open('/home/guihaoyue_bishe/mvtec/bottle/train/good/014.png')).unsqueeze(0)
    # print(image.size())
    model = timm.create_model("swin_base_patch4_window7_224", pretrained=True)
    # print(model.named_children())
    # with torch.no_grad():
    #     features = model(image)
    # print(features.size())
    # 注册hook函数
    # 注册hook函数
    extractor = FeatureExtractor()
    model.layers.__dict__["_modules"]["0"].register_forward_hook(extractor)
    model.layers.__dict__["_modules"]["1"].register_forward_hook(extractor)
    model.layers.__dict__["_modules"]["2"].register_forward_hook(extractor)
    model.layers.__dict__["_modules"]["3"].register_forward_hook(extractor)

    # model.layer1.register_forward_hook(extractor)
    # model.layer2.register_forward_hook(extractor)
    # model.layer3.register_forward_hook(extractor)
    # model.layer4.register_forward_hook(extractor)

    # 获取输入图片
    input_tensor = torch.rand((1, 3, 224, 224))

    # 模型前向传播，并通过hook函数获取特征
    with torch.no_grad():
        model(input_tensor)

    # 提取所需的特征
    features = {}
    features['layer1'] = extractor.features[model.layers.__dict__["_modules"]["0"]]
    features['layer2'] = extractor.features[model.layers.__dict__["_modules"]["1"]]
    features['layer3'] = extractor.features[model.layers.__dict__["_modules"]["2"]]
    features['layer4'] = extractor.features[model.layers.__dict__["_modules"]["3"]]

    print(features['layer1'].size())
    print(features['layer2'].size())
    print(features['layer3'].size())
    print(features['layer4'].size())
    # (1,49,1024)
