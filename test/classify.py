from PIL import Image
import matplotlib.pyplot as plt

img = Image.open(
    '/home/guihaoyue_bishe/data/data_detection/AITEX/ground_truth/Broken_end/0001_002_00_13_mask.png').convert(
    'L')  # 转换为灰度图像
# print(list(img.getdata()))
# print(img.getpixel((0, 0)))
threshold_value = 128  # 阈值
img = img.point(lambda x: 0 if x < threshold_value else 255, '1')  # 二值化
print(list(img.getdata()))
print(img.getpixel((0, 0)))
