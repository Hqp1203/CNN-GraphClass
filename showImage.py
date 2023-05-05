import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 加载CIFAR-10数据集
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                 transform=transforms.ToTensor())
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
                                transform=transforms.ToTensor())

# 获取一张图片
image, label = train_dataset[1]

# 显示图片
image = np.transpose(image, (1, 2, 0))  # 将通道维度放到最后
plt.imshow(image)
plt.show()
