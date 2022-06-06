import os
import struct
import numpy as np
from PIL import Image

path = './MNIST/raw'
kind = 'train'

labels_path = os.path.join(path,'%s-labels-idx1-ubyte' % kind)
images_path = os.path.join(path,'%s-images-idx3-ubyte' % kind)

with open(labels_path, 'rb') as lbpath:
    # 使用struct.unpack方法读取前两个数据，>代表高位在前，I代表32位整型。lbpath.read(8)表示一次从文件中读取8个字节
    # 这样读到的前两个数据分别是magic number和样本个数
    magic, n = struct.unpack('>II',lbpath.read(8))
    # 使用np.fromfile读取剩下的数据
    labels = np.fromfile(lbpath,dtype=np.uint8) # labels是60000 * 1的列向量
with open(images_path, 'rb') as imgpath:
    magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
    # images是60000*784的矩阵。每行一个图像。注意28^2=784
    images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)

select = []
ground_truth = []
lines = images.shape[0]

# 随机选取100张图片
for i in range(100):
    rand_num = np.random.randint(lines)
    select.append(images[rand_num])
    ground_truth.append(labels[rand_num])

show_seq = np.array(select)
temp1, temp2, temp3 = [], [], []

# 将100*784的矩阵重塑为280*280的矩阵
for i in range(10):
    for j in range(10):
        temp1.append(show_seq[i * 10 + j].reshape(28, 28))
    temp2.append(np.hstack(temp1))
    temp1.clear()
result = np.vstack(temp2)
ground_truth = np.array(ground_truth).reshape(10, 10)


result = 255 - result       # 图片反色
print('Ground truth:\n', ground_truth)      # 打印 ground-truth
im = Image.fromarray(result)    # 生成图片
im.show()   #显示图片