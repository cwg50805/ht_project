# 資料處理套件
import cv2
import csv
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import tensorflow as tf
import torchvision
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# 讀取資料集標籤檔
csvfile = open('train.csv')
reader = csv.reader(csvfile)

# 讀取csv標籤
labels = []
count = 1
for line in reader:
    if count:
        count = 0
        continue
    tmp = [line[0],line[1]]
    # print tmp
    labels.append(tmp)

csvfile.close()
csvfile = open('dev.csv')
reader = csv.reader(csvfile)
dev_labels = []
count = 1
for line in reader:
    if count:
        count = 0
        continue
    tmp = [line[0],line[1]]
    # print tmp
    dev_labels.append(tmp)

csvfile.close()
picnum = len(labels)
print("芒果圖片數量: ",picnum)
print("dev數量: ",len(dev_labels))

X = []
y = []
dev_X = []
dev_y = []
# 轉換圖片的標籤
for i in range(len(labels)):
    if labels[i][1] == "A":
        labels[i][1] = 0
    if labels[i][1] == "B":
        labels[i][1] = 1
    if labels[i][1] == "C":
        labels[i][1] = 2
for i in range(len(dev_labels)):
    if dev_labels[i][1] == "A":
        dev_labels[i][1] = 0
    if dev_labels[i][1] == "B":
        dev_labels[i][1] = 1
    if dev_labels[i][1] == "C":
        dev_labels[i][1] = 2

cfg = { 'VGG11': [64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
        'VGG19': [64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M'] }

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        #print(cfg)
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(18432,3)
    
    def forward(self,x):
        out = self.features(x)
        out = out.view(out.size(0),-1)
        out = self.classifier(out)
        return out
    def _make_layers(self,cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU()]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

cnn = VGG('VGG19').cuda()
#print(cnn)
LR = 0.00005
BATCH_SIZE = 10
optimizer = torch.optim.Adam(cnn.parameters(),lr=LR)
EPOCH = 120
loss_func = nn.CrossEntropyLoss()
train_loader = Data.DataLoader(dataset=labels, batch_size=BATCH_SIZE,shuffle=True)
transform1 = transforms.Compose([
    transforms.Resize((200,200)),
    transforms.RandomRotation(45),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]
    )

def readimg(b_x,BATCH_SIZE):
    tup = [] 
    for i in range(0,BATCH_SIZE):
        img = Image.open("C1-P1_Train/" + b_x[i]).convert('RGB')
        img = transform1(img)
        tup.append(img)
    tup = torch.stack(tup,dim=0)
    return tup
def readimg2(b_x,BATCH_SIZE):
    tup = [] 
    for i in range(0,BATCH_SIZE):
        img = Image.open("C1-P1_Dev/" + b_x[i])
        img = transform1(img)
        tup.append(img)
    tup = torch.stack(tup,dim=0)
    return tup

nround = int(len(dev_labels) / BATCH_SIZE)
name = np.array(dev_labels)[:,0]
dev_y = np.array(dev_labels)[:,1]

image = cv2.imread("/tmp2/b06902053/mango/C1-P1_Train/00302.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#gray = image[:,:,1]
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#blurred = cv2.bilateralFilter(gray, 5, 300, 300)
#cv2.imwrite("blur_1.jpg", blurred)

#ret,th1 = cv2.threshold(blurred,127,255,cv2.THRESH_BINARY)
binaryIMG = cv2.Canny(blurred, 25, 40)
#cv2.imwrite("canny.jpg", binaryIMG)

binaryIMG = cv2.dilate(binaryIMG, None, iterations = 26)  # 默认(3x3)
cv2.imwrite("Dilate_0.jpg", binaryIMG)
binaryIMG = cv2.erode(binaryIMG, None, iterations= 30)
cv2.imwrite("Erode_0.jpg", binaryIMG)
# binaryIMG = cv2.erode(binaryIMG, None, iterations= 5)
# cv2.imwrite("Erode.jpg", binaryIMG)
# binaryIMG = cv2.dilate(binaryIMG, None, iterations = 2)  # 默认(3x3)
# cv2.imwrite("Dilate.jpg", binaryIMG)

(cnts, _) = cv2.findContours(binaryIMG.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

clone = image.copy()

#cv2.drawContours(clone, cnts, -1, (0, 255, 0), 2)

# ellipse = cv2.fitEllipse(cnts)
# img = cv2.ellipse(clone,ellipse,(0,255,0),2)
# cv2.imwrite("tmp.jpg", img)
max_area = 0
tmp_c = None
for c in cnts:
    area = cv2.contourArea(c)

    if area >= max_area:
        max_area = area
        tmp_c = c

    #cv2.drawContours(clone, [box], -1, (0, 255, 0), 2)
    # approx = cv2.approxPolyDP(c, 0.01 * peri, False) #取1%為\epsilon 值

    # (x, y, w, h) = cv2.boundingRect(approx)
    # cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # ellipse = cv2.fitEllipse(c)
    # cv2.ellipse(clone, ellipse, (0, 255, 0), 2)
    # cv2.putText(clone, "Vertex:"+str(len(approx)), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
box = cv2.minAreaRect(tmp_c)

ellipse = cv2.fitEllipse(tmp_c)

cv2.ellipse(clone, ellipse, (0, 255, 0), 2)

box = np.int0(cv2.boxPoints (box))  #–> int0會省略小數點後方的數字

cv2.drawContours(clone, [box], -1, (255, 0, 0), 2)
cv2.imwrite("tmp.jpg", clone)

# for c in cnts:        

#         mask = np.zeros(gray.shape, dtype="uint8″)  #依Contours圖形建立mask

#         cv2.drawContours(mask, [c], -1, 255, -1) #255        →白色, -1→塗滿

#         # show the images

#         cv2.imshow(“Image", image)

#         cv2.imshow(“Mask", mask)

# #將mask與原圖形作AND運算

#         cv2.imshow(“Image + Mask", cv2.bitwise_and(image, image, mask=mask))  

#         cv2.waitKey(0)


# for epoch in range(0,EPOCH):
#     print("epoch:",epoch)
#     for step,(b_x,b_y) in enumerate(train_loader): 
#         b_x = readimg(b_x, BATCH_SIZE).cuda()
#         b_y = b_y.cuda()
#         output = cnn(b_x)
#         loss = loss_func(output, b_y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     if epoch % 79 == 0 and epoch != 0:
#         filename = "vgg19" + str(epoch) + ".pkl"
#         torch.save(cnn.state_dict(), filename)
#     nsum = 0
#     npre = 0
#     for i in range(0,nround):
#         b_x = name[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
#         test_x = readimg2(b_x,BATCH_SIZE).cuda()
#         test_output = cnn(test_x)
#         pred_y = torch.max(test_output,1)[1].cuda().data.squeeze().cpu().numpy()
#         for j in range(0,BATCH_SIZE):
#             if str(pred_y[j]) == dev_y[i*BATCH_SIZE + j] :
#                 npre = npre + 1
#         nsum = nsum + BATCH_SIZE
#     print("accuracy:",npre/nsum)
