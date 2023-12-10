---
title:  "Tensorboard"
excerpt: "Tensorboard"
categories:
  - Pytorch & Tensorflow & Coding
  
tags:
  - pytorch
  
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-07-06T14:48:00-05:00
---
### 요약
tensorboardX나, tensorboard_logger를 써도 되는데 언젠가 torch로 병합될 것 같아서 새로 torch에 있는 tensorboard를 사용하자.
#### 1. 설치
```bash
pip install tensorboard
```
#### 2. 텐서보드로 시각화 할 데이터 저장
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(<텐서보드로 시각화 할 데이터 저장할 곳>)
```

#### 3. 터미널에서 웹 키기
```bash
tensorboard --logdir <2번에서 데이터 저장한 경로>
```

#### 4. 인터넷 창에서 localhost 진입.
IP 입력


### 로컬 예제
아래 코드는 간단한 전체 훈련 코드이다.
```python
#### 라이브러리 로드 ####
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from einops.layers.torch import Rearrange
####
#### 텐서보드에 나타낼 데이터를 저장하는 폴더경로 지정. ####
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("./runs/cifar10_exp5")

#### 일반적인 모델 설정 ####
class args:
    n_epochs = 100000
    lr = 0.002
    iter_snapshot = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class simple_cnn(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 32, 3, 2, 1),  # 14
            nn.Conv2d(32, 64, 3, 1, 1),  # 14
            nn.Conv2d(64, 128, 3, 1, 1),  # 14
            nn.Conv2d(128, 256, 3, 2, 1), # 7
            nn.AvgPool2d(kernel_size=(7,7)),
            nn.Flatten(start_dim=1),
            nn.Linear(256, 10),
            nn.Softmax()
        )

transform = T.Compose([
    T.ToTensor()
])
train_dataset = torchvision.datasets.CIFAR10(root="/home/jeonghokim/", download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, drop_last=True, num_workers=8)

model = simple_cnn().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
####

#### 훈련 단계에서 로스와 정확도를 기록한다고 해보자. ####
for epoch in range(args.n_epochs):
    total_loss = 0
    correct = 0
    for iter_, (img, label) in enumerate(train_loader):
        img = img.to(device)
        label = label.to(device)

        output = model(img)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        _, pred = torch.max(output, 1)
        correct += (label==pred).sum().item()
        total_loss += loss.item()


    acc = "{:.2f}".format(correct / len(train_loader))
    print("acc: {}".format(acc))
    #### 여기에서 writer에게 값과 x축을 준다. 앞에는 기록될 그래프의 이름 ####
    writer.add_scalar("lossasd", total_loss / len(train_loader), epoch)  
    writer.add_scalar("accadsfkljhsadf", float(acc), epoch)
```
위의 코드를 실행하면 데이터들이 기록된다. 다음으로 웹에서 텐서보드를 실행하여 확인. 먼저 가상환경 이름을 rlawjdghek 라고 해보자.

```bash
conda activate rlawjdghek
tensorboard --logdir <위에서 경로. 이 예제에서는 /home/jeonghokim/runs/>
```
하면 아래와 같이 IP가 뜬다. 여기로 들어가보면 두개의 그래프가 생성된 것을 확인 할 수 있다.
![](/assets/images/2021-07-06-tensorboard/1.JPG)
왼쪽에 체크하는 것이 5개인 것은 실험을 5번 하면서 writer에 경로를 서로 다르게 주었기 때문.
![](/assets/images/2021-07-06-tensorboard/2.JPG)


### 서버 파이참 예제
서버는 우선 코드는 같다. 하지만 마지막에 텐서보드를 웹에 띄우는 과정이 약간 다르다. 먼저 ssh로

```bash
tensorboard --logdir <경로>
```
를 치면 아래와 같이 로컬과 똑같이 나온다.
![](/assets/images/2021-07-06-tensorboard/3.JPG)
이 때 그냥 로컬 크롬에서 키면 안되고, 로컬과 연결시켜야 한다. **로컬 아나콘다**에서

```bash
ssh -N -L localhost:16006:localhost:6006 serverID@serverIP 
```
를 치면 계속 실행이 되는데 이 때 크롬에서 **localhost:16006**을 실행하면 서버의 훈련 과정을 볼 수 있다.

### 서버 주피터
서버를 로컬에서 한번 더 연결하기가 귀찮을 때 서버의 주피터에서 볼 수 있는 방법도 있다. 서버의 텐서보드 데이터 root 경로와 주피터 파일이 같은 경로에 있다고 가정할 때, 셀에
```jupyterpython
%load_ext tensorboard
%tensorboard --logdir <경로>
```
를 쳐도 나온다. 하지만 업데이트 속도가 느리고 가시성이 좀 떨어져서 번거로워도 위의 방법을 사용하자.

