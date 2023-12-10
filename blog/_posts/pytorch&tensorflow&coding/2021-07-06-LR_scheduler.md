---
title:  "pytorch lr schuduler"
excerpt: "pytorch lr schuduler"
categories:
  - Pytorch & Tensorflow & Coding
  
tags:
  - Pytorch & Tensorflow & Coding
  
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-07-05T12:48:00-05:00
---

### StepLR
일정 스텝마다 학습률을 일정하게 감소시킨다.

step size : step()이 몇번 불릴때마다 감소시키는지 정함.
gamma : 얼마나 감소시킬지 정함

### MultiStepLR
StepLR과 달리 감소될 에폭을 정해준다. 예를 들어서 아래와 같이 쓰면, 100, 210에폭에서 학습률을 $\frac{1}{10}$ 씩 감소시킨다. 
```python
scheduler = torch.optim.lr_schduler.MultiStepLR(optimizer, milestones=[100, 210], gamma=0.1) 
```

### ExponentialLR
StepLR의 step size가 1인 함수이다. 잘 안쓸듯


### ReduceLROnPlateau
제일 많이 보이는 건데, 이건 특이하게 step 함수에 값을 넣는다. 이 값이 줄지 않는다면 patience를 기록해 두었다가 도달하면 감소시키는 식.

* mode : "min" or "max"를 넣는다. min이면 감소를 안할때 학습률을 조정하고, max이면 증가를 안할 때 학습률 조정. 기본값 : "min"
* factor : 위의 gamma와 같은 역할. 기본값 : 0.1
* patience : 계속 작아졌다고 할 때, 몇 epoch를 참을 것인지 결정. 기본값 : 10
* threshold : step함수에 들어가는 값이 이전에 업데이트때 들어간 값에 비해 얼만큼 차이나야 업데이트를 시킬 수 있다고 볼 수 있는지 결정. 기본값 : 1e-4
* min_lr : 감소되는 학습률의 최솟값
* eps : threshold는 step안에 들어가는 값을 기준으로 차이가 나야 되는 것이고, 이거는 학습률을 기준으로 이전 
학습률에서 다음 학습률의 차이가 eps보다 작으면 업데이트를 안한다. 예를 들어 이전 학습률이 1e-8이고 다음 업데이트 학습률이 1e-9이면 업데이트가 안된다. 기본값 1e-8.

### CyclicLR
step함수에 에폭을 넣어준다. 이 스케줄러는 에폭을 기준으로 학습률을 변화시킨다. cyclic 이므로 계속 최저점, 최고점 왔다갔다함.

* base_lr : 가장 낮은 학습률
* max_lr : 가장 높은 학습률
* step_size_up : 예를 들어 10을 입력하면 10번동안 base_lr에서 max_lr로 올라감. 기본값 2000
* step_size_down : 값을 입력 안하면 step_size_up과 같다. 보통 입력안하는게 맞을듯
* mode : 세가지 모드가 있다. "triangular", "triangular2", "exp_range". 아래 그림 참조. 기본값 "triangular"
* gamma : "exp_range" 모드 일때 gamma**cycilc_iteration으로 설정됨. 기본값 1.0
![](/assets/images/2021-07-06-lr_scheduler/1.JPG)

### CosineAnnealingLR
학습률이 코사인 함수를 따라서 eat_min까지 떨어졌다 다시 초기 학습률까지 올라온다. 아래 식 참조 :
![](/assets/images/2021-07-06-lr_scheduler/2.JPG)

* T_max: 위의 수식을 보면 주기가 $2T_{max}$임을 알 수 있다. 따라서 $T_{max}$는 주기의 반을 나타낸다.
* eta_min: 최대가 옵티마이저에서 지정한 학습률이고, 최소는 이 값이다. 결론적으로 전체적인 학습률은 이 값과 옵티마이저의 학습률 사이에서 움직인다. 기본값: 0 

### CosineAnnealingWarmRestat
CosineAnnelingLR과 식을 공유한다. 한가지 다른 점은 이건 올라가는게 코사인 함수가 아니라 내가 정해준 학습률부터 최솟값 eta_min까지 다시 내려간다.

* T_0: 이게 주기를 정한다. 위에서는 주기가 $T_{max}$의 2배였다면 이 스케줄러는 $T_0$이다.
* T_mult: 주기를 정하는 변수. 만약 2이면 다음 주기는 2배가 된다.
* eta_min: 최소 학습률. 기본값: 0


