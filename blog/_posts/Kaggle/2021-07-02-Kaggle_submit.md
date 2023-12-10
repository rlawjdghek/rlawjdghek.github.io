---
title:  "Kaggle Submit"
excerpt: "캐글 서브밋 하는법"
categories:
  - Kaggle
  
tags:
  - Kaggle
  
toc: true
toc_sticky: true
toc_label: "On this page"
    
last_modified_at: 2021-07-02T14:48:00-05:00
---

### submission.csv만 덮어 씌워서 제출하는 법 (HubMap 가장 쉬운 방법.)

#### 1. 대회 링크로 들어간다.
#### 2. new notebook을 만든 뒤, 추론 코드를 작성.
2-1. 예를 들어 HubMap 대회 같은 경우는 최종 제출 파일이 submission.csv에 예측 값을 적어서 내는 형식이다. 다라서 최종코드를 제출하는 방식인데, 
우리는 다른 방법을 썼다. 먼저 new notebook에 들어가면 오른쪽 데이터셋에 hubmap 데이터가 있고, 그 안에 sample_submission.csv가 있다.

2-2. 이 sample_submission.csv를 불러 온 다음, 우리가 미리 로컬에서 다 만들어둔 데이터를 그냥 집어넣는 방식이다.  (아래 이미지 참조)

2-3. 그러기 위해서는 먼저 로컬에서 만들어둔 엑셀 파일을 내 캐글 데이터셋에 넣어야 하는데 이걸 넣으려면 오른쪽 Add data에서 넣는다. 
주의 할 점은 엑셀을 업로드 하고 create 버튼을 누르면 바로 되는 것이 아니라 좀 기다려야 되는 것.

2-4. 이렇게 데이터를 넣었으면 코드에서 볼 수 있듯이 내가 넣은 데이터를 불러오고 sample_submission에 있는, 예측 해야할 
데이터의 이름에 우리가 예측한 값들을 넣는 형식이다. 그런 뒤, 예측 값이 채워진 sample_submission을 submission.csv로 만드는 코드로 끝. 

#### 3. save & 추론
이렇게 new notebook을 완성하면, 오른쪽 위의 save version을 누른다. 현재는 이 코드로 10번을 commit했기 때문에 version이 10이 뜬다.
save version을 누르면 save & Run all (commit)이 뜨는데 이걸 지정하고 save를 누르면 전체 코드가 위에서 부터 한번 싹 돈다.
그러면 우리는 지금 추론 코드가 그냥 기존 로컬에서 작업한 엑셀을 읽고 submission.csv를 쓰는 것이므로 submission.csv가 output파일로 생긴다. 
만약 맨 마지막 print가 잘 작동 한다면 save를 한 뒤에 code - Your work - jeongho_submit을 들어가 보면 출력 문장 맨 마지막에 success!!메세지와
함께 왼쪽을 보면 output이 있다. 즉, 이 추론 코드의 output은 submission.csv라는 말이다. 

### 4. submit
이제 대회 홈페이지로 가서 submit prediction을 누르면 내가 방금 올린 추론코드를 선택할 수 있게되는데 이걸 선택하면 이 
추론코드의 output을 선택 할 수 있게 된다. output 선택하고 추론하면 run 되면서 최종 public 점수가 뜬다. 

![](/assets/images/2021-07-02-kaggle_submit/submission.JPG)

### 안에서 완성된 모델만 넣고 추론하여 submission.csv 만들기 (Birdcall)

아마 이 경우면 이제 모든 서브미션을 다 할 수 있을 것 같다. (모델 훈련을 커널에서 하는것 제외하고는) 개요만 정리하자면,
1. 추론을 위해 사용할 모델을 add data로 올린다. 
2. 올린 모델 path로 모델을 불러오고 sample_submission.csv를 보고 이 포맷을 익혀 이렇게 만드려고 한다. 
이 때 데이터는 원본만 존재하므로 birdcall의 경우는 원본 소리 데이터를 이미지로 바꾸는 작업을 커널에서 한다. 
3. 추론 한 데이터를 submission.csv로 만든다. 이 때 제출 경로는 절대경로는 "/kaggle/working/submission.csv" 그냥 data.to_csv("submission.csv")로 하면 된다.  

자세하게 보자. 먼저 나는 이 대회를 할 때 외부 모델 패키지를 사용했으므로 pip install resnest와 pip install timm을 해야 한다. 
**그런데 보통 대회들은 인터넷이 사용할 수 없기 때문에 이 패키지를 add data로 추가하고 직접 path를 불러와야한다. 다행인 것은 우리는 이미 
훈련된 모델을 사용할 것이기 때문에 모델틀만 있으면 된다. 이 순서를 timm을 예로 정리하면**
1. add data에 timm을 검색하여 대표적인 것을 찾고 옆에 add버튼으로 추가한다.
2. 보통 찾은 것에 들어가보면 path 추가하는 것이 있는데 이걸 참고 하던가 아니면 아래 코드를 보면 된다. 

```python
import numpy as np
import libroas as lb
import soundfile as sf
import padnas as pd
import cv2
import refrom pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from tqdm.notebook import tqdm
import time
import sys
sys.path.append("../input/resnext50-fast-package/resnext-0.0.6b20200701/resnest/")
sys.path.appedn("../input/timm-pytorch-iamge-models/pytorch-iamge-models-master")

import timm
from resnest.torch import resnest50

data = pd.DataFrame(
    [(path.stem, *path.stem.split("_"), path) for path in Path(TEST_AUDIO_ROOT).glob("*.ogg")],
columns = ["filename", id]
)

df-_train = pd.read_csv("../input/birdclef-2021/train_metadata.csv")
LABEL_IDS = {label: label_id, for label_id, label in ...}
nets = [load_net(model_name, save_path) for model_name, save_path in zip(model_names, save_paths)]

preds = ...
sub = preds_as_df(data, preds)
sub.to_csv("submission.csv", index=False)
```

