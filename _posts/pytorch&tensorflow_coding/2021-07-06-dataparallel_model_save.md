---
title:  "병렬처리 했을 경우 모델 저장"
excerpt: "병렬처리 했을 경우 모델 저장"
categories:
  - Pytorch & Tensorflow & Coding
  
tags:
  - Pytorch
  
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-07-05T12:48:00-05:00
---

2021-04-22 hubmap 모델 5개 앙상블 병렬처리 하다가 생긴 문제.

model을 저장하고 로드하는데에 key가 안맞는 오류가 생겨서 헤맸다

=> GPU 병렬처리하면 key들의 맨 앞에 module.이 추가로 붙는다 따라서 아래 코드로 이걸 다 없얘줘야 로드가 가능하다. 

```python
for key in list(state_dict.keys()):
    if "module." in key:
        state_dict[key.replace("module.", "")] = state_dict[key]
        del state_dict[key]
```

**del을 안하면 state_dict 는 OrderedDict 이므로 이전의 module.이 붙은 key들이 안사라지고 새로운 module.안붙은 것이 추가만 되므로 꼭 삭제 해 줘야한다.** 