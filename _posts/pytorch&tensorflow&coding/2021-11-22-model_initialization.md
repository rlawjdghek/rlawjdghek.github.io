---
title:  "Model initialization 코드 메모"
excerpt: "Model initialization"
categories:
  - pytorch
  
tags:
  - model initialization
 
published: True
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-11-22T15:04:00-05:00
---


```python
from torch.nn import init
def init_weight(model, init_type="normal", init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__nam__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, 0.0, init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    model.apply(init_func)
    return model
```