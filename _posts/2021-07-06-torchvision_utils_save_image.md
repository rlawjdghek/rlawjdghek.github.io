---
title:  "torchvision.utils.save_image"
excerpt: "torchvision.utils.save_image"
categories:
  - Pytorch & Tensorflow & Coding
  
tags:
  - Pytorch
  
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-07-06T12:48:00-05:00
---
```python
torchvision.utils.save_image(tensor, file_path, normalize=False, nrow=4)
``````
만약 텐서가 0에서 1사이값이면 normalize는 해도되고 안해도되고, 텐서가 0에서 255면 해야된다

tensor가 cuda에 있는 cpu든 상관없다
