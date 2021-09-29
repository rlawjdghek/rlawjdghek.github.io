---
title:  "[CVPR2021]Revisiting Knowledge Distillation: An Inheritance and Exploration Framework"
excerpt: "[CVPR2021]Revisiting Knowledge Distillation: An Inheritance and Exploration Framework"
categories:
  - Paper & Knowledge
  
tags:
  - Knowledge Distillation
 
published: True
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-09-21T14:33:00-05:00
---

### Abstract & Introduction
지금까지의 KD방법들은 모두 teacher network를 따라가는데에 급급하였다. 하지만 teacher는 optimal한 정보를 전달하지 않으므로 student의 입장에서
teacher가 주는 대로 학습하는 것은 분명 한계가 존재한다. 이 논문에서는 기존과 같이 teacher를 따라가는 특성 inheritance와 teacher를 벗어나 더 창의성을 추구하는 
exploration을 소개한다. 논문의 표현을 그대로 적자면,

"**The inheritance part** is learnd with a similarity loss to transfer the existing learnd knowledge from the teacher model to the student model,"

"while **the exploration part** is encouraged to learn representations different from the inherited ones with a dis-similarity loss."

이 두 방법을 결합함으로써, 저자들은 teacher의 정보보다 더 정확한 정보를 전달 할 수 있다고 주장한다. 
\\이것이 가능한 이유는 

