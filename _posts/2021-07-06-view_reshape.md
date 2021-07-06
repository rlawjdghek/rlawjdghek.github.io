---
title:  "pytorch model.train(), eval()"
excerpt: "pytorch model.train(), eval()"
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

기능은 같지만 한번도 다루지 않은 데이터같은 경우는 view로 해도 되는데 중간에 네트워크에서 다루다가 크기를 조정 할 
떄에는 reshape를 써야한다. 이유는 reshape는 contiguous한 메모리 배열을 가능하게 한다.

view 는 contiguous 한 텐서에 대해서만 적용가능하다. 

reshape은 그렇지 않아도 적용 가능하고, contiguous 하지 않은 텐서에 적용하고 반환하는 경우 바뀐 텐서를 먼제 copy한 뒤에 contiguous하게 만들어 준다. 

그래서 view 를 사용한 코드는 중간중간에 텐서를 contiguous하게 바꿔 주는 경우가 종종 있다. 메모리를 아끼려면 copy를 하지 않고
view와 contiguous 함수만 사용해서 짜면 좋지만 복잡해 질 수 있다. 