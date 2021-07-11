---
title:  "주피터 노트북 이미지 넣기"
excerpt: "주피터 노트북 이미지 넣기"
categories:
  - python memo
  
tags:
  - python memo
  
toc: true
toc_sticky: true
toc_label: "On this page"
    
last_modified_at: 2021-06-29T23:55:00-05:00
---

**마크다운으로 넣는걸 추천한다. 왜냐하면 코드 쉘 모드는 from PIL import Image와 겹쳐 오류가 발생 할 수 있다.**

현재 파일 경로에서 이미지가 1.jpg가 있을때 마크다운 모드에서 
```markdown
![](./1.jpg)
```
하면 된다.

일반 코드 쉘에서 넣는건 
```python
from IPython.display import Image
Image("./1.jpg")
```