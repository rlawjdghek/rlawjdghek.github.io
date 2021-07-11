---
title:  "Linux sudo 권한 부여"
excerpt: "Linux sudo 권한 부여"
categories:
  - Linux
  
tags:
  - Linux
  
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-07-04T14:48:00-05:00
---

```bash
sudo vi /etc/sudoers
```
들어가서 asdasd 계정을 sudo 로 하고싶다 하면

asdasd ALL=(ALL:ALL) ALL 추가하면 끝

만약에 이거 잘못 입력해서 sudo 명령어 자체가 안먹힐 수도 있는데 전에 구글링 하니까 해결해서 다음에 이런일 생기면 그 때 구글링으로 해결하면서 작성하자