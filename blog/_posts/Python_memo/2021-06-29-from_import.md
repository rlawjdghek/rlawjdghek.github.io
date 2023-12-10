---
title:  "상위 및 하위 폴더에 있을 때 from import"
excerpt: "상위 및 하위 폴더에 있을 때 from import"
categories:
  - python memo
  
tags:
  - python memo
  
toc: true
toc_sticky: true
toc_label: "On this page"
    
last_modified_at: 2021-06-29T10:48:00-05:00
---

main.py가 메인으로 돌릴 파일이면, 같은 폴더의 다른 파일 asdasd.py에서 os.getcwd()를 출력하라고 하면 당연히 ./main.py가 나온다. 
따라서 내가 model.py에서 모델의 경로를 ./save_model/model.pth를 주었을때 실제 이 경로는 main.py를 상대위치로 잡아 main.py와 같은 
폴더에 있는 폴더 save_model의 model.pth가 있는지 찾는다. 따라서 model.py가 다른 위치에 있다고 하더라도 이 경우에는 model.py가 위치한
폴더에 save_model을 만드는 것이 아니라는 것을 염두하자.

반면, 

folder\
----folder1\
--------asdasd.py\
--------qweqwe.py\  
----folder2\
--------zxczxc.py

로 있을때, qweqwe.py는 from asdasd import cls_asd가 있고, zxczxc.py에서 qweqwe.py를 import 하여 asdasd.py를 갖고올때 당연히 안된다. 
왜냐하면 현재 zxczxc.py가 위치한 folder2에는 asdasd가 없기 때문에 from asdasd에서 에러발생. 이러한 경우에는 qweqwe.py에서 from .asdasd import 
cls_asd라고 해야 사앧경로로 적용하여 qweqwe.py가 위치한 folder1에서 asdasd.py를 import 할 수 있다. 그래서 항상 from .asdasd 이런식으로 상대경로로 지정해야 된다. 

**하지만 상대경로로 지정할 겨우 주의해야 할 점이 있다. 상대경로로 지정한 뒤 그 파이썬 파일을 직접 실행하면 "__main__ is not a package" 라는 에러메시지가 뜬다. 
from . 에서 .은 main.이라는 말과 같은데 main.py에서 정의를 해주지 않았으므로 main이 패키지가 아니라고 하는 것이다. 상대경로를 사용한 파일이 다른 파일에 임포트 
되어 사용된다면 상대경로는 에러가 뜨지 않는다.**

**먄약 위의 파일 구조에서 zxczxc.py에서 asdasd.py를 import 하려면 뒤로 가는 방법이 없기 때문에
from ..folder1 import asdasd 이런건 안된다. 잘못되게 역으로 올라가므로 오류가 뜬다.  따라서 루트에서부터 출발하기 위해 항상 루트에서 시작하는 것을 잊지 말자.
root 디렉토리인 folder를 이용해야 된다. 따라서 folder의 절대 경로가 D:/rlawjdghek/test/folder이면 
sys.path.append("D:/rlawjdghek/test/folder")를 하면 folder에서 시작할 수 있다고 생각하면 된다. 따라서 이명령어를 치고 from folder1 import asdasd 하면 된다.**