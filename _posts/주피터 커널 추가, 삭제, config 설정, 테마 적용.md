---
title:  "주피터 커널 추가, 삭제, config 설정, 테마적용"

categories:
  - 환경설정
  
tags:
  - 환경설정

last_modified_at: 2021-06-28T10:48:00-05:00
---


기본적으로 **내 환경에 설정되어 있는것.**

먼저 jupyter notebook --generate-config로 파일 만들기

2021-03-26 주피터 랩보다 주피터 노트북이 더 좋음 extension이 좋다. 또한 가장 큰 문제인 탭 많이 늘어나는건 크롬 그룹핑으로 해결 할 수 있다. 

먼저 설치하기. **pip으로 설치하면 아래에서 기능 추가할 떄 에러 뜬다. **

conda install jupyter notebook

주피터 노트북이 랩보다 좋은 점은 아무래도 오래 되다 보니 추가적으로 붙은 기능이 굉장히 많아졌다는 것이다. 이걸 활용하기 위해 아래 명령어를 친 뒤 설정을 해주면 된다.

conda install -c conda-forge jupyter\_contrib\_nbextensions

윈도우는 아래 명령어도 추가로 해줘야 한다. 

jupyter contrib nbextension install

그러고 주피터 노트북에 들어가서 가장 오른쪽 탭에 nbextension들어간다.

쓰면 좋은 기능들은

1\. 코드가 길어질 경우 켜는데 오래 걸리고 렉도 걸릴 수 있으므로 함수별, 셀 별 닫아 놓는 기능이 있다. 

먼저 맨 위에 disable ~~~ 체크 해제한 뒤

AutoSaveTime, Collapsible Heading, Codefolding,  ScrollDown 체크하면 된다. 나머지 설정은 아직은 건드리지 말기. 

~\# Configuration file for jupyter-notebook.  
c = get\_config()~  
~c.JupyterApp.config\_file\_name = "jupyter\_notebook\_config.py"  
c.LabApp.allow\_origin = '\*'  
c.LabApp.ip = "115.145.190.163"  
c.LabApp.open\_browser = False  
c.LabApp.password = 'argon2:$argon2id$v=19$m=10240,t=10,p=8$nqkJ+bJ6q5VmUG~

~17QM0ulw$N3TZ5pp9aLnv1te7naDQWA'  
c.LabApp.port = 8890~

~보통 시작 위치 바꾸는 건 안넣음 media접근이 안될 수 있음.~

jupyter notebook --generate-config 로 파일 생성

\# Configuration file for jupyter-notebook.

c = get\_config()

c.JupyterApp.config\_file\_name = "jupyter\_notebook\_config.py"

c.NotebookApp.allow\_origin = "\*"

c.NotebookApp.ip = "115.145.190.195"

c.NotebookApp.open\_browser = False

c.NotebookApp.password = 'argon2:$argon2id$v=19$m=10240,t=10,p=8$nqkJ+bJ6q5VmUG17QM0ulw$N3TZ5pp9aLnv1te7naDQWA'

c.NotebookApp.port = 8890

커널 추가

1\. pip install ipykernel

2\. python -m ipykernel install --user --name <가상환경 이름> --display-name "이름" 

ex) python -m ipykernel install --user --name rlawjdghek --display-name "rlawjdghek"

커널 삭제

jupyter kernelspec uninstall rlawjdghek

### **테마 및 폰트 크기 조절 **

**먼저**

**pip install jupyterthemes를 설치한다. **

**jt -l 로 어느 테마를 적용할 수 있는지 리스트 보여주기.**

[##_Image|kage@dglnV5/btq1aRVFDVT/jdHxoedSkX1S8220s4VT21/img.png|alignCenter|data-origin-width="901" data-origin-height="891" data-ke-mobilestyle="widthContent"|||_##]

내 주 설정은

**jt -t gruvboxd -N -kl -f roboto -fs 11 -nfs 11 -tfs 11 -ofs 11 -cellw 80% -lineh 130 -cursc r -cursw 1 **