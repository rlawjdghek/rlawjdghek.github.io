---
title:  "주피터 환경설정"
excerpt: "주피터 커널 추가, 삭제, config 설정, 테마적용"
categories:
  - configuration
  
tags:
  - configuration
  
toc: true
toc_sticky: true
toc_label: "On this page"
    
last_modified_at: 2021-06-28T10:48:00-05:00
---

 
\* 2021-03-26: 주피터 랩보다 주피터 노트북이 개인적으로 더 좋은 것 같다. extension이 활성화가 잘 되어있어서
주피터 랩 이상의 기능들이 내재되어있다. (아래에서 주로 쓰는 설정 예시 확인)

### 1. 설치
```bash
conda install jupyter notebook
```

### 2. nbextension 설치 & 설정
```bash
conda install -c conda-forge jupyter_contrib_nbextensions
```

\+ 원도우는 아래 명렁어도 추가로 해줘야 한다. 
```bash
jupyter contrib nbextension install
```

주피터 노트북 가장 오른쪽 탭 nbextention에 들어가서 4가지 기능 체크

2-1. AutoSaveTime: 세이트 시간을 초 단위로 설정할 수 있다.\
2-2. Collapsible Heading: 마크다운 셀을 기준으로 접을 수 있다.\
2-3. Codefolding: 코드 함수를 접을 수 있다. \
2-4. ScollDown: 출력이 길어질 때 자동으로 닫을 수 있다. 


### 3. config파일 설정 (컴퓨터 하나당 한번 만 하면 된다.)
```bash
jupyter notebook --generate-config
```
먼저 위의 명령어로 config파일 만들기.
```python
# config.py
c = get_config()
c.JupyterApp.config_file_name = "jupyter_notebook_config.py"
c.NotebookApp.allow_origin = "*"
c.NotebookApp.ip = "115.145.190.195"
c.NotebookApp.open_browser = False
c.NotebookApp.password = 'argon2:argn2idv'
c.NotebookApp.port = 8890
```

가상환경을 커널에 추가
```s
pip install ipykernel
python -m ipykernel install --user --name rlawjdghek --display-name "rlawjdghek"
```

\* 커널 삭제
```bash
jupyter kernelspec uninstall rlawjdghek
```

### 4. 테마 및 폰트 조절
```bash
pip install jupyterthemes
```

```bash
jt -l 
```
을 활용하여 어느 테마를 적용할 수 있는지 리스트 보여주기.

![jupytertheme command](/assets/images/2021-06-28_jupyter_config/jupytertheme_command.JPG)

내 주 설정은 
```bash
jt -t gruvboxd -N -kl -f roboto -fs 9 -nfs 9 -tfs 9 -ofs 9 -cellw 80% -lineh 130 -cursc r -cursw 1
```
 





