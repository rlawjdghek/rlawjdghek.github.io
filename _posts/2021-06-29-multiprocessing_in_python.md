---
title:  "Multiprocessing in Python"
excerpt: "Multiprocessing in Python"
categories:
  - python memo
  
tags:
  - python memo
  
toc: true
toc_sticky: true
toc_label: "On this page"
    
last_modified_at: 2021-06-29T10:48:00-05:00
---

파이썬의 multiprocessing에는 여러가지 패키지를 사용할 수 있다. 

### import multiprocess
이 패키지는 주피터에서는 사용할 수 없다. (DDP도 그렇듯 그냥 멀티로 할때에는 주피터 쓰지 말자.)

먼저 한개의 프로세스를 넣는 방법. 즉, 한 개의 프로세스만 코드상에서 멀티 프로세싱 하는 것으로 지정 해 두고 돌린다. 

```python
import os
import multiprocessing as mp

def func1(a):
    print("a: {}".format(a))
    print("parent: {}".format(os.getppid()))
    print("pid: {}".format(os.getpid()))

if __name__=="__main__":
    process = mp.Porcess(target=func1, args=(3,))
    process.start()
    process.join()
```
위와 같이 하면 그냥 func1함수를 3을 매개변수로 넣어 하나의 프로세스를 돌리는 것이다. 

```python
import os
import multiprocessing as mp
import time
semaphore = 0

def func1(a):
    global semaphore
    while semaphore:
        pass
    semaphore = 1
    print("a: {}".format(a))
    print("parent: {}".format(os.getppid()))
    print("pid: {}".format(os.getpid()))
    semaphore = 0
    time.sleep(2)
    return

if __name__=="__main__":
    pool = mp.Pool(processes=4,)
    pool.map(func1, [1,2,3,4,5,6])
    pool.close()
    pool.join()
```
위와 같이 하면 아래에 6개의 서로 같은 parent를 갖고 다른 pid를 갖는 프로세스가 생긴다. 대신 처리는 한번에 
하나씩 한다. 즉, a는 1,2,3,4,5,6으로 출력된다.

![](/assets/images/2021-06-29-multiprocessing_in_python/mp_code.jpg)


### import concurrent.futures import ThreadPoolExecutor

이것도 Pool과 마찬가지로 멀티코어로 병렬처리하는 방법.
```python
from concurrent.futures import ThreadPoolExecutor
from os.path import join as opj

def predict_on_video_set(videos, num_workers):
    def process_file(i):
        file_path = opj(test_dir, videos[i]) 
        y_pred = predict_on_video(file_path, batch_size = FRAMES_PER_VIDEO)
        print("prcess file y_pred: {}".format(y_pred))
        return y_pred
    
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        predictions = ex.map(process_file, range(len(videos)))
        print(predictions)
        return predictions
```
간단하다. def process_file을 내장함수로 저장해 두고, pool과 같이 map함수를 사용한다. 여기는 range함수를 썻는데 pool에서 처럼 [1,2,3,4,5...] 이런식으로 해도되고,
pool에서도 마찬가지로 range함수를 써도 된다.