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
    
last_modified_at: 2021-06-29T23:48:00-05:00
---

ogg file들을 wav로 바꿀때 사용한 코드 

```python
from pydub import AudioSegment
from glob import glob
import os
from os.path import join as opj
from tqdm import tqdm



bc2022_train_ps = sorted(glob("./DATA/train_gene_wav/birdclef_2022/*/*"))
bc2023_train_ps = sorted(glob("./DATA/train_gene_wav/birdclef_2023/*/*"))
print(f"total : {len(bc2022_train_ps) + len(bc2023_train_ps)}")
ps = bc2022_train_ps + bc2023_train_ps
n_process = 20
unit = len(ps) // 20

def ogg2wav(ofn, to_path):
    x = AudioSegment.from_file(ofn)
    x.export(to_path, format='ogg') 

def main_worker(i):
    if i == n_process-1:
        target_ps = ps[unit*i:]
    else:
        target_ps = ps[unit*i:unit*(i+1)]
    print(f"process id : {i}, target ps : {len(target_ps)}")
    for p in tqdm(target_ps):
        to_path = p.replace("train_gene_wav", "train_gene_ogg")
        to_path = to_path.replace(".wav", ".ogg")
        to_dir = "/".join(to_path.split("/")[:-1])
        os.makedirs(to_dir, exist_ok=True)
        ogg2wav(p, to_path)
    
if __name__ == "__main__":
    import multiprocessing 

    procs = []
    for i in range(n_process):
        p = multiprocessing.Process(target=main_worker, args=(i,))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    print("Done")
    
```

