---
title:  "collate_fn 정리"
excerpt: "collate_fn 정리"
categories:
  - Pytorch & Tensorflow & Coding
  
tags:
  - Pytorch
  
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-09-12T21:16:00-05:00
---
  
  
HAI하다가 pytorch transformer의 batch_first를 사용하기 위해 1.7.1에서 1.9.0으로 변경하면서 data loader에 더 규제가 강화되어 data loader에서 발생하는 오류를 메모. 
먼저 아래와 같이 dataset에서 
1. input: [batch_size x time_len x feature]  => 이상치 탐지를 위한 시계열 데이터. window size를 40이라 할 때 1~39번쨰까지를 반환 
2. label: [batch_size x time_len x feature]  => 새로운것을 예측하기 위해 label로 사용. window size를 40이라 할때 2~40번째까지를 반환
3. label_ts: [batch_size x time_len]  => 이게 string 형식이라 까다로움
4. attack: [batch_size x time_len] => 0또는 1값으로, validation이면, attack이라고 이상치인지 아닌지를 판단하는 최종 label
  
  
아래와 같은 HAIDataset과 collate_fn를 썼다고 쳐보자.
```python
class HAIDataset2(Dataset):  # label이 2번쨰부터 window_size번째까지
    def __init__(self, tss, df, window_size, stride, attacks=None):
        self.tss = np.asarray(tss)
        self.df = np.array(df, dtype=np.float32)
        self.window_size = window_size
        self.stride = stride
        self.idxs = []
        with tqdm(range(len(self.tss) - self.window_size + 1), total=len(self.tss) - self.window_size + 1, leave=False) as dataset_loop: # 시간이 연속된 것만 유요한 윈도우로 사용할 수 있다. 
            for L in dataset_loop:
                R = L + self.window_size - 1
                du_R = dateutil.parser.parse(self.tss[R])
                du_L = dateutil.parser.parse(self.tss[L])
                if du_R - du_L == timedelta(seconds=self.window_size - 1):
                    self.idxs.append(L)
        self.idxs = np.asarray(self.idxs, dtype=np.int32)[::self.stride]
        self.n_idxs = len(self.idxs)
        print("There is {} data instances".format(self.n_idxs))
        if attacks is not None:
            self.attacks = np.array(attacks, dtype=np.float32)
            self.is_attack = True
        else:
            self.is_attack = False
        
    def __len__(self):
        return self.n_idxs
    
    def __getitem__(self, idx):
        idx = self.idxs[idx]
        label_idx = idx + self.window_size - 1
        instance = {"attack": torch.from_numpy(self.attacks[idx+1:label_idx+1])} if self.is_attack else {}
        instance["label_ts"] = self.tss[idx+1:label_idx+1]
        instance["input"] = torch.from_numpy(self.df[idx:label_idx])
        instance["label"] = torch.from_numpy(self.df[idx+1:label_idx+1])
        return instance 

def HAIDataset2_collate_fn(datas):
    label_ts = np.stack([data["label_ts"] for data in datas])
    input_ = [data["input"] for data in datas]
    label = [data["label"] for data in datas]
    if "attack" in datas[0].keys():
        attack = [data["attack"] for data in datas]
        return {
        "input": torch.stack(input_).contiguous(),
        "label": torch.stack(label).contiguous(),
        "label_ts": label_ts,
        "attack": torch.stack(attack).contiguous()
        }
    return {
        "input": torch.stack(input_).contiguous(),
        "label": torch.stack(label).contiguous(),
        "label_ts": label_ts
    }
```
        
먼저 collate_fn이 없으면 원래의 default collate fn이 전부다 텐서로 묶기 때문에 중간에 numpy string을 변수로 갖는 label_ts가 처리되지 못한다.
**torch의 tensor는 항상 숫자여야 한다.**
따라서 collate_fn에서 반환하는 형식까지 만들어 준다. label_ts는 그냥 그대로 반환하고 나머지는 stack을 이용하여 배치화 해준다.