#!/usr/bin/env python3
"""
efficientnet_lstm_updated.py

Tuned EfficientNet-B0 + BiLSTM pipeline:
 - milder RandomResizedCrop (0.9–1.1)
 - removed ColorJitter
 - head LR=5e-4, EPOCHS_HEAD=8
 - fine-tune LR=3e-5, PATIENCE=4
 - gradient clipping in all loops
 - ReduceLROnPlateau scheduler
 - multiprocessing guard
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pydicom
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data._utils.collate import default_collate

# CONFIGURATION
DATA_DIR    = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
CSV_PATH    = os.path.join(DATA_DIR, "data/final_cropped_full.csv")
BATCH_SIZE  = 16
LR_HEAD     = 5e-4
LR_FINETUNE = 3e-5
EPOCHS_HEAD = 8
EPOCHS_FULL = 30
PATIENCE    = 4
WEIGHT_DECAY= 1e-4
DEVICE      = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# collate to skip None

def collate_fn(batch):
    batch=[b for b in batch if b is not None]
    return default_collate(batch)

# DATASET
class MultiViewDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        pivot=df.pivot_table(index="patient_id",columns="view",values="full_path_x",aggfunc="first").reset_index()
        pivot["path_low"]=df.groupby("patient_id")["pathology_x"].first().str.lower().str.strip().values
        pivot["label"]=pivot["path_low"].replace({"benign without callback":"benign"}).map({"benign":0,"malignant":1})
        pivot.dropna(subset=["label","CC","MLO"],inplace=True)
        self.df=pivot.reset_index(drop=True)
        self.data_dir=data_dir
        self.transform=transform
    def __len__(self): return len(self.df)
    def __getitem__(self,idx):
        row=self.df.iloc[idx]
        imgs=[]
        for view in ("CC","MLO"):
            path=os.path.join(self.data_dir,row[view])
            try:
                ds=pydicom.dcmread(path); arr=ds.pixel_array.astype(np.float32)
            except:
                return None
            if arr.max()>arr.min(): arr=(arr-arr.min())/(arr.max()-arr.min())
            else: arr=np.zeros_like(arr)
            img=Image.fromarray((arr*255).astype(np.uint8)).convert("RGB")
            if self.transform: img=self.transform(img)
            imgs.append(img)
        return torch.stack(imgs,0), torch.tensor(int(row["label"]),dtype=torch.long)

# MODEL
class EfficientNetLSTM(nn.Module):
    def __init__(self,hidden=512):
        super().__init__()
        self.backbone=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_feats=self.backbone.classifier[1].in_features
        self.backbone.classifier=nn.Identity()
        self.lstm=nn.LSTM(in_feats,hidden,batch_first=True,bidirectional=True)
        self.head=nn.Sequential(nn.Linear(hidden*2,256),nn.ReLU(True),nn.Dropout(0.5),nn.Linear(256,2))
    def forward(self,x):
        B,T,C,H,W=x.shape; x=x.view(B*T,C,H,W)
        f=self.backbone(x).view(B,T,-1)
        o,_=self.lstm(f); last=o[:,-1,:]
        return self.head(last)

# TRAIN/VAL

def main():
    print("Device:",DEVICE)
    df=pd.read_csv(CSV_PATH)
    pats=df["patient_id"].unique()
    tr,te=train_test_split(pats,test_size=0.15,random_state=42)
    tr,va=train_test_split(tr,test_size=0.1765,random_state=42)
    df_tr,df_va,df_te=df[df.patient_id.isin(tr)],df[df.patient_id.isin(va)],df[df.patient_id.isin(te)]
    # class weights
    lbls=df_tr.groupby("patient_id")["pathology_x"].first().str.lower().str.strip().replace({"benign without callback":"benign"}).map({"benign":0,"malignant":1}).dropna().astype(int).values
    cw=torch.tensor([1/np.bincount(lbls)[0],1/np.bincount(lbls)[1]],dtype=torch.float32,device=DEVICE)
    # transforms
    tr_tf=transforms.Compose([transforms.RandomResizedCrop(224,scale=(0.9,1.1)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    ev_tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    dl_tr=DataLoader(MultiViewDataset(df_tr,DATA_DIR,tr_tf),batch_size=BATCH_SIZE,shuffle=True,num_workers=4,collate_fn=collate_fn)
    dl_va=DataLoader(MultiViewDataset(df_va,DATA_DIR,ev_tf),batch_size=BATCH_SIZE,shuffle=False,num_workers=4,collate_fn=collate_fn)
    dl_te=DataLoader(MultiViewDataset(df_te,DATA_DIR,ev_tf),batch_size=BATCH_SIZE,shuffle=False,num_workers=4,collate_fn=collate_fn)

    model=EfficientNetLSTM().to(DEVICE)
    crit=nn.CrossEntropyLoss(weight=cw)
    # Stage1
    for p in model.backbone.parameters(): p.requires_grad=False
    opt=torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),lr=LR_HEAD,weight_decay=WEIGHT_DECAY)
    best=0
    for ep in range(1,EPOCHS_HEAD+1):
        model.train()
        for x,y in dl_tr:
            x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad(); out=model(x); loss=crit(out,y); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval(); ys,ps=[],[]
        with torch.no_grad():
            for x,y in dl_va: x=x.to(DEVICE); pr=torch.softmax(model(x),1)[:,1].cpu().numpy(); ps.extend(pr); ys.extend(y.numpy())
        auc=roc_auc_score(ys,ps); print(f"Stage1 Ep{ep} AUC{auc:.4f}")
        if auc>best: best=auc; torch.save(model.state_dict(),"s1.pth")
    # Stage2
    model.load_state_dict(torch.load("s1.pth"))
    for p in model.backbone.parameters(): p.requires_grad=True
    opt=torch.optim.Adam(model.parameters(),lr=LR_FINETUNE,weight_decay=WEIGHT_DECAY)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=0.5,patience=PATIENCE)
    best=0; cnt=0
    for ep in range(1,EPOCHS_FULL+1):
        model.train()
        for x,y in dl_tr:
            x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad(); out=model(x); loss=crit(out,y); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval(); ys,ps=[],[]
        with torch.no_grad():
            for x,y in dl_va: x=x.to(DEVICE); pr=torch.softmax(model(x),1)[:,1].cpu().numpy(); ps.extend(pr); ys.extend(y.numpy())
        auc=roc_auc_score(ys,ps); sched.step(auc); print(f"Stage2 Ep{ep} AUC{auc:.4f}")
        if auc>best: best=auc; cnt=0; torch.save(model.state_dict(),"best.pth")
        else: cnt+=1;
        if cnt>=PATIENCE: break
    # Test
    model.load_state_dict(torch.load("best.pth")); model.eval()
    ys,ps,pr=[],[],[]
    with torch.no_grad():
        for x,y in dl_te: x=x.to(DEVICE); pr_prob=torch.softmax(model(x),1)[:,1].cpu().numpy(); pre=(pr_prob>=0.5).astype(int); ys.extend(y.numpy()); ps.extend(pr_prob); pr.extend(pre)
    print(f"Test AUC:{roc_auc_score(ys,ps):.4f}",f"Acc:{accuracy_score(ys,pr):.4f}")

if __name__=="__main__":
    from multiprocessing import freeze_support; freeze_support(); main()
