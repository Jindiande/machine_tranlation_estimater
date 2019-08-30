import numpy as np
import os
import sys
from model import *
from utils import tokenize
from utils import *
file1=""
file2=""
data_name=""
filaname_hter=""
def load_with_voca():
    data=[]
    src_vocab=load_vocab(file1,"src")
    tgt_vocab=load_vocab(file2,"tgt")
    fo=open(data_name,"r")
    i=0
    for line in fo:
        i+=1
        try:
            src,tgt=line.split("\t")
        except Exception:
            continue
            print("ignore this line")
        src_tokens=tokenize(src,UNIT)
        tgt_tokens=tokenize(tgt,UNIT)
        src_seq=[]
        tgt_seq=[]
        for w in src_tokens:
            if w not in src_vocab:
                src_seq.append(str(UNK_IDX))
            else:
                src_seq.append(str(tgt_vocab[w]))
        for w in tgt_tokens:
            if w not in src_vocab:
                tgt_seq.append(str(UNK_IDX))
            else:
                tgt_seq.append(str(tgt_vocab[w]))
        data.append((src_seq,tgt_seq))
    fo.close()
    return data

def load_hter(data):
    i=0
    dic={}
    data1=[]
    f1=open(filaname_hter,"r")
    HTER=[]
    length=[]
    line=f1.readline()
    for ite in data:
        line=line.strip()
        line=float(line)
        if(" ".join(ite[0])+"\t"+" ".join(ite[1])not in dic):
            dic[".join"(ite[0])+"\t"+" ".join(ite[1])+"\t"+str(i)]=line
        line=f1.readline()
        i+=1
    dic1=sorted(dic.items(),key=lambda x:-len(x[0].split("\t")[0].split(" ")))
    print(len(dic))
    for ite in dic1:
        HTER.append(str(ite[1]))
        src,tgt,_=ite[0].split("\t")
        data1.append((src,tgt))
    return data1,HTER

def save_data1(data):
    fo=open(data_name+".csv","w")
    for seq in data:
        fo.write(seq[0]+"\t"+seq[1]+"\n")
    fo.close()
def save_data(data):
    data.sort(key=lambda x: -len(x[0]))
    fo = open(data_name + ".csv", "w")
    for seq in data:
        fo.write(" ".join(seq[0]) + "\t" + " ".join(seq[0]) + "\n")
    fo.close()
def save_hter(data):
    fo = open(filaname_hter + "sorted", "w")
    for seq in  data:
        fo.write(seq+"\n")
    fo.close()
if __name__ == "__main__":
    data=load_with_voca()
    data,HTER=load_hter(data)
    save_hter(HTER)
    save_data1(data)






