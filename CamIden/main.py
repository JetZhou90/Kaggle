from autoCnn import autoCnn
import os
import numpy as np
import csv

cls_name=[]
models={}
Sim={}
label={}
def train_model():
    for filename in os.listdir(r'train'):
        if filename.find('.'):
            cls_name.append(filename)
    for name in cls_name:
        save_path = 'model/'
        model_name = 'cnn_model_'+name+'.ckpt'
        filename='Data-grey/'+name+'.npy'
        print('Training'+filename+'.........')
        auto_model=autoCnn(bath_size=15,w=128,h=128,save_path=save_path,model_name=model_name)
        auto_model.load(filename)
        auto_model.model()
        auto_model.train()

def pre_model():

    data=np.load('TestData.npy')
    filename_list=np.load('FileName.npy')
    for filename in os.listdir(r'train'):
        if filename.find('.'):
            cls_name.append(filename)


    for name in cls_name:
        save_path = 'model/'
        model_name = 'cnn_model_' + name + '.ckpt'
        auto_model = autoCnn(bath_size=15, w=128, h=128, save_path=save_path, model_name=model_name)
        auto_model.model()
        models[name]=auto_model.pre(data)
        print('PreX from '+ name)
        Sim[name]=cal_dis(data,models[name])

    csv_file=open('submission.csv','w')
    csv_writer=csv.writer(csv_file)
    csv_writer.writerow(['fname','camera'])
    for index in range(data.shape[0]):
        temp={}
        for name in cls_name:
            temp[name]=Sim[name][index]
        sort_dis=sorted(temp.items(),key=lambda x:x[1],reverse=False)
        cls=sort_dis[0][0]
        filename=filename_list[index]
        label[filename]=cls
        csv_writer.writerow([filename,cls])






def cal_Cosdis(t_x,p_x):
    dotProduct=(t_x*p_x).sum(1)
    lengthT=(t_x**2).sum(1)
    lengthT=lengthT**0.5
    lengthP=(p_x**2).sum(1)
    lengthP=lengthP**0.5
    cosSim=dotProduct/(lengthP*lengthT)
    return cosSim

def cal_dis(t_x,p_x):
    diff=(t_x-p_x)**2
    diff=diff.sum(1)
    dis=diff**0.5
    return  dis


if __name__ == '__main__':
    # train_model()
    pre_model()