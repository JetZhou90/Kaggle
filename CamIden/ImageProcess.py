from PIL import Image
import numpy as np
import os
import csv
import pylab


i=0
w=128
h=128
cam_img=[] # image from cam
cam_label=[]   # labels
cls=10
# cls_name=['HTC-1-M7','iPhone-4s','iPhone-6','LG-Nexus-5x',
#           'Motorola-Droid-Maxx','Motorola-Nexus-6','Motorola-X',
#           'Samsung-Galaxy-Note3','Samsung-Galaxy-S4','Sony-NEX-7']
cls_name=[]

def one_hot(index,n):
    lb=np.zeros(n,dtype=np.int)
    lb[index]=1
    return lb

# for filename in os.listdir(r'train'):
#     if filename.find('.'):
#         cls_name.append(filename)

# for i in range(cls):
#     for filename in os.listdir(r'train/'+cls_name[i]):
#         if(filename!='Thumbs.db'):
#             basedir = 'train/'+cls_name[i]+'/'+filename
#             print(basedir)
#             image = Image.open(basedir)
#             m,n=image.size
#             c_x=m/2
#             c_y=n/2
#             x=c_x-w/2
#             y=c_y-h/2
#             region=image.crop((x,y,x+w,y+h))
#             region = region.resize((w, h), Image.BILINEAR)
#             region = region.convert("L")
#             img_ndarray = np.asarray(region, dtype='float32')/255
#                 # cam_img[i]=np.ndarray.flatten(img_ndarray)
#             img_ndarray=img_ndarray.reshape(w*h*1)
#             cam_img.append(img_ndarray)
#     np.save('Data-grey/'+cls_name[i]+'.npy',cam_img)
#     cam_img=[]

# cam_img=np.asarray(cam_img,dtype='float32')
# cam_label=np.asarray(cam_label,dtype=np.int)
#
# np.save('CamData.npy',cam_img)
# np.save('CamLabel.npy',[cam_label,cls_name])

filename_list=[]
for filename in os.listdir(r'test'):
    if (filename != 'Thumbs.db'):
        basedir = 'test/'+ filename
        print('Open'+basedir)
        image=Image.open(basedir)
        image=image.convert("L")
        image=image.resize((w,h),Image.BILINEAR)
        img_ndarray = np.asarray(image, dtype='float32') / 255
        img_ndarray = img_ndarray.reshape(w * h * 1)
        cam_img.append(img_ndarray)
        filename_list.append(filename)
cam_img=np.asarray(cam_img,dtype='float32')
# filename_list=np.asarray(filename_list)
np.save('TestData.npy',cam_img)
np.save('FileName.npy',filename_list)




# #
# # # image show
# #
# # img0=cam_img[200]
# # pylab.imshow(img0)
# # pylab.gray()
# # pylab.show()
#
# # save images


# # load images
#
cam=np.load('TestData.npy')
print(cam)
