import os
import cv2

path='/home/wuwei10/qat_worksapce/q-diffusion-stable/q-diffusion/ckpts/float/2023-10-08-18-01-18/samples'
#'/home/wuwei10/qat_worksapce/q-diffusion-stable/q-diffusion/ckpts/qdiff/2023-10-08-15-54-26/samples'
#/home/wuwei10/qat_worksapce/q-diffusion-stable/q-diffusion/ckpts/ours/2023-10-08-16-08-05/samples'
#'/home/wuwei10/qat_worksapce/q-diffusion-stable/q-diffusion/ckpts/float/2023-10-08-18-01-18/samples' #/home/wuwei10/qat_worksapce/PTQ4DM/PTQ4DM/data/church_outdoor_val'
img_path=[]
path2='/home/wuwei10/qat_worksapce/q-diffusion-stable/q-diffusion/ckpts/float/float_final/samples'
def get_img_path(path,img_path):
    for i in range(1380):
        img_name=str(i).zfill(5)+'.png'
        img_path.append(os.path.join(path,img_name))

get_img_path(path,img_path)
print(len(img_path))
i=3620
for temp in img_path:
    img=cv2.imread(temp)
    new_name=str(i).zfill(5)+'.png'
    new_path=os.path.join(path2,new_name)
    cv2.imwrite(new_path,img)
    i=i+1