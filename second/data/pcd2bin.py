import os #定义一个三维点类 
import numpy as np
class Point(object): 
     def __init__(self,x,y,z): 
           self.x = x 
           self.y = y 
           self.z = z 

data = r'/data/5data/pcd/'
save_dir= r'/data/5data/bin/'
def pcd2bin(filename,save_path):
    points=[]
    #读取pcd文件,从pcd的第12行开始是三维点 
    with open(filename) as f: 
       for line in f.readlines()[11:len(f.readlines())-1]: 
          strs = line.split(' ')
          print(strs)
          if len(strs[0]) < 0:
              continue
          points.append([float(strs[0]),float(strs[1]),float(strs[2]),0])
    p = np.array(points,dtype=np.float32)
    print(p.shape)
    p.tofile(save_path)

pcd_list = os.listdir(data)
for i in pcd_list:
   print(i)
   filename = data + i
   pcd2bin(filename,save_dir+i.replace('.pcd', '.bin'))

