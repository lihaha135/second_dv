# -*- coding: utf-8 -*-
import os
import numpy as np
import open3d as o3d
import shutil


if __name__=="__main__":
    data_path=r'/data/wj_data/32data'
    result_path=r'/data/wj_data/label0228'
    data_cvtpath=r'/data/wj_data/cvtdata'
    label_cvtpath=r'/data/wj_data/cvtlabel0228'
    process_mode='data11'
    label_namemode='n'
    if process_mode=='data':
        if not os.path.exists(data_cvtpath):
            os.mkdir(data_cvtpath)
        filelists = os.listdir(data_path)
        for filelist in filelists:
            data_numpath = os.path.join(data_path, filelist)
            data_timefiles = os.listdir(data_numpath)
            for data_timefile in data_timefiles:
                data_timepath = os.path.join(data_numpath, data_timefile)
                data_pcdfiles=os.listdir(data_timepath)
                for data_pcdfile in data_pcdfiles:
                    if data_pcdfile.split('_')[-1]=='xyzI.pcd':
                        data_pcdpath=os.path.join(data_timepath,data_pcdfile)
                        pcd = o3d.io.read_point_cloud(data_pcdpath)
                        # o3d.visualization.draw_geometries([pcd])
                        pc_data = np.asarray(pcd.points,dtype=np.float32)
                        # PC_data = PC_data.astype(np.float32)
                        print(pc_data.shape)
                        data_stpath=os.path.join(data_cvtpath,data_pcdfile.split('.')[0]+'.bin')
                        pc_data.tofile(data_stpath)

        print('data_collection is over')
    else:
        if not os.path.exists(label_cvtpath):
            os.mkdir(label_cvtpath)
        labelnumfiles=os.listdir(result_path)
        for labelnumfile in labelnumfiles:
            label_numpath=os.path.join(result_path,labelnumfile)
            print(labelnumfile)
            if os.path.isdir(label_numpath):
                label_files=os.listdir(label_numpath)
                num=0
                for label_file in label_files:

                    label_fpath=os.path.join(label_numpath,label_file)
                    if label_namemode=='y':
                        num+=1
                        label_rpath=os.path.join(label_numpath,label_file.split('_')[-3]+'.csv_error')
                        os.rename(label_fpath,label_rpath)
                        label_fpath=label_rpath

                    label_stpath=os.path.join(label_cvtpath,label_file)
                    shutil.copyfile(label_fpath,label_stpath)

        print(num)



