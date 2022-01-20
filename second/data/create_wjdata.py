
#from wjlidar_common import create_wjdata_infos
from second.data.all_dataset import create_groundtruth_database

def wj_data_prep(root_path):
    print('wjdata')
    # create_wjdata_infos(root_path)
    # data_path = '/data/wj_data/wjdata_info_train.pkl'
    # create_groundtruth_database('WjDataset',root_path,data_path)
    # create_groundtruth_database('WjDataset', root_path, root_path+'/wjdata_info_val.pkl')
    create_groundtruth_database('WjDataset', root_path, root_path+'/wjdata_info_train.pkl')
 



if __name__ == '__main__':
    root_path = '/data/second_dv/datasets/gate8_double/train'
    print(root_path)
    wj_data_prep(root_path)


