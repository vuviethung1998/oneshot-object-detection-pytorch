import pickle
import json


res_path = '/home/aiotlab/emed_test/oneshot/oneshot-object-detection-pytorch/output/res50/coco_2017_val/faster_rcnn_unseen'
file_name = '/detection_results.pkl'

if __name__=="__main__":
    file_path = res_path + file_name 

    with open(file_path, 'rb') as pickle_file:
        res = pickle.load(pickle_file)
    print(res)