'''
    1. name2id
    2. id2name
    3. train id 
    4. test id 
    5. evaluate the accuracy on each class  
'''
import pickle as pkl
import argparse
import json 

model_path = '/home/aiotlab/emed_test/oneshot/data'
res_path = '/home/aiotlab/emed_test/oneshot/oneshot-object-detection-pytorch/output/res50/coco_2017_val/faster_rcnn_unseen'
eval_path  = res_path + '/detections_val2017_results.json'

with open(model_path + "/name2id.pkl", "rb") as f:
  name2id = pkl.load(f)

id2name =  dict((v,k) for k,v in name2id.items())

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--g', dest='group',
                      help='which group to train, split coco to four group',
                      default=0)

  args = parser.parse_args()
  return args

if __name__=="__main__":
    args = parse_args()
    
    group = args.group

    list_id = id2name.keys()
    test_id = [id  for id in list_id if id % 4 == group]
    train_id = list(set(list_id).difference(set(test_id)))

    with open(eval_path, 'r') as f:
        val = json.load(f)
    
    dct = {}
    dct_occurence = {}

    for result in val:
        if result['category_id'] not in result.keys():
            dct[result['category_id']] = 0
            dct_occurence[result['category_id']] = 0    
        dct[result['category_id']] += result['score']
        dct_occurence[result['category_id']] += 1

    for result in val:
        dct[result['category_id']] = dct[result['category_id']] / dct_occurence[result['category_id']]

    with open('res.json', 'w') as res_f:
        json.dump(dct, res_f)