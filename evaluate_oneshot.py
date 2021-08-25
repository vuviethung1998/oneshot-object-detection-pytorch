'''
    1. name2id
    2. id2name
    3. train id 
    4. test id 
    5. evaluate the accuracy on each class  
'''
import pickle as pkl

model_path = '/home/aiotlab/emed_test/oneshot/data'

with open(model_path + "/name2id.pkl", "rb") as f:
  name2id = pkl.load(f)

id2name =  dict((v,k) for k,v in name2id.items())


if __name__=="__main__":
    print(id2name)

