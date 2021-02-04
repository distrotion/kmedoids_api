import random
import numpy as np
import pandas as pd
import json
import uuid 

Choice = []
Choice1 = []
Result = []
Week=[]
Uid=[]
Skip=[]
id = uuid.uuid1() 

for i in range(1000):

    
    for j in range(9):
        if i%4 == 0:
            Result.append(random.uniform(5.00,10.00))
        elif i%4 == 1:
            Result.append(random.uniform(10.00,15.00))
        elif i%4 == 2:
            Result.append(random.uniform(15.00,20.00))
        elif i%4 == 3:
            Result.append(random.uniform(20.00,26.00))
    Week.append(random.randint(0,1))
    Skip.append(random.randint(0,1))
    #print(i)
    for k in range(9):
        if i%4 == 0:
            Choice1.append(random.randint(4,5))
        elif i%4 == 1:
            Choice1.append(random.randint(3,4))
        elif i%4 == 2:
            Choice1.append(random.randint(2,3))
        elif i%4 == 3:
            Choice1.append(random.randint(0,2))
          
    Choice.append(Choice1)
    Choice1 = []
    Uid.append(id.hex+"x{}".format(i))

#print(Result)

data = []
dic={}
for i,x in enumerate(Choice):
    dic["Choice"] = Choice[i]
    dic["Result"] = Result[i]
    dic["Week"] = Week[i]
    dic["Uid"] = Uid[i]
    dic["Skip"] = Skip[i]
    data.append(dic)
    dic={}

########################################################################################################################################

def Average(lst): 
    return sum(lst) / len(lst) 

pre_data = []
for i,x in enumerate(data):
    #pre_data.append([Average(data[i]["Choice"]) , data[i]["Result"]] )
    pre_data.append([Average(data[i]["Choice"]) , data[i]["Result"] * (data[i]["Skip"]+1)] )
    #pre_data.append([Average(data[i]["Choice"]) * (data[i]["Week"]+1) , data[i]["Result"]] )


#print(pre_data)

df = pd.DataFrame(pre_data)
df.to_csv (r'test_set_01.csv', index = False, header=True)

#with open('pre_data.json', 'w') as json_file:
#    json.dump(data, json_file)