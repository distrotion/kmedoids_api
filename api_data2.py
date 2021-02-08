import random
import numpy as np
import pandas as pd
import json
import uuid 
from flask import Flask, request, jsonify
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
id = uuid.uuid1() 

@app.route('/test', methods=['POST'])
def INPUTDATA():
    Choice = []
    Choice1 = []
    Result = []
    Week=[]
    UID=[]
    Skip=[]

    for i in range(1):

        Result.append(random.uniform(4.00,8.00))
        Week.append(random.randint(0,1))
        Skip.append(random.randint(0,1))
        #print(i)
        for j in range(9):
            if i%5 == 1:
                Choice1.append(random.randint(1,5))
            elif i%5 == 2:
                Choice1.append(random.randint(4,9))
            elif i%5 == 3:
                Choice1.append(random.randint(4,9))
            elif i%5 == 4:
                Choice1.append(random.randint(4,9))
            elif i%5 == 0:
                Choice1.append(random.randint(4,9))            
        Choice.append(Choice1)
        Choice1 = []
        UID.append(id.hex+"x{}".format(i))

    #print(Result)

    data = []
    dic={}
    for i,x in enumerate(Choice):
        dic["Choice"] = Choice[i]
        dic["Result"] = Result[i]
        dic["Week"] = Week[i]
        dic["Uid"] = UID[i]
        dic["Skip"] = Skip[i]
        data.append(dic)
        dic={}

    #print(data)

    return jsonify(data[0])


if __name__ == '__main__':
   app.run(debug = True,port=6005)