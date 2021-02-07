from flask import Flask, request, jsonify
import json
import requests
import time
import random
import numpy as np
import pandas as pd
import uuid 

app = Flask(__name__)
id = uuid.uuid1() 

@app.route('/test', methods=['POST'])
def my_json():

    Choice1 = []

    for i in range(50):

        for j in range(9):
            if i%4 == 0:
                Result=(random.uniform(4.00,6.00))
            elif i%4 == 1:
                Result=(random.uniform(12.00,15.00))
            elif i%4 == 2:
                Result=(random.uniform(18.00,21.00))
            elif i%4 == 3:
                Result=(random.uniform(27.00,28.00))
        Week=(random.randint(0,1))
        Skip=(random.randint(0,1))
        #print(i)
        for k in range(9):
            if i%4 == 0:
                Choice1.append(random.randint(0,9))
            elif i%4 == 1:
                Choice1.append(random.randint(0,9))
            elif i%4 == 2:
                Choice1.append(random.randint(0,9))
            elif i%4 == 3:
                Choice1.append(random.randint(0,9))        
        Choice=(Choice1)
        Choice1 = []
        UID=(id.hex+"x{}".format(i))

        dic={}

        dic["Choice"] = Choice
        dic["Result"] = Result
        dic["Week"] = Week
        dic["Uid"] = UID
        dic["Skip"] = Skip
        

        print("ROUND: {} -->".format(i),dic)	

        res2 = requests.post('https://kmedoids-deploy-nmdlf3uxjq-as.a.run.app/push_data', json=dic)	

        print("RESULT: {} <--".format(i),res2.json())			

        dic={}
        time.sleep(2)		
    return jsonify(dic)


if __name__ == '__main__':
    app.run(debug = True,port=6000)