import numpy as np
import pandas as pd
import torch
import sys
import csv
import math
import os

from flask  import Flask ,request ,jsonify

import firebase_admin
from firebase_admin import db

app = Flask(__name__)

firebase_admin.initialize_app(options={
    'databaseURL': 'https://first-test-api-01-default-rtdb.firebaseio.com',})

mindfit = db.reference('mindfit')

def Average(lst): 
    return sum(lst) / len(lst)   

@app.route('/push_data', methods=['POST'])
def create_data():  

    req = request.json
    data_push = mindfit.push(req)

    return jsonify({"Status":"OK"})

@app.route('/get_data', methods=['POST'])
def read_data():
    data_get = mindfit.get()

    return jsonify(data_get)

@app.route('/k_medoids', methods=['POST'])
def k__medoids():

	############################################################################################################################

    data_input = request.json
    data_input_ori = request.json 
    data_push_p = mindfit.push(data_input)
    input_j = mindfit.get()
    input_key_list = [*input_j]
    data = []
    for i in range(len(input_key_list)):
        data.append(input_j["{}".format(input_key_list[i])])

	

    pre_data = []
    for i,x in enumerate(data):
        #pre_data.append([Average(data[i]["Choice"]) , data[i]["Result"]] )
        pre_data.append([Average(data[i]["Choice"]) , data[i]["Result"] * (data[i]["Skip"]+1)] )
        #pre_data.append([Average(data[i]["Choice"]) * (data[i]["Skip"]+1) , data[i]["Result"]] )


    #print(pre_data)

    df = pd.DataFrame(pre_data)
    df.to_csv (r'data_set_01.csv', index = False, header=True)

	############################################################################################################################

    data_input = pd.read_csv('data_set_01.csv')
    data = np.array([list(row) for row in data_input.values])
    print(type(data))
    print(len(data))

    df = pd.DataFrame(data)
    normalized_df=(df-df.min())/(df.max()-df.min())
    #normalized_df=(df-df.mean())/df.std()
    normalized_df.to_csv (r'normalized_df.csv', index = False, header=True)

    data = np.array(normalized_df)
    # construct the similarity matrix
    num = len(data)
    similarity_matrix = np.zeros((num, num))

    for i in range(0, num):
        for j in range(i+1, num):
            diff = data[i] - data[j]
            dist_tmp = np.linalg.norm(diff)
            similarity_matrix[i][j] = dist_tmp
            similarity_matrix[j][i] = dist_tmp



    similarity_matrix = torch.from_numpy(similarity_matrix)




    def k_medoids(similarity_matrix, k):
        
        # Step 1: Select initial medoids
        num = len(similarity_matrix)
        row_sums = torch.sum(similarity_matrix, dim=1)
        normalized_sim = similarity_matrix.T / row_sums
        normalized_sim = normalized_sim.T    
        priority_scores = -torch.sum(normalized_sim, dim=0)
        values, indices = priority_scores.topk(k)
        tmp = -similarity_matrix[:, indices]
        tmp_values, tmp_indices = tmp.topk(1, dim=1)
        min_distance = -torch.sum(tmp_values)
        cluster_assignment = tmp_indices.resize_(num)
        print(min_distance)
        
        # Step 2: Update medoids
        for i in range(k):
            sub_indices = (cluster_assignment == i).nonzero()
            sub_num = len(sub_indices)
            sub_indices = sub_indices.resize_(sub_num)
            sub_similarity_matrix = torch.index_select(similarity_matrix, 0, sub_indices)
            sub_similarity_matrix = torch.index_select(sub_similarity_matrix, 1, sub_indices)
            sub_row_sums = torch.sum(sub_similarity_matrix, dim=1)
            sub_medoid_index = torch.argmin(sub_row_sums)
            # update the cluster medoid index
            indices[i] = sub_indices[sub_medoid_index]


        # Step 3: Assign objects to medoids
        tmp = -similarity_matrix[:, indices]
        tmp_values, tmp_indices = tmp.topk(1, dim=1)
        total_distance = -torch.sum(tmp_values)
        cluster_assignment = tmp_indices.resize_(num)
        print(total_distance)
            
        while (total_distance < min_distance):
            min_distance = total_distance
            # Step 2: Update medoids
            for i in range(k):
                sub_indices = (cluster_assignment == i).nonzero()
                sub_num = len(sub_indices)
                sub_indices = sub_indices.resize_(sub_num)
                sub_similarity_matrix = torch.index_select(similarity_matrix, 0, sub_indices)
                sub_similarity_matrix = torch.index_select(sub_similarity_matrix, 1, sub_indices)
                sub_row_sums = torch.sum(sub_similarity_matrix, dim=1)
                sub_medoid_index = torch.argmin(sub_row_sums)
                # update the cluster medoid index
                indices[i] = sub_indices[sub_medoid_index]

            # Step 3: Assign objects to medoids
            tmp = -similarity_matrix[:, indices]
            tmp_values, tmp_indices = tmp.topk(1, dim=1)
            total_distance = -torch.sum(tmp_values)
            cluster_assignment = tmp_indices.resize_(num)
            #print(total_distance)
            
        return indices

    indices = k_medoids(similarity_matrix, k=4)
    medoids = []
    for i in range(4):
        medoids.append(data[indices[i]])


    df = pd.DataFrame(medoids)
    df.to_csv (r'medoids.csv', index = False, header=False)

    medoids = np.asarray(medoids)
        

    ################################################################################## find result

    with open('normalized_df.csv', newline='') as f:
        reader = csv.reader(f)
        normalized_data = list(reader)

    print(normalized_data[len(normalized_data)-1])

    with open('medoids.csv', newline='') as f:
        reader = csv.reader(f)
        medoids_list_b = list(reader)

    y_data_m=[]
    for i in range(4):
        y_data_m.append(float(medoids_list_b[i][1]))

    print("before sort ------------->",medoids_list_b)
    y_data_m_s = sorted(y_data_m)
    print("sort y",sorted(y_data_m))

    medoids_list=[]
    for i in range(4):
        for j in range(4):
            if float(medoids_list_b[j][1]) == y_data_m_s[i]:
                medoids_list.append(medoids_list_b[j])
                break

    print("after sort ------------->",medoids_list)


    last_data = normalized_data[len(normalized_data)-1]
    resu = []

    for i in range(len(medoids_list)):
        resu.append(   math.sqrt((float(medoids_list[i][0])-float(last_data[0])) ** 2) + ((float(medoids_list[i][1])-float(last_data[1])) ** 2)   )
    print(resu)


    min_resu = min(resu)

    for i in range(len(resu)):
        if min_resu == resu[i]:
            resu_no = i
    print(resu_no)


    set_resu_min = []

    for i in range(len(resu)):
        if resu[i] > min_resu:
            set_resu_min.append(resu[i])
    print(set_resu_min)

    min_set_resu_min = min(set_resu_min)

    for i in range(len(resu)):
        if resu[i] == min_set_resu_min:
            set_resu_min_no = i
    print(set_resu_min_no)

    max_len = (math.sqrt((float(medoids_list[resu_no][0])-float(medoids_list[set_resu_min_no][0])) ** 2) + ((float(medoids_list[resu_no][1])-float(medoids_list[set_resu_min_no][1])) ** 2))/2
    print(max_len)


    if resu_no == 0:
        if min_resu > max_len:
            point = 5
        elif min_resu > (max_len*5)/6 and min_resu <= max_len:
            point = 5
        elif min_resu > (max_len*4)/6 and min_resu <= (max_len*5)/6:
            point = 4
        elif min_resu > (max_len*3)/6 and min_resu <= (max_len*4)/6:
            point = 3
        elif min_resu > (max_len*2)/6 and min_resu <= (max_len*3)/6:
            point = 2
        elif min_resu > (max_len*1)/6 and min_resu <= (max_len*2)/6:
            point = 1
        elif min_resu > 0 and min_resu <= (max_len*1)/6:
            point = 0

    if resu_no == 1:
        if min_resu > max_len:
            point = 12
        elif min_resu > (max_len*6)/7 and min_resu <= max_len:
            point = 12
        elif min_resu > (max_len*5)/7 and min_resu <= (max_len*6)/7:
            point = 11
        elif min_resu > (max_len*4)/7 and min_resu <= (max_len*5)/7:
            point = 10
        elif min_resu > (max_len*3)/7 and min_resu <= (max_len*4)/7:
            point = 9
        elif min_resu > (max_len*2)/7 and min_resu <= (max_len*3)/7:
            point = 8
        elif min_resu > (max_len*1)/7 and min_resu <= (max_len*2)/7:
            point = 7
        elif min_resu > 0 and min_resu <= (max_len*1)/7:
            point = 6

    if resu_no == 2:
        if min_resu > max_len:
            point = 18
        elif min_resu > (max_len*5)/6 and min_resu <= max_len:
            point = 18
        elif min_resu > (max_len*4)/6 and min_resu <= (max_len*5)/6:
            point = 17
        elif min_resu > (max_len*3)/6 and min_resu <= (max_len*4)/6:
            point = 16
        elif min_resu > (max_len*2)/6 and min_resu <= (max_len*3)/6:
            point = 15
        elif min_resu > (max_len*1)/6 and min_resu <= (max_len*2)/6:
            point = 14
        elif min_resu > 0 and min_resu <= (max_len*1)/6:
            point = 13

    if resu_no == 3:
        if min_resu > max_len:
            point = 24
        elif min_resu > (max_len*5)/6 and min_resu <= max_len:
            point = 24
        elif min_resu > (max_len*4)/6 and min_resu <= (max_len*5)/6:
            point = 23
        elif min_resu > (max_len*3)/6 and min_resu <= (max_len*4)/6:
            point = 22
        elif min_resu > (max_len*2)/6 and min_resu <= (max_len*3)/6:
            point = 21
        elif min_resu > (max_len*1)/6 and min_resu <= (max_len*2)/6:
            point = 20
        elif min_resu > 0 and min_resu <= (max_len*1)/6:
            point = 19


    print(point)

    output = {
        "Uid":data_input_ori['Uid'],
        "point":point,
        "position":last_data,
        "medoids":medoids_list,
        "po_length":resu
    }

    print(output)	

    return jsonify(output)

if __name__ == '__main__':
    app.run(debug = True,port=int(os.environ.get('PORT',6001)))