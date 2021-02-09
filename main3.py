import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import sys
import random
import csv
import math



'''
mean1 = [-2, 0]
cov1 = [[2, 0.9], [0.9, 1]]
data1 = np.random.multivariate_normal(mean1, cov1, 300)

mean2 = [3, 4]
cov2 = [[3, 0.1], [0.1, 3]]
data2 = np.random.multivariate_normal(mean2, cov2, 300)

mean3 = [6, -6]
cov3 = [[2, -1], [-1, 2]]
data3 = np.random.multivariate_normal(mean3, cov3, 300)

data = np.concatenate((data1, data2, data3), axis=0)
centers = np.asarray([mean1, mean2, mean3])

# Plot
fig1 = plt.scatter(data1[:, 0], data1[:, 1])
fig2 = plt.scatter(data2[:, 0], data2[:, 1])
fig3 = plt.scatter(data3[:, 0], data3[:, 1])
fig4 = plt.scatter(centers[:, 0], centers[:, 1], marker="+", s=100)
# plt.scatter(data[:, 0], data[:, 1])
plt.title('Gaussian-Mixture Model')
plt.xlabel('x')
plt.ylabel('y')
plt.legend((fig1, fig2, fig3, fig4),
           ('Gaussian1', 'Guassian2', 'Guassian3', 'Means'),
           scatterpoints=1,
           loc='lower left',
           ncol=2,
           fontsize=10)
plt.savefig('before_k_medoids.png')
plt.show()


'''
data_input = pd.read_csv('data_set_01.csv')
data = np.array([list(row) for row in data_input.values])
print(type(data))
print(len(data))



df = pd.DataFrame(data)
normalized_df=(df-df.min())/(df.max()-df.min())
#normalized_df=(df-df.mean())/df.std()
#normalized_df.to_csv (r'normalized_df.csv', index = False, header=True)

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
    #print(min_distance)
    
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

print(medoids)
medoids_list_b = medoids

'''
df = pd.DataFrame(medoids)
df.to_csv (r'medoids.csv', index = False, header=False)
'''


medoids = np.asarray(medoids)
#print(indices)
print(medoids)
fig1 = plt.scatter(data[:, 0], data[:, 1])
#fig2 = plt.scatter(data2[:, 0], data2[:, 1])
#fig3 = plt.scatter(data3[:, 0], data3[:, 1])
#fig4 = plt.scatter(centers[:, 0], centers[:, 1], marker="+", s=200)
fig5 = plt.scatter(medoids[:, 0], medoids[:, 1], marker="*", s=200)
plt.title('Gaussian-Mixture Model')
plt.xlabel('x')
plt.ylabel('y')
comm = '''
'''
plt.legend((fig1,  fig5),
           ('Gaussian1',  'Medoids'),
           scatterpoints=1,
           loc='lower left',
           ncol=2,
           fontsize=10)

plt.savefig('after_k_medoids.png')
plt.show()

################################################################################## find result
'''
with open('normalized_df.csv', newline='') as f:
    reader = csv.reader(f)
    normalized_data = list(reader)
'''
normalized_data = [normalized_df.columns.values.tolist()] + normalized_df.values.tolist()

print(normalized_data[len(normalized_data)-1])
'''
with open('medoids.csv', newline='') as f:
    reader = csv.reader(f)
    medoids_list_b = list(reader)
'''


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

out_put_set =[]

for i in range(len(normalized_data)):

    last_data = normalized_data[i]
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
        "point":point,
        "position":last_data,
        "medoids":medoids_list,
    }
    out_put_set.append(output)
    output={}

#print(out_put_set)
