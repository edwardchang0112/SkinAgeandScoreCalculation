import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import datetime
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def readcsv(file_name, factor):
    dataset = pd.read_csv(file_name)
    #print (dataset[factor])
    return dataset[factor]


def skin_age_classification(age_data, input_data, Sorted_Cluster_centroids, age_Sorted_Cluster_centroids):
    skin_age = []
    skin_age_hyd = []
    skin_age_sh = []
    skin_score = []
    average_score = 70
    skin_age_weight = 0.3
    skin_score_weight = 0.08
    #print ("len(age_data) = ", len(age_data))
    for i in range(len(age_data)):
        ### For hydration part ###
        if input_data[i][0] > Sorted_Cluster_centroids[0][0]:
            skin_age_hyd.append(age_Sorted_Cluster_centroids[0]-skin_age_weight*(input_data[i][0]-Sorted_Cluster_centroids[0][0])/1.2)
        elif input_data[i][0] < Sorted_Cluster_centroids[0][0] and input_data[i][0] >= Sorted_Cluster_centroids[1][0]:
            ### (in_hyd-hyd0)/(hyd1-hyd0) = (pred_age-age0)/(age1-age0) ###
            skin_age_hyd.append((age_Sorted_Cluster_centroids[1]-age_Sorted_Cluster_centroids[0])*skin_age_weight*(input_data[i][0]-Sorted_Cluster_centroids[0][0])/(Sorted_Cluster_centroids[1][0]-Sorted_Cluster_centroids[0][0])+age_Sorted_Cluster_centroids[0])
        elif input_data[i][0] < Sorted_Cluster_centroids[1][0] and input_data[i][0] >= Sorted_Cluster_centroids[2][0]:
            skin_age_hyd.append((age_Sorted_Cluster_centroids[2]-age_Sorted_Cluster_centroids[1])*skin_age_weight*(input_data[i][0]-Sorted_Cluster_centroids[1][0])/(Sorted_Cluster_centroids[2][0]-Sorted_Cluster_centroids[1][0])+age_Sorted_Cluster_centroids[1])
        
        elif input_data[i][0] < Sorted_Cluster_centroids[2][0] and input_data[i][0] >= Sorted_Cluster_centroids[3][0]:
            skin_age_hyd.append((age_Sorted_Cluster_centroids[3]-age_Sorted_Cluster_centroids[2])*skin_age_weight*(input_data[i][0]-Sorted_Cluster_centroids[2][0])/(Sorted_Cluster_centroids[3][0]-Sorted_Cluster_centroids[2][0])+age_Sorted_Cluster_centroids[2])

        elif input_data[i][0] < Sorted_Cluster_centroids[3][0] and input_data[i][0] >= Sorted_Cluster_centroids[4][0]:
            skin_age_hyd.append((age_Sorted_Cluster_centroids[4]-age_Sorted_Cluster_centroids[3])*skin_age_weight*(input_data[i][0]-Sorted_Cluster_centroids[3][0])/(Sorted_Cluster_centroids[4][0]-Sorted_Cluster_centroids[3][0])+age_Sorted_Cluster_centroids[3])

        elif input_data[i][0] < Sorted_Cluster_centroids[4][0] and input_data[i][0] >= Sorted_Cluster_centroids[5][0]:
            skin_age_hyd.append((age_Sorted_Cluster_centroids[5]-age_Sorted_Cluster_centroids[4])*skin_age_weight*(input_data[i][0]-Sorted_Cluster_centroids[4][0])/(Sorted_Cluster_centroids[5][0]-Sorted_Cluster_centroids[4][0])+age_Sorted_Cluster_centroids[4])
        
        else:
            skin_age_hyd.append(age_Sorted_Cluster_centroids[5]-skin_age_weight*(input_data[i][0]-Sorted_Cluster_centroids[5][0])/0.99)
        
        ### For skinhealth part ###
        if input_data[i][1] > Sorted_Cluster_centroids[0][1]:
            skin_age_sh.append(age_Sorted_Cluster_centroids[0]-skin_age_weight*(input_data[i][1]-Sorted_Cluster_centroids[0][1])/1.27)
        elif input_data[i][1] < Sorted_Cluster_centroids[0][1] and input_data[i][1] >= Sorted_Cluster_centroids[1][1]:
            ### (in_hyd-hyd0)/(hyd1-hyd0) = (pred_age-age0)/(age1-age0) ###
            skin_age_sh.append((age_Sorted_Cluster_centroids[1]-age_Sorted_Cluster_centroids[0])*skin_age_weight*(input_data[i][1]-Sorted_Cluster_centroids[0][1])/(Sorted_Cluster_centroids[1][0]-Sorted_Cluster_centroids[0][1])+age_Sorted_Cluster_centroids[0])
        elif input_data[i][1] < Sorted_Cluster_centroids[1][1] and input_data[i][1] >= Sorted_Cluster_centroids[2][1]:
            skin_age_sh.append((age_Sorted_Cluster_centroids[2]-age_Sorted_Cluster_centroids[1])*skin_age_weight*(input_data[i][1]-Sorted_Cluster_centroids[1][1])/(Sorted_Cluster_centroids[2][1]-Sorted_Cluster_centroids[1][1])+age_Sorted_Cluster_centroids[1])
        
        elif input_data[i][1] < Sorted_Cluster_centroids[2][1] and input_data[i][1] >= Sorted_Cluster_centroids[3][1]:
            skin_age_sh.append((age_Sorted_Cluster_centroids[3]-age_Sorted_Cluster_centroids[2])*skin_age_weight*(input_data[i][1]-Sorted_Cluster_centroids[2][1])/(Sorted_Cluster_centroids[3][1]-Sorted_Cluster_centroids[2][1])+age_Sorted_Cluster_centroids[2])
        
        elif input_data[i][1] < Sorted_Cluster_centroids[3][1] and input_data[i][1] >= Sorted_Cluster_centroids[4][1]:
            skin_age_sh.append((age_Sorted_Cluster_centroids[4]-age_Sorted_Cluster_centroids[3])*skin_age_weight*(input_data[i][1]-Sorted_Cluster_centroids[3][1])/(Sorted_Cluster_centroids[4][1]-Sorted_Cluster_centroids[3][1])+age_Sorted_Cluster_centroids[3])
        
        elif input_data[i][1] < Sorted_Cluster_centroids[4][1] and input_data[i][1] >= Sorted_Cluster_centroids[5][1]:
            skin_age_sh.append((age_Sorted_Cluster_centroids[5]-age_Sorted_Cluster_centroids[4])*skin_age_weight*(input_data[i][1]-Sorted_Cluster_centroids[4][1])/(Sorted_Cluster_centroids[5][1]-Sorted_Cluster_centroids[4][1])+age_Sorted_Cluster_centroids[4])
        
        else:
            skin_age_sh.append(age_Sorted_Cluster_centroids[0]-skin_age_weight*(input_data[i][1]-Sorted_Cluster_centroids[0][1])/0.84)

        skin_age.append((skin_age_hyd[-1]+skin_age_sh[-1])/2)
        skin_score.append(average_score+(30*np.tanh(skin_score_weight*(age_data[i]-skin_age[-1]))))

    skin_age = np.stack(skin_age)
    skin_score = np.stack(skin_score)
    return skin_age, skin_score

def store_csv(skin_age_pred, skin_score_pred):
    with open('skin_age_single'+str(datetime.datetime.now().strftime("%Y_%m_%d"))+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(skin_age_pred)
        writer.writerow(skin_score_pred)

if __name__ == '__main__':
    file_path = 'All_data/Phase2/'
    age = readcsv(file_path+'P2_ICI_test_data_SinglePoint.csv', 'Age')
    hydration = readcsv(file_path+'P2_ICI_test_data_SinglePoint.csv', 'Hydration')
    skinhealth = readcsv(file_path+'P2_ICI_test_data_SinglePoint.csv', 'Skinhealth')
    
    x_data = np.stack((hydration, skinhealth), axis=1)
    
    avg_x_data = np.mean((x_data),axis=0)
    print ("avg_x_data = ", avg_x_data)
    #print("x_data = ", x_data)
    
    print ("age = ", np.asarray(age))
    #age_data = np.stack((np.asarray(age)), axis=1)
    age_kmeans = KMeans(n_clusters=6, random_state=0).fit(np.asarray(age).reshape(-1, 1))
    age_Cluster_centroids = age_kmeans.cluster_centers_
    age_Sorted_Cluster_centroids = sorted(age_Cluster_centroids, key = lambda x : x[0])
    print ("age_Sorted_Cluster_centroids = ", age_Sorted_Cluster_centroids)
    age_Sorted_Cluster_centroids = np.stack(age_Sorted_Cluster_centroids, axis=1)
    print ("age_Sorted_Cluster_centroids = ", age_Sorted_Cluster_centroids)
    '''
    kmeans = KMeans(n_clusters=6, random_state=0).fit(x_data)
    #print ("kmeans.labels_ = ", kmeans.labels_)
    Cluster_centroids = kmeans.cluster_centers_
    Sorted_Cluster_centroids = sorted(Cluster_centroids, key = lambda x : x[0], reverse=True)
    '''
    hyd_kmeans = KMeans(n_clusters=6, random_state=0).fit(np.asarray(hydration).reshape(-1, 1))
    hyd_Cluster_centroids = hyd_kmeans.cluster_centers_
    hyd_Sorted_Cluster_centroids = sorted(hyd_Cluster_centroids, key = lambda x : x[0], reverse=True)
    
    skinh_kmeans = KMeans(n_clusters=6, random_state=0).fit(np.asarray(skinhealth).reshape(-1, 1))
    skinh_Cluster_centroids = skinh_kmeans.cluster_centers_
    skinh_Sorted_Cluster_centroids = sorted(skinh_Cluster_centroids, key = lambda x : x[0], reverse=True)
    
    #print ("F_skinh_Sorted_Cluster_centroids = ", F_skinh_Sorted_Cluster_centroids)
    
    x_data_k_centroids = np.concatenate((hyd_Sorted_Cluster_centroids, skinh_Sorted_Cluster_centroids), axis=1)
    
    print ("x_data_k_centroids = ", x_data_k_centroids)
    input("===")
    #print ("score = ", kmeans.score(x_data))
    #Sorted_Cluster_centroids = [[74.9, 43.21, 77, 47.88, 66.4, 44.51], [68.69, 40.78, 67.57, 44.79, 71.57, 46.19], [62.86, 38, 64.51, 40.91, 61.54, 40.43], [61.79, 43.44, 57.86, 43.28, 60, 44.37], [59.32, 37.74, 58.84, 39.51, 48.58, 39.91], [45, 30.42, 47.5, 33.48, 55, 36.55]]
    
    
    #print ("kmeans.score = ", kmeans.score(x_data))
    #weights_1 = 0.3
    skin_age_pred, skin_score_pred = skin_age_classification(age, x_data, x_data_k_centroids, age_Sorted_Cluster_centroids[0])
    store_csv(skin_age_pred, skin_score_pred)

    while True:
        test_age = input("Age:")
        print ("----------")
        test_hydration = input("hydration:")
        print ("----------")
        test_skinhealth = input("skinhealth:")
        print ("----------")
        test_x_data = np.stack(([int(test_hydration)], [int(test_skinhealth)]), axis=1)
        test_skin_age_pred, test_skin_score_pred = skin_age_classification([int(test_age)], test_x_data, x_data_k_centroids, weights_1, avg_x_data, age_Sorted_Cluster_centroids[0])
        print("test_skin_age_pred = ", test_skin_age_pred)
        input("============")
        print("test_skin_score_pred = ", test_skin_score_pred)
        input("===Press Enter to continue...===")














