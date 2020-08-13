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


def skin_age_avg_classification(age_data, input_data, Sorted_Cluster_centroids, age_Sorted_Cluster_centroids, weight):
    skin_age_avg_hyd = []
    skin_age_avg_sh = []
    skin_age = []
    skin_score = []
    average_score = 70
    for i in range(len(age_data)):
        ### For avg_hydration part ###
        if input_data[i][0] > Sorted_Cluster_centroids[0][0]:
            skin_age_avg_hyd.append(age_Sorted_Cluster_centroids[0]-weight*(input_data[i][0]-Sorted_Cluster_centroids[0][0])/1.12)
        elif input_data[i][0] < Sorted_Cluster_centroids[0][0] and input_data[i][0] >= Sorted_Cluster_centroids[1][0]:
            ### (in_hyd-hyd0)/(hyd1-hyd0) = (pred_age-age0)/(age1-age0) ###
            skin_age_avg_hyd.append((age_Sorted_Cluster_centroids[1]-age_Sorted_Cluster_centroids[0])*weight*(input_data[i][0]-Sorted_Cluster_centroids[0][0])/(Sorted_Cluster_centroids[1][0]-Sorted_Cluster_centroids[0][0])+age_Sorted_Cluster_centroids[0])
        elif input_data[i][0] < Sorted_Cluster_centroids[1][0] and input_data[i][0] >= Sorted_Cluster_centroids[2][0]:
            skin_age_avg_hyd.append((age_Sorted_Cluster_centroids[2]-age_Sorted_Cluster_centroids[1])*weight*(input_data[i][0]-Sorted_Cluster_centroids[1][0])/(Sorted_Cluster_centroids[2][0]-Sorted_Cluster_centroids[1][0])+age_Sorted_Cluster_centroids[1])
        
        elif input_data[i][0] < Sorted_Cluster_centroids[2][0] and input_data[i][0] >= Sorted_Cluster_centroids[3][0]:
            skin_age_avg_hyd.append((age_Sorted_Cluster_centroids[3]-age_Sorted_Cluster_centroids[2])*weight*(input_data[i][0]-Sorted_Cluster_centroids[2][0])/(Sorted_Cluster_centroids[3][0]-Sorted_Cluster_centroids[2][0])+age_Sorted_Cluster_centroids[2])
        
        elif input_data[i][0] < Sorted_Cluster_centroids[3][0] and input_data[i][0] >= Sorted_Cluster_centroids[4][0]:
            skin_age_avg_hyd.append((age_Sorted_Cluster_centroids[4]-age_Sorted_Cluster_centroids[3])*weight*(input_data[i][0]-Sorted_Cluster_centroids[3][0])/(Sorted_Cluster_centroids[4][0]-Sorted_Cluster_centroids[3][0])+age_Sorted_Cluster_centroids[3])
        
        elif input_data[i][0] < Sorted_Cluster_centroids[4][0] and input_data[i][0] >= Sorted_Cluster_centroids[5][0]:
            skin_age_avg_hyd.append((age_Sorted_Cluster_centroids[5]-age_Sorted_Cluster_centroids[4])*weight*(input_data[i][0]-Sorted_Cluster_centroids[4][0])/(Sorted_Cluster_centroids[5][0]-Sorted_Cluster_centroids[4][0])+age_Sorted_Cluster_centroids[4])
        
        else:
            skin_age_avg_hyd.append(age_Sorted_Cluster_centroids[5]-weight*(input_data[i][0]-Sorted_Cluster_centroids[5][0])/1.07)

        ### For avg_skinhealth part ###
        if input_data[i][1] > Sorted_Cluster_centroids[0][1]:
            skin_age_avg_sh.append(age_Sorted_Cluster_centroids[0]-weight*(input_data[i][1]-Sorted_Cluster_centroids[0][1])/0.88)
        elif input_data[i][1] < Sorted_Cluster_centroids[0][1] and input_data[i][1] >= Sorted_Cluster_centroids[1][1]:
            ### (in_hyd-hyd0)/(hyd1-hyd0) = (pred_age-age0)/(age1-age0) ###
            skin_age_avg_sh.append((age_Sorted_Cluster_centroids[1]-age_Sorted_Cluster_centroids[0])*weight*(input_data[i][1]-Sorted_Cluster_centroids[0][1])/(Sorted_Cluster_centroids[1][0]-Sorted_Cluster_centroids[0][1])+age_Sorted_Cluster_centroids[0])
        elif input_data[i][1] < Sorted_Cluster_centroids[1][1] and input_data[i][1] >= Sorted_Cluster_centroids[2][1]:
            skin_age_avg_sh.append((age_Sorted_Cluster_centroids[2]-age_Sorted_Cluster_centroids[1])*weight*(input_data[i][1]-Sorted_Cluster_centroids[1][1])/(Sorted_Cluster_centroids[2][1]-Sorted_Cluster_centroids[1][1])+age_Sorted_Cluster_centroids[1])
        
        elif input_data[i][1] < Sorted_Cluster_centroids[2][1] and input_data[i][1] >= Sorted_Cluster_centroids[3][1]:
            skin_age_avg_sh.append((age_Sorted_Cluster_centroids[3]-age_Sorted_Cluster_centroids[2])*weight*(input_data[i][1]-Sorted_Cluster_centroids[2][1])/(Sorted_Cluster_centroids[3][1]-Sorted_Cluster_centroids[2][1])+age_Sorted_Cluster_centroids[2])
        
        elif input_data[i][1] < Sorted_Cluster_centroids[3][1] and input_data[i][1] >= Sorted_Cluster_centroids[4][1]:
            skin_age_avg_sh.append((age_Sorted_Cluster_centroids[4]-age_Sorted_Cluster_centroids[3])*weight*(input_data[i][1]-Sorted_Cluster_centroids[3][1])/(Sorted_Cluster_centroids[4][1]-Sorted_Cluster_centroids[3][1])+age_Sorted_Cluster_centroids[3])
        
        elif input_data[i][1] < Sorted_Cluster_centroids[4][1] and input_data[i][1] >= Sorted_Cluster_centroids[5][1]:
            skin_age_avg_sh.append((age_Sorted_Cluster_centroids[5]-age_Sorted_Cluster_centroids[4])*weight*(input_data[i][1]-Sorted_Cluster_centroids[4][1])/(Sorted_Cluster_centroids[5][1]-Sorted_Cluster_centroids[4][1])+age_Sorted_Cluster_centroids[4])
        
        else:
            skin_age_avg_sh.append(age_Sorted_Cluster_centroids[0]-weight*(input_data[i][1]-Sorted_Cluster_centroids[0][1])/0.82) ###

        print("skin_age_avg_hyd[-1] = ", skin_age_avg_hyd[-1])
        print("skin_age_avg_sh[-1] = ", skin_age_avg_sh[-1])

        skin_age.append((skin_age_avg_hyd[-1]+skin_age_avg_sh[-1])/2)
        
        print("(skin_age_avg_hyd[-1]+skin_age_avg_sh[-1])/2 = ", (skin_age_avg_hyd[-1]+skin_age_avg_sh[-1])/2)

        skin_score.append(average_score+(30*np.tanh(0.08*(age_data[i]-skin_age[-1])))[-1])

    return skin_age, skin_score


def store_csv(skin_age_pred, skin_score_pred):
    with open('avg_skin_age'+str(datetime.datetime.now().strftime("%Y_%m_%d"))+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(skin_age_pred)
        writer.writerow(skin_score_pred)

if __name__ == '__main__':
    age_Sorted_Cluster_centroids = [25.29, 30.23, 36.49, 42.36, 47.93, 52.77]
    Sorted_Cluster_centroids = [[72.24293785, 50.60867937], [66.70491803, 46.25772864], [62.04793028, 42.77640714], [57.78754579, 40.17853418], [53.41542289, 37.42862222], [48.20098039, 33.4825359]]
    Sorted_Cluster_centroids = np.stack(Sorted_Cluster_centroids, axis=0)
    weights = 0.3

    while True:
        test_age = input("Age:")
        print ("----------")
        test_left_hydration = input("Left_hydration:")
        print ("----------")
        test_left_skinhealth = input("Left_skinhealth:")
        print ("----------")
        test_right_hydration = input("Right_hydration:")
        print ("----------")
        test_right_skinhealth = input("Right_skinhealth:")
        print ("----------")
        test_fore_hydration = input("Fore_hydration:")
        print ("----------")
        test_fore_skinhealth = input("Fore_skinhealth:")
        print ("----------")

        # Average 3 parts of hydration and oxygen
        test_avg_hydration = (np.float(test_left_hydration)+np.float(test_right_hydration)+np.float(test_fore_hydration))/3
        test_avg_skinhealth = (np.float(test_left_skinhealth)+np.float(test_right_skinhealth)+np.float(test_fore_skinhealth))/3

        test_x_data = np.stack(([test_avg_hydration], [test_avg_skinhealth]), axis=1)
        print("test_x_data = ", test_x_data)
        test_skin_age_pred, test_skin_score_pred = skin_age_avg_classification([[int(test_age)]], test_x_data, Sorted_Cluster_centroids, age_Sorted_Cluster_centroids, weights)
        print("test_skin_age_pred = ", test_skin_age_pred)
        input("============")
        print("test_skin_score_pred = ", test_skin_score_pred)
        input("===Press Enter to continue...===")













