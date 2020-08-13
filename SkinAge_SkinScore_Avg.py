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


def skin_age_score_calculation(age_data, input_data, Sorted_Cluster_centroids, weights_1, avg_x_data, age_Sorted_Cluster_centroids):
    skin_age = []
    skin_score = []
    skin_age_left_hyd = []
    skin_age_left_sh = []
    skin_age_right_hyd = []
    skin_age_right_sh = []
    skin_age_fore_hyd = []
    skin_age_fore_sh = []
    average_score = 60
    #print ("len(age_data) = ", len(age_data))
    for i in range(len(age_data)):
        ### For left_hydration part ###
        if input_data[i][0] > Sorted_Cluster_centroids[0][0]:
            skin_age_left_hyd.append(age_Sorted_Cluster_centroids[0]-(input_data[i][0]-Sorted_Cluster_centroids[0][0])/1.32)
        elif input_data[i][0] < Sorted_Cluster_centroids[0][0] and input_data[i][0] >= Sorted_Cluster_centroids[1][0]:
            ### (in_hyd-hyd0)/(hyd1-hyd0) = (pred_age-age0)/(age1-age0) ###
            skin_age_left_hyd.append((age_Sorted_Cluster_centroids[1]-age_Sorted_Cluster_centroids[0])*(input_data[i][0]-Sorted_Cluster_centroids[0][0])/(Sorted_Cluster_centroids[1][0]-Sorted_Cluster_centroids[0][0])+age_Sorted_Cluster_centroids[0])
        elif input_data[i][0] < Sorted_Cluster_centroids[1][0] and input_data[i][0] >= Sorted_Cluster_centroids[2][0]:
            skin_age_left_hyd.append((age_Sorted_Cluster_centroids[2]-age_Sorted_Cluster_centroids[1])*(input_data[i][0]-Sorted_Cluster_centroids[1][0])/(Sorted_Cluster_centroids[2][0]-Sorted_Cluster_centroids[1][0])+age_Sorted_Cluster_centroids[1])
        
        elif input_data[i][0] < Sorted_Cluster_centroids[2][0] and input_data[i][0] >= Sorted_Cluster_centroids[3][0]:
            skin_age_left_hyd.append((age_Sorted_Cluster_centroids[3]-age_Sorted_Cluster_centroids[2])*(input_data[i][0]-Sorted_Cluster_centroids[2][0])/(Sorted_Cluster_centroids[3][0]-Sorted_Cluster_centroids[2][0])+age_Sorted_Cluster_centroids[2])
        
        elif input_data[i][0] < Sorted_Cluster_centroids[3][0] and input_data[i][0] >= Sorted_Cluster_centroids[4][0]:
            skin_age_left_hyd.append((age_Sorted_Cluster_centroids[4]-age_Sorted_Cluster_centroids[3])*(input_data[i][0]-Sorted_Cluster_centroids[3][0])/(Sorted_Cluster_centroids[4][0]-Sorted_Cluster_centroids[3][0])+age_Sorted_Cluster_centroids[3])
        
        elif input_data[i][0] < Sorted_Cluster_centroids[4][0] and input_data[i][0] >= Sorted_Cluster_centroids[5][0]:
            skin_age_left_hyd.append((age_Sorted_Cluster_centroids[5]-age_Sorted_Cluster_centroids[4])*(input_data[i][0]-Sorted_Cluster_centroids[4][0])/(Sorted_Cluster_centroids[5][0]-Sorted_Cluster_centroids[4][0])+age_Sorted_Cluster_centroids[4])
        
        else:
            skin_age_left_hyd.append(age_Sorted_Cluster_centroids[5]-(input_data[i][0]-Sorted_Cluster_centroids[5][0])/1.22)
        
        ### For left_skinhealth part ###
        if input_data[i][1] > Sorted_Cluster_centroids[0][1]:
            skin_age_left_sh.append(age_Sorted_Cluster_centroids[0]-(input_data[i][1]-Sorted_Cluster_centroids[0][1])/0.87)
        elif input_data[i][1] < Sorted_Cluster_centroids[0][1] and input_data[i][1] >= Sorted_Cluster_centroids[1][1]:
            ### (in_hyd-hyd0)/(hyd1-hyd0) = (pred_age-age0)/(age1-age0) ###
            skin_age_left_sh.append((age_Sorted_Cluster_centroids[1]-age_Sorted_Cluster_centroids[0])*(input_data[i][1]-Sorted_Cluster_centroids[0][1])/(Sorted_Cluster_centroids[1][0]-Sorted_Cluster_centroids[0][1])+age_Sorted_Cluster_centroids[0])
        elif input_data[i][1] < Sorted_Cluster_centroids[1][1] and input_data[i][1] >= Sorted_Cluster_centroids[2][1]:
            skin_age_left_sh.append((age_Sorted_Cluster_centroids[2]-age_Sorted_Cluster_centroids[1])*(input_data[i][1]-Sorted_Cluster_centroids[1][1])/(Sorted_Cluster_centroids[2][1]-Sorted_Cluster_centroids[1][1])+age_Sorted_Cluster_centroids[1])
        
        elif input_data[i][1] < Sorted_Cluster_centroids[2][1] and input_data[i][1] >= Sorted_Cluster_centroids[3][1]:
            skin_age_left_sh.append((age_Sorted_Cluster_centroids[3]-age_Sorted_Cluster_centroids[2])*(input_data[i][1]-Sorted_Cluster_centroids[2][1])/(Sorted_Cluster_centroids[3][1]-Sorted_Cluster_centroids[2][1])+age_Sorted_Cluster_centroids[2])
        
        elif input_data[i][1] < Sorted_Cluster_centroids[3][1] and input_data[i][1] >= Sorted_Cluster_centroids[4][1]:
            skin_age_left_sh.append((age_Sorted_Cluster_centroids[4]-age_Sorted_Cluster_centroids[3])*(input_data[i][1]-Sorted_Cluster_centroids[3][1])/(Sorted_Cluster_centroids[4][1]-Sorted_Cluster_centroids[3][1])+age_Sorted_Cluster_centroids[3])
        
        elif input_data[i][1] < Sorted_Cluster_centroids[4][1] and input_data[i][1] >= Sorted_Cluster_centroids[5][1]:
            skin_age_left_sh.append((age_Sorted_Cluster_centroids[5]-age_Sorted_Cluster_centroids[4])*(input_data[i][1]-Sorted_Cluster_centroids[4][1])/(Sorted_Cluster_centroids[5][1]-Sorted_Cluster_centroids[4][1])+age_Sorted_Cluster_centroids[4])
        
        else:
            skin_age_left_sh.append(age_Sorted_Cluster_centroids[0]-(input_data[i][1]-Sorted_Cluster_centroids[0][1])/0.54)
        
        
        ### For right_hydration part ###
        if input_data[i][2] > Sorted_Cluster_centroids[0][2]:
            skin_age_right_hyd.append(age_Sorted_Cluster_centroids[0]-(input_data[i][2]-Sorted_Cluster_centroids[0][2])/1.58)
        elif input_data[i][2] < Sorted_Cluster_centroids[0][2] and input_data[i][2] >= Sorted_Cluster_centroids[1][2]:
            ### (in_hyd-hyd0)/(hyd1-hyd0) = (pred_age-age0)/(age1-age0) ###
            skin_age_right_hyd.append((age_Sorted_Cluster_centroids[1]-age_Sorted_Cluster_centroids[0])*(input_data[i][2]-Sorted_Cluster_centroids[0][2])/(Sorted_Cluster_centroids[1][2]-Sorted_Cluster_centroids[0][2])+age_Sorted_Cluster_centroids[0])
        elif input_data[i][2] < Sorted_Cluster_centroids[1][2] and input_data[i][2] >= Sorted_Cluster_centroids[2][2]:
            skin_age_right_hyd.append((age_Sorted_Cluster_centroids[2]-age_Sorted_Cluster_centroids[1])*(input_data[i][2]-Sorted_Cluster_centroids[1][2])/(Sorted_Cluster_centroids[2][2]-Sorted_Cluster_centroids[1][2])+age_Sorted_Cluster_centroids[1])
        
        elif input_data[i][2] < Sorted_Cluster_centroids[2][2] and input_data[i][2] >= Sorted_Cluster_centroids[3][2]:
            skin_age_right_hyd.append((age_Sorted_Cluster_centroids[3]-age_Sorted_Cluster_centroids[2])*(input_data[i][2]-Sorted_Cluster_centroids[2][2])/(Sorted_Cluster_centroids[3][2]-Sorted_Cluster_centroids[2][2])+age_Sorted_Cluster_centroids[2])
        
        elif input_data[i][2] < Sorted_Cluster_centroids[3][2] and input_data[i][2] >= Sorted_Cluster_centroids[4][2]:
            skin_age_right_hyd.append((age_Sorted_Cluster_centroids[4]-age_Sorted_Cluster_centroids[3])*(input_data[i][2]-Sorted_Cluster_centroids[3][2])/(Sorted_Cluster_centroids[4][2]-Sorted_Cluster_centroids[3][2])+age_Sorted_Cluster_centroids[3])
        
        elif input_data[i][2] < Sorted_Cluster_centroids[4][2] and input_data[i][2] >= Sorted_Cluster_centroids[5][2]:
            skin_age_right_hyd.append((age_Sorted_Cluster_centroids[5]-age_Sorted_Cluster_centroids[4])*(input_data[i][2]-Sorted_Cluster_centroids[4][2])/(Sorted_Cluster_centroids[5][2]-Sorted_Cluster_centroids[4][2])+age_Sorted_Cluster_centroids[4])
        
        else:
            skin_age_right_hyd.append(age_Sorted_Cluster_centroids[5]-(input_data[i][2]-Sorted_Cluster_centroids[5][2])/0.71)
        
        ### For right_skinhealth part ###
        if input_data[i][3] > Sorted_Cluster_centroids[0][3]:
            skin_age_right_sh.append(age_Sorted_Cluster_centroids[0]-(input_data[i][3]-Sorted_Cluster_centroids[0][3])/0.6)
        elif input_data[i][3] < Sorted_Cluster_centroids[0][3] and input_data[i][3] >= Sorted_Cluster_centroids[1][3]:
            ### (in_hyd-hyd0)/(hyd1-hyd0) = (pred_age-age0)/(age1-age0) ###
            skin_age_right_sh.append((age_Sorted_Cluster_centroids[1]-age_Sorted_Cluster_centroids[0])*(input_data[i][3]-Sorted_Cluster_centroids[0][3])/(Sorted_Cluster_centroids[1][3]-Sorted_Cluster_centroids[0][3])+age_Sorted_Cluster_centroids[0])
        elif input_data[i][3] < Sorted_Cluster_centroids[1][3] and input_data[i][3] >= Sorted_Cluster_centroids[2][3]:
            skin_age_right_sh.append((age_Sorted_Cluster_centroids[2]-age_Sorted_Cluster_centroids[1])*(input_data[i][3]-Sorted_Cluster_centroids[1][3])/(Sorted_Cluster_centroids[2][3]-Sorted_Cluster_centroids[1][3])+age_Sorted_Cluster_centroids[1])
        
        elif input_data[i][3] < Sorted_Cluster_centroids[2][3] and input_data[i][3] >= Sorted_Cluster_centroids[3][3]:
            skin_age_right_sh.append((age_Sorted_Cluster_centroids[3]-age_Sorted_Cluster_centroids[2])*(input_data[i][3]-Sorted_Cluster_centroids[2][3])/(Sorted_Cluster_centroids[3][3]-Sorted_Cluster_centroids[2][3])+age_Sorted_Cluster_centroids[2])
        
        elif input_data[i][3] < Sorted_Cluster_centroids[3][3] and input_data[i][3] >= Sorted_Cluster_centroids[4][3]:
            skin_age_right_sh.append((age_Sorted_Cluster_centroids[4]-age_Sorted_Cluster_centroids[3])*(input_data[i][3]-Sorted_Cluster_centroids[3][3])/(Sorted_Cluster_centroids[4][3]-Sorted_Cluster_centroids[3][3])+age_Sorted_Cluster_centroids[3])
        
        elif input_data[i][3] < Sorted_Cluster_centroids[4][3] and input_data[i][3] >= Sorted_Cluster_centroids[5][3]:
            skin_age_right_sh.append((age_Sorted_Cluster_centroids[5]-age_Sorted_Cluster_centroids[4])*(input_data[i][3]-Sorted_Cluster_centroids[4][3])/(Sorted_Cluster_centroids[5][3]-Sorted_Cluster_centroids[4][3])+age_Sorted_Cluster_centroids[4])
        
        else:
            skin_age_right_sh.append(age_Sorted_Cluster_centroids[0]-(input_data[i][3]-Sorted_Cluster_centroids[0][3])/0.43)
        
        ### For fore_hydration part ###
        if input_data[i][4] > Sorted_Cluster_centroids[0][4]:
            skin_age_fore_hyd.append(age_Sorted_Cluster_centroids[0]-(input_data[i][4]-Sorted_Cluster_centroids[0][4])/2.74)
        elif input_data[i][4] < Sorted_Cluster_centroids[0][4] and input_data[i][4] >= Sorted_Cluster_centroids[1][4]:
            ### (in_hyd-hyd0)/(hyd1-hyd0) = (pred_age-age0)/(age1-age0) ###
            skin_age_fore_hyd.append((age_Sorted_Cluster_centroids[1]-age_Sorted_Cluster_centroids[0])*(input_data[i][4]-Sorted_Cluster_centroids[0][4])/(Sorted_Cluster_centroids[1][4]-Sorted_Cluster_centroids[0][4])+age_Sorted_Cluster_centroids[0])
        elif input_data[i][4] < Sorted_Cluster_centroids[1][4] and input_data[i][4] >= Sorted_Cluster_centroids[2][4]:
            skin_age_fore_hyd.append((age_Sorted_Cluster_centroids[2]-age_Sorted_Cluster_centroids[1])*(input_data[i][4]-Sorted_Cluster_centroids[1][4])/(Sorted_Cluster_centroids[2][4]-Sorted_Cluster_centroids[1][4])+age_Sorted_Cluster_centroids[1])
        
        elif input_data[i][4] < Sorted_Cluster_centroids[2][4] and input_data[i][4] >= Sorted_Cluster_centroids[3][4]:
            skin_age_fore_hyd.append((age_Sorted_Cluster_centroids[3]-age_Sorted_Cluster_centroids[2])*(input_data[i][4]-Sorted_Cluster_centroids[2][4])/(Sorted_Cluster_centroids[3][4]-Sorted_Cluster_centroids[2][4])+age_Sorted_Cluster_centroids[2])
        
        elif input_data[i][4] < Sorted_Cluster_centroids[3][4] and input_data[i][4] >= Sorted_Cluster_centroids[4][4]:
            skin_age_fore_hyd.append((age_Sorted_Cluster_centroids[4]-age_Sorted_Cluster_centroids[3])*(input_data[i][4]-Sorted_Cluster_centroids[3][4])/(Sorted_Cluster_centroids[4][4]-Sorted_Cluster_centroids[3][4])+age_Sorted_Cluster_centroids[3])
        
        elif input_data[i][4] < Sorted_Cluster_centroids[4][4] and input_data[i][4] >= Sorted_Cluster_centroids[5][4]:
            skin_age_fore_hyd.append((age_Sorted_Cluster_centroids[5]-age_Sorted_Cluster_centroids[4])*(input_data[i][4]-Sorted_Cluster_centroids[4][4])/(Sorted_Cluster_centroids[5][4]-Sorted_Cluster_centroids[4][4])+age_Sorted_Cluster_centroids[4])
        
        else:
            skin_age_fore_hyd.append(age_Sorted_Cluster_centroids[5]-(input_data[i][4]-Sorted_Cluster_centroids[5][4])/0.81)
        
        ### For fore_skinhealth part ###
        if input_data[i][5] > Sorted_Cluster_centroids[0][5]:
            skin_age_fore_sh.append(age_Sorted_Cluster_centroids[0]-(input_data[i][5]-Sorted_Cluster_centroids[0][5])/0.7)
        elif input_data[i][5] < Sorted_Cluster_centroids[0][5] and input_data[i][5] >= Sorted_Cluster_centroids[1][5]:
            ### (in_hyd-hyd0)/(hyd1-hyd0) = (pred_age-age0)/(age1-age0) ###
            skin_age_fore_sh.append((age_Sorted_Cluster_centroids[1]-age_Sorted_Cluster_centroids[0])*(input_data[i][5]-Sorted_Cluster_centroids[0][5])/(Sorted_Cluster_centroids[1][5]-Sorted_Cluster_centroids[0][5])+age_Sorted_Cluster_centroids[0])
        elif input_data[i][5] < Sorted_Cluster_centroids[1][5] and input_data[i][5] >= Sorted_Cluster_centroids[2][5]:
            skin_age_fore_sh.append((age_Sorted_Cluster_centroids[2]-age_Sorted_Cluster_centroids[1])*(input_data[i][5]-Sorted_Cluster_centroids[1][5])/(Sorted_Cluster_centroids[2][5]-Sorted_Cluster_centroids[1][5])+age_Sorted_Cluster_centroids[1])
        
        elif input_data[i][5] < Sorted_Cluster_centroids[2][5] and input_data[i][5] >= Sorted_Cluster_centroids[3][5]:
            skin_age_fore_sh.append((age_Sorted_Cluster_centroids[3]-age_Sorted_Cluster_centroids[2])*(input_data[i][5]-Sorted_Cluster_centroids[2][5])/(Sorted_Cluster_centroids[3][5]-Sorted_Cluster_centroids[2][5])+age_Sorted_Cluster_centroids[2])
        
        elif input_data[i][5] < Sorted_Cluster_centroids[3][5] and input_data[i][5] >= Sorted_Cluster_centroids[4][5]:
            skin_age_fore_sh.append((age_Sorted_Cluster_centroids[4]-age_Sorted_Cluster_centroids[3])*(input_data[i][5]-Sorted_Cluster_centroids[3][5])/(Sorted_Cluster_centroids[4][5]-Sorted_Cluster_centroids[3][5])+age_Sorted_Cluster_centroids[3])
        
        elif input_data[i][5] < Sorted_Cluster_centroids[4][5] and input_data[i][5] >= Sorted_Cluster_centroids[5][5]:
            skin_age_fore_sh.append((age_Sorted_Cluster_centroids[5]-age_Sorted_Cluster_centroids[4])*(input_data[i][5]-Sorted_Cluster_centroids[4][5])/(Sorted_Cluster_centroids[5][5]-Sorted_Cluster_centroids[4][5])+age_Sorted_Cluster_centroids[4])
        
        else:
            skin_age_fore_sh.append(age_Sorted_Cluster_centroids[0]-(input_data[i][5]-Sorted_Cluster_centroids[0][5])/0.3)
        
        
        skin_age.append((skin_age_left_hyd[-1]+skin_age_left_sh[-1]+skin_age_right_hyd[-1]+skin_age_right_sh[-1]+skin_age_fore_hyd[-1]+skin_age_fore_sh[-1])/6)
        
        skin_score.append(average_score+(40*np.tanh(0.01*(age_data[i]-skin_age[-1])))[-1])

        #input("###")

    return skin_age, skin_score


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
            skin_age_avg_sh.append(age_Sorted_Cluster_centroids[0]-weight*(input_data[i][1]-Sorted_Cluster_centroids[0][1])/0.82)

        skin_age.append((skin_age_avg_hyd[-1]+skin_age_avg_sh[-1])/2)
        
        skin_score.append(average_score+(30*np.tanh(0.08*(age_data[i]-skin_age[-1])))[-1])

    return skin_age, skin_score


def store_csv(skin_age_pred, skin_score_pred):
    with open('avg_skin_age'+str(datetime.datetime.now().strftime("%Y_%m_%d"))+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(skin_age_pred)
        writer.writerow(skin_score_pred)

if __name__ == '__main__':
    
    file_path = 'All_data/P1+P2_test_data.csv'
    age = readcsv(file_path, 'Age')
    avg_hydration = readcsv(file_path, 'Avg_hydration')
    avg_skinhealth = readcsv(file_path, 'Avg_skinhealth')

    x_avg_data = np.stack((avg_hydration, avg_skinhealth), axis=1)
    print ("x_avg_data = ", x_avg_data)
    avg_x_avg_data = np.mean((x_avg_data),axis=0)
    print ("avg_x_avg_data = ", avg_x_avg_data)

    age = np.stack(np.asarray(age).reshape(-1, 1))
    
    age_kmeans = KMeans(n_clusters=6, random_state=0).fit(age)
    age_Cluster_centroids = age_kmeans.cluster_centers_
    age_Sorted_Cluster_centroids = sorted(age_Cluster_centroids, key = lambda x : x[0])
    print ("age_Sorted_Cluster_centroids = ", age_Sorted_Cluster_centroids)
    age_Sorted_Cluster_centroids = np.stack(age_Sorted_Cluster_centroids, axis=1)
    print ("age_Sorted_Cluster_centroids = ", age_Sorted_Cluster_centroids)
    print ("age_Sorted_Cluster_centroids[0] = ", age_Sorted_Cluster_centroids[0])

    avg_hyd_kmeans = KMeans(n_clusters=6, random_state=0).fit(np.asarray(avg_hydration).reshape(-1, 1))
    avg_hyd_Cluster_centroids = avg_hyd_kmeans.cluster_centers_
    avg_hyd_Sorted_Cluster_centroids = sorted(avg_hyd_Cluster_centroids, key = lambda x : x[0], reverse=True)

    avg_skinh_kmeans = KMeans(n_clusters=6, random_state=0).fit(np.asarray(avg_skinhealth).reshape(-1, 1))
    avg_skinh_Cluster_centroids = avg_skinh_kmeans.cluster_centers_
    avg_skinh_Sorted_Cluster_centroids = sorted(avg_skinh_Cluster_centroids, key = lambda x : x[0], reverse=True)

    x_avg_data_k_centroids = np.concatenate((avg_hyd_Sorted_Cluster_centroids, avg_skinh_Sorted_Cluster_centroids), axis=1)
    x_avg_data_k_centroids = np.stack(x_avg_data_k_centroids)
    print ("x_avg_data_k_centroids = ", x_avg_data_k_centroids)

    weight = 0.3 # the larger weight value, the more vulnerable skin age will get. (change a little in hyd and oxy value, get a large difference in skin age.)
    
    skin_age_pred, skin_score_pred = skin_age_avg_classification(age, x_avg_data, x_avg_data_k_centroids, age_Sorted_Cluster_centroids[0], weight)
    store_csv(skin_age_pred, skin_score_pred)
    # store the results
    with open('test_avg_skin_age'+str(datetime.datetime.now().strftime("%Y_%m_%d"))+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        # load test csv data
        prefix_path = '/'
        file_path = ['01.csv', '02.csv', '03.csv', '04.csv', '05.csv', '06.csv', '07.csv']
        for file_name in file_path:   
            test_age = readcsv(prefix_path+file_name, 'Age')
            test_avg_hydration = readcsv(prefix_path+file_name, 'Avg3_Hydration')
            test_avg_skinhealth = readcsv(prefix_path+file_name, 'Avg3_Skinhealth')

            test_age = np.stack(np.asarray(test_age).reshape(-1, 1))
            test_x_data = np.stack((test_avg_hydration, test_avg_skinhealth), axis=1)
            test_skin_age_pred, test_skin_score_pred = skin_age_avg_classification(test_age, test_x_data, x_avg_data_k_centroids, age_Sorted_Cluster_centroids[0], weight)

            writer.writerow(test_skin_age_pred)
            writer.writerow(test_skin_score_pred)

    '''
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
        test_x_data = np.stack(([int(test_left_hydration)], [int(test_left_skinhealth)], [int(test_right_hydration)], [int(test_right_skinhealth)], [int(test_fore_hydration)], [int(test_fore_skinhealth)]), axis=1)
        print("test_x_data = ", test_x_data)
        test_skin_age_pred, test_skin_score_pred = skin_age_score_calculation([[int(test_age)]], test_x_data, x_data_k_centroids, weights_1, avg_x_data, age_Sorted_Cluster_centroids[0])
        print("test_skin_age_pred = ", test_skin_age_pred)
        input("============")
        print("test_skin_score_pred = ", test_skin_score_pred)
        input("===Press Enter to continue...===")
    '''













