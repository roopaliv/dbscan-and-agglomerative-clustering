import argparse
import csv
import math
import numpy as np
import sys
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from ggplot import *
def get_coefficients(data_len, results, ground_truth):
    check = {True:1, False:0}
    derived_clusters = [[0 for x in range(data_len)] for y in range(data_len)]
    ground_clusters = [[0 for x in range(data_len)] for y in range(data_len)]
    for i in range(data_len):
        for j in range(i,data_len):
            derived_clusters[i][j] = derived_clusters[j][i] = check[results[i] == results[j]]
            ground_clusters[i][j] = ground_clusters[j][i] = check[ground_truth[i][1] == ground_truth[j][1]]
    
    TP, TN, FP, FN = (0,)*4
    same_results = {False:'TP += 1', True:'TN += 1'}
    different_results = {False:'FP += 1', True:'FN += 1'}

    for i in range(data_len):
        for j in range(data_len):
            if derived_clusters[i][j] != ground_clusters[i][j]:
                exec(different_results[(derived_clusters[i][j] == 0 and ground_clusters[i][j] == 1)])
            else:
                exec(same_results[(derived_clusters[i][j] == 0)])

    rand_index = (float) (TP+TN) / (TP+FP+FN+TN)
    jaccard_coeff = (float) (TP) / (TP+FP+FN)
    return rand_index, jaccard_coeff
#calculate and print Jaccard Index
def calc_jc_index(obtained_mat, given_mat):
    similar = np.sum(np.logical_and(obtained_mat,given_mat))
    non_similar = np.sum(np.logical_or(obtained_mat,given_mat))
    return (similar/ non_similar)

# Convert labels to 2d sparse matrix
def convert_label_to_sparse(output_cluster):
    N = len(output_cluster)
    ouput_mat = np.zeros((N  , N))
    for i in range(0,N):
        for j in range(0, N):
            if(output_cluster[i] == output_cluster[j]):
                ouput_mat[i,j] = 1
        
    return ouput_mat

def read_file(file_path):
    with open(file_path) as file:
        return list(csv.reader(file, delimiter="\t"))
    
''' Credit for the plot : https://stackoverflow.com/questions/21654635/scatter-plots-in-pandas-pyplot-how-to-plot-by-category '''    

def plot_graphs(l1, l2, y, name, data_set):
    df = pd.DataFrame(dict(x=l1, y=l2, label=y))
    g = ggplot(aes(x='x', y='y', color='label'), data=df) + geom_point(size=50) + theme_bw()
    g.save(name+"_"+data_set+".jpg")


def recursive_dendrogram(data_distances):
    global total_clusters, clusters, current_count

    min_distance=1000
    min_first_d, min_second_d=(-1,)*2
 
    for k in range (0, current_count):
        for l in range (0, current_count):
            if(k!=l):
                if(min_distance>data_distances[k][l]):
                    min_distance=data_distances[k][l]
                    min_first_d=k
                    min_second_d=l

    if min_first_d==-1 or min_second_d==-1:
        return

    clusters[total_clusters]=[]
    clusters[total_clusters].append(data_distances[min_first_d][current_count])
    clusters[total_clusters].append(data_distances[min_second_d][current_count])
    data_distances[min_first_d][current_count]=total_clusters
    total_clusters+=1

    for m in range (0, current_count):
        if(m!=min_first_d or m!=min_second_d):
            temp_min = min(data_distances[min_first_d][m],data_distances[min_second_d][m])
            data_distances[min_first_d][m]= temp_min
            data_distances[m][min_first_d]=data_distances[min_first_d][m]

    data_distances=np.delete(data_distances,min_second_d,0)
    data_distances=np.delete(data_distances,min_second_d,1)
    current_count-=1

    recursive_dendrogram(data_distances)

    return

if __name__ == "__main__":

    files = ['new_dataset_2.txt']
    for file in files:
        data = read_file(file)
        data_len = len(data)
        num_clusters=3
        dimensions = len(data[0])
        dimensions = dimensions-2

        new_data_set=np.zeros((data_len, dimensions+3))
        for i in range (0, data_len):
            for j in range (0, dimensions+2):
                new_data_set[i][j]=data[i][j]

        data_distances= euclidean_distances(new_data_set[0:data_len,2:dimensions+2], new_data_set[0:data_len,2:dimensions+2])
        stacking = 'np.append(data_distances,np.ones([len(data_distances),1]),1)'
        data_distances=np.append(data_distances,np.ones([len(data_distances),1]),1)
        data_distances=np.append(data_distances,np.ones([len(data_distances),1]),1)

        for j in range (0, data_len):
            data_distances[j][data_len]=j+1
            data_distances[j][data_len+1]=-1


        total_clusters=data_len+1
        clusters={}
        current_count=data_len

        recursive_dendrogram(data_distances)
        clusterinfo=np.zeros(shape=(total_clusters-num_clusters+2,1))
        clusterinfo[0]=1
        first=total_clusters-num_clusters
        current_cluster=1
        cluster_cloud={}

        while first>0:
            cluster_numbers=[]
            clusterinfo[first]=1
            while len(clusters[first])>0:
                num_cluster=int(clusters[first].pop())
                clusterinfo[num_cluster]=1
                if num_cluster>data_len:
                    for j in range (0, len(clusters[num_cluster])):
                        clusters[first].append(clusters[num_cluster][j])
                else:
                    cluster_numbers.append(num_cluster)
            cluster_cloud[current_cluster]=cluster_numbers
            current_cluster+=1

            while clusterinfo[first]==1:
                if first>data_len+1:
                    first=first-1
                else:
                    first=0
                    break

        this_cluster=1

        for c in cluster_cloud:
            for j in range(0, len(cluster_cloud[c])):
                new_data_set[cluster_cloud[c][j]-1][dimensions+2]=this_cluster
            this_cluster=this_cluster+1

       
        results = new_data_set[:,-1]
        results = [int(i) for i in results]
        print(results)
        results_sparse = convert_label_to_sparse(results)

        ground_truth_sparse = convert_label_to_sparse(np.array(data)[:,1])
        
        rand_index, jaccard_coeff = get_coefficients( data_len, results,data)
        
        jaccard_coeff = calc_jc_index( results_sparse, ground_truth_sparse)
        pca = PCA(n_components=2)
        input_vector = pca.fit_transform(np.array(data)[:,2:np.array(data).shape[1]])
        plot_graphs(input_vector[:,0],input_vector[:,1], results, "dbscan",file)
    
    
        print ("####### Results for : "+ file + " ##############")
        print ( " Jaccard Coeff:"+ str(jaccard_coeff) + " rand index: "+ str(rand_index) )
        