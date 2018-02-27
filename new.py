import argparse
import csv
import math
import numpy as np
import sys
from sklearn.decomposition import PCA
import pandas as pd
from ggplot import *

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

def get_region(_id, epsilon, data_len, distance_matrix):
    neigh_list = []
    for index in range(data_len):
        #check if point is within epsilon distance
        if distance_matrix[_id][index] < epsilon and _id != index:
          neigh_list.append(index)
    return neigh_list

def get_distance(recored_one, recored_two):
    distance = 0.00
    for index in range(len(recored_one)):
        distance +=math.pow((float(recored_one[index]) - float(recored_two[index])),2)
    distance = math.sqrt(distance)
    return distance  

def visit_unvisited_points(data_len, distance_matrix, epsilon, total_clusters):
    is_visited = [0 for i in range(data_len)]
    results = [0 for i in range(data_len)]
    for index in range(data_len):
        if is_visited[index] == 0:
            neigh_list = get_region(index, epsilon, data_len, distance_matrix)
            is_visited[index] = 1
            #check if it is a noise point
            if len(neigh_list) < min_points:
                results[index] = -1
            else:
                total_clusters = grow_cluster(index, neigh_list, total_clusters,  data_len, distance_matrix, results, is_visited, epsilon)
    return results, total_clusters

def get_distance_matrix(data, data_len):
    distance_matrix = [[0 for i in range(data_len)] for j in range(data_len)]
    for i in range(data_len):
        for j in range(data_len):
            distance_matrix[i][j] = get_distance(data[i][2:], data[j][2:])
    return distance_matrix

def grow_cluster(_id, neighbours, total_clusters,  data_len, distance_matrix, results, is_visited, epsilon):
    total_clusters += 1        
    results[_id] = total_clusters
    for index in range(len(neighbours)):
        current_neighbour = neighbours[index]
        if is_visited[current_neighbour] == 0:
            is_visited[current_neighbour] = 1
            recursive_negh = get_region(current_neighbour,epsilon,  data_len, distance_matrix)
            if len(recursive_negh) >= min_points:
                neighbours.extend(recursive_negh)
        if results[current_neighbour] == 0:
            results[current_neighbour] = total_clusters
    return total_clusters

def read_file(file_path):
    with open(file_path) as file:
        return list(csv.reader(file, delimiter="\t"))
    
''' Credit for the plot : https://stackoverflow.com/questions/21654635/scatter-plots-in-pandas-pyplot-how-to-plot-by-category '''    

def plot_graphs(l1, l2, y, name, data_set):
    df = pd.DataFrame(dict(x=l1, y=l2, label=y))
    g = ggplot(aes(x='x', y='y', color='label'), data=df) + geom_point(size=50) + theme_bw()
    g.save(name+"_"+data_set+".jpg")

if __name__ == "__main__":

    '''parser = argparse.ArgumentParser(description='Usage of DBSCAN')
    parser.add_argument('--data', default='cho', type=str, help='type of dataset (cho or iyer)')
    args = parser.parse_args()
    file_path = args.data+'.txt'  '''
    files = ['iyer.txt','cho.txt','new_dataset_1.txt']
    for file in files:
        if (file == 'cho.txt'):
            epsilon = 0.8#0.9
            min_points = 3#5 or 6
        elif (file == 'iyer.txt'):
            epsilon = 1.9
            min_points = 9
        else:
            epsilon = 2.3
            min_points = 4
    
    
    
        data = read_file(file)
        data_len = len(data)
        distance_matrix =get_distance_matrix(data, data_len)
        total_clusters=0
        results, total_clusters = visit_unvisited_points(data_len, distance_matrix, epsilon, total_clusters)
        results_sparse = convert_label_to_sparse(results)
        #ground_t = data[,][1]
        ground_truth_sparse = convert_label_to_sparse(np.array(data)[:,1])
        
        jaccard_coeff = calc_jc_index( results_sparse, ground_truth_sparse)
        
        pca = PCA(n_components=2)
        input_vector = pca.fit_transform(np.array(data)[:,2:np.array(data).shape[1]])
        plot_graphs(input_vector[:,0],input_vector[:,1], results, "dbscan",file)
    
    
        print ("####### Results for : "+ file + " ##############")
        print ("Number of Clusters:" + str(total_clusters)+ " Jaccard Coeff:"+ str(jaccard_coeff) + " with epsilon: "+ str(epsilon) + " and min points: " + str(min_points))
