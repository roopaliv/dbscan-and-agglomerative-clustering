import argparse
import csv
import math
import numpy as np
import sys

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Usage of DBSCAN')
    parser.add_argument('--data', default='cho', type=str, help='type of dataset (cho or iyer)')
    args = parser.parse_args()
    file_path = args.data+'.txt'
    if args.data == 'cho':
        epsilon = 0.8#0.9
        min_points = 3#5 or 6
    elif  args.data == 'iyer':
        epsilon = 1.5
        min_points = 8
    elif args.data == 'new_dataset_1':
        epsilon = 1.9
        min_points = 9


    data = read_file(file_path)
    data_len = len(data)
    distance_matrix =get_distance_matrix(data, data_len)
    total_clusters=0
    results, total_clusters = visit_unvisited_points(data_len, distance_matrix, epsilon, total_clusters)
    rand_index, jaccard_coeff = get_coefficients(data_len, results, data)
    print(results)


    print "####### Results for : "+ args.data + " ##############"
    print "Number of Clusters:" + str(total_clusters)+ " Jaccard Coeff:"+ str(jaccard_coeff) +  " Rand Index: " + str(rand_index) + " with epsilon: "+ str(epsilon) + " and min points: " + str(min_points)
