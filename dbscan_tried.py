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

def visit_unvisited_points(data_len, distance_matrix, epsilon, min_points):
    total_clusters=0
    is_visited = [0 for i in range(data_len)]
    results = [0 for i in range(data_len)]
    for index in range(data_len):
        if is_visited[index] == 0:
            neigh_list = get_region(index, epsilon, data_len, distance_matrix)
            is_visited[index] = 1
            #check if it is a noise point
            print(min_points)
            if len(neigh_list) < min_points:
                results[i] = -1
            else:
                total_clusters = grow_cluster(index, neigh_list, total_clusters,  data_len, distance_matrix, results, is_visited, epsilon, min_points)
    return results, total_clusters

def get_distance_matrix(data, data_len):
    distance_matrix = [[0 for i in range(data_len)] for j in range(data_len)]
    for i in range(data_len):
        for j in range(data_len):
            distance_matrix[i][j] = get_distance(data[i][2:], data[j][2:])
    return distance_matrix

def grow_cluster(_id, neighbours, total_clusters,  data_len, distance_matrix, results, is_visited, epsilon, min_points):
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
    print(total_clusters)
    return total_clusters

def read_file(file_path):
    with open(file_path) as file:
        return list(csv.reader(file, delimiter="\t"))

def run_dbscan(file_path, epsilon, min_points):
    epsilon = float(epsilon/ 100.00)
    data = read_file(file_path)
    data_len = len(data)
    distance_matrix =get_distance_matrix(data, data_len)
    results, total_clusters = visit_unvisited_points(data_len, distance_matrix, epsilon, min_points)
    rand_index, jaccard_coeff = get_coefficients(data_len, results, data)
    return rand_index, jaccard_coeff, total_clusters
   
def optimize(file_path):
    optimization_results = []
    for min_points in range(4,5):
        for epsilon in range(1, 20):
            epsilon=(epsilon/10.00)
            rand_index, jaccard_coeff, total_clusters = run_dbscan(file_path, epsilon, min_points)
            optimization_results.append([min_points, epsilon, rand_index, jaccard_coeff, total_clusters])
            print "####### Current Jaccard for : "+ str(optimization_results[-1]) + " ##############"
    optimization_results.sort(key=lambda x: x[3])
    print "####### Highest Jaccard for : "+ str(optimization_results[-1]) + " ##############"
    print "####### Lowest Jaccard for : "+ str(optimization_results[0]) + " ##############"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Usage of DBSCAN')
    parser.add_argument('--data', default='cho', type=str, help='type of dataset (cho or iyer)')
    parser.add_argument('--optimize', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default = False, help='True | False')

    args = parser.parse_args()
    file_path = args.data+'.txt'
    if args.optimize == True:
        optimize(file_path)
    else:
        if args.data == 'cho':
            epsilon = 0.81
            min_points = 4
        else:
            epsilon = 0.3
            min_points = 4
        rand_index, jaccard_coeff, total_clusters = run_dbscan(file_path, epsilon, min_points)
        print "####### Results for : "+ args.data + " ##############"
        print "Number of Clusters:" + str(total_clusters)+ " Jaccard Coeff:"+ str(jaccard_coeff) +  " Rand Index: " + str(rand_index) + " with epsilon: "+ str(epsilon) + " and min points: " + str(min_points)

   