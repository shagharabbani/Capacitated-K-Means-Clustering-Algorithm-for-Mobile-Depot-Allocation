from turtle import Shape
import torch
import numpy as np
import random
import sys
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
#from Torch.datasets import dataset2
#from Torch.datasets import dataset4
#from Torch.datasets import dataset2
from datasets import dataset2
import time


device_gpu = torch.device('cuda')
device_cpu = torch.device('cpu')

# Choosing `num_centers` random data points as the initial centers
def random_init(dataset, num_centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    used = torch.zeros(num_points, dtype=torch.long)
    indices = torch.zeros(num_centers, dtype=torch.long)
    for i in range(num_centers):
        while True:
            cur_id = random.randint(0, num_points - 1)
            if used[cur_id] > 0:
                continue
            used[cur_id] = 1
            indices[i] = cur_id
            break
    indices = indices.to(device_cpu)
    centers = torch.gather(dataset, 0, indices.view(-1, 1).expand(-1, dimension))
    return centers

def relu(x):
    if x>0:
        return x
    else:
        return 0

# Compute for each data point the closest center
def compute_codes(customer_xy, centers,demand,capacity):
    num_points = customer_xy.size(0)
    dimension = customer_xy.size(1)
    num_centers = centers.size(0)
    codes = torch.zeros(num_points, dtype=torch.long, device=device_cpu)
    centers_t = torch.transpose(centers, 0, 1)
    centers_norms = torch.sum(centers ** 2, dim=1).view(1, -1)
    dataset_norms = torch.sum(customer_xy ** 2, dim=1).view(-1, 1)
    distances = torch.mm(customer_xy, centers_t)
    distances *= -2.0
    distances += dataset_norms
    distances += centers_norms
    
    lmbd = 4000
    cost=torch.zeros(num_points,num_centers, dtype=torch.float, device=device_cpu)    
    load=torch.zeros(num_centers, dtype=torch.float, device=device_cpu)
    # for i in range(0,num_points):
    #     invest_point=torch.topk(demand,num_points).indices[i]
    #     if (demand == demand[invest_point]).nonzero(as_tuple=False).squeeze(0).shape[0]>1:
    #         same_dem_idx=(demand == demand[invest_point]).nonzero(as_tuple=False).squeeze()
    #         max_dist=torch.topk(torch.sum(distances[same_dem_idx].squeeze(),dim=1),1).indices
    #         invest_point=same_dem_idx[max_dist].squeeze()
    #_,_,demand=dataset1()
    dem=demand.detach().clone()
    mask=torch.zeros(num_points,dtype=torch.bool)
    while False in mask:
        invest_point=torch.topk(dem,num_points).indices[0]
        if (dem == dem[invest_point]).nonzero(as_tuple=False).squeeze(0).shape[0]>1:
            same_dem_idx=(dem == dem[invest_point]).nonzero(as_tuple=False).squeeze()
            max_dist=torch.topk(torch.sum(distances[same_dem_idx],dim=1),1).indices
            invest_point=same_dem_idx[max_dist].squeeze()
        mask[invest_point]=True
        
        for j in range(0,num_centers):
            cost[invest_point,j]=distances[invest_point,j]*0.5+lmbd*relu(demand[invest_point]-(3.0-load[j]))
        _, min_ind = torch.min(cost[invest_point], dim=0)
        load[min_ind]=load[min_ind]+demand[invest_point]
        capacity[min_ind]=3.0-load[min_ind]
        dem[invest_point]=0
           

    #_, min_ind = torch.min(cost, dim=1)
    codes = torch.min(cost,dim=1).indices
    chosen_cost=torch.sum(torch.min(cost,dim=1).values)
    return codes,load,chosen_cost,distances
   

# Compute new centers as means of the data points forming the clusters
def update_centers(customer_xy, codes, num_centers):
    num_points = customer_xy.size(0)
    dimension = customer_xy.size(1)
    centers = torch.zeros(num_centers, dimension, dtype=torch.float, device=device_cpu)
    cnt = torch.zeros(num_centers, dtype=torch.float, device=device_cpu)
    centers.scatter_add_(0, codes.view(-1, 1).expand(-1, dimension), customer_xy)
    cnt.scatter_add_(0, codes, torch.ones(num_points, dtype=torch.float, device=device_cpu))
    # Avoiding division by zero
    # Not necessary if there are no duplicates among the data points
    cnt = torch.where(cnt > 0.5, cnt, torch.ones(num_centers, dtype=torch.float, device=device_cpu))
    centers /= cnt.view(-1, 1)
    return centers

def cluster(customer_xy,demand,num_centers,capacity):
    centers = random_init(customer_xy, num_centers)
    codes,_,chosen_cost,_ = compute_codes(customer_xy, centers,demand,capacity)
    num_iterations = 0
    while True:
        sys.stdout.write('.')
        sys.stdout.flush()
        num_iterations += 1
        centers = update_centers(customer_xy, codes, num_centers)
        # _,_,demand=dataset1()
        new_codes,_,new_chosen_cost,_ = compute_codes(customer_xy, centers,demand,capacity)
        # Waiting until the clustering stops updating altogether
        # This is too strict in practice
        if new_chosen_cost>=chosen_cost:
        #if torch.equal(chosen_cost,new_chosen_cost):
            sys.stdout.write('\n')
            print('Converged in %d iterations' % num_iterations)
            break
        chosen_cost = new_chosen_cost
        codes=new_codes
    return centers, codes, chosen_cost

def capacitatedkmeans(customer_xy,pos_depot_xy,demand,capa):
    sum_demand=torch.sum(demand)
    num_centerstensor=torch.ceil(sum_demand/capa).int()
    num_centers=num_centerstensor.item()
    capacity=capa*torch.ones(num_centers)
    print('Starting clustering')
    centers, _,_ = cluster(customer_xy,demand,num_centers,capacity)
    dist_cen_posdep=torch.cdist(centers,pos_depot_xy,p=2)
    depots=torch.min(dist_cen_posdep,1).indices
    depot_xy=pos_depot_xy[depots,:]
    fin_codes,load,chosen_cost,distances=compute_codes(customer_xy, depot_xy,demand,capacity)
    codes_index=fin_codes[:,None]               #check shavad
    codes_label = torch.gather(input = depots[:,None], dim = 0, index = codes_index).squeeze()
    return depot_xy,codes_index,depots,codes_label,load,chosen_cost,distances,codes_index

def distancess(customer_xy,centers):
    centers_t = torch.transpose(centers, 0, 1)
    centers_norms = torch.sum(centers ** 2, dim=1).view(1, -1)
    dataset_norms = torch.sum(customer_xy ** 2, dim=1).view(-1, 1)
    distances = torch.mm(customer_xy, centers_t)
    distances *= -2.0
    distances += dataset_norms
    distances += centers_norms

    return distances

# IDEA
def capacitatedkmeansbatch(batch_size,customer_xy,pos_depot_xy,demand,capa_depot,capa_car):
    M=0.9
    temp=M*torch.ones([batch_size,pos_depot_xy.shape[1],2])
    num_car=torch.zeros(batch_size,pos_depot_xy.shape[1])
    total_dem=torch.zeros(batch_size,pos_depot_xy.shape[1])
    for bs in range(batch_size):
            depot_xy,codes_index,depots=capacitatedkmeans(customer_xy[bs],pos_depot_xy[bs],demand[bs],capa_depot)
            temp[bs][0:depot_xy.shape[0]][:]=depot_xy.squeeze()
            for dep in range(pos_depot_xy.shape[1]):
                total_dem[bs,dep]=torch.sum(demand[bs,(codes_index.squeeze()==dep).nonzero(as_tuple=False)])
                num_car[bs,dep]=torch.ceil(total_dem[bs,dep]/capa_car).int()

    return temp,num_car
            


if __name__ == '__main__':
    
    t1=time.time()
    customer_xy,pos_depot_xy,demand = dataset2()
    print("demand: "+str(demand))
    print("total demand: " +str(torch.sum(demand)))

    depot_xy,codes_index,depots,codes_label,load,chosen_cost,distances,codes_index=capacitatedkmeans(customer_xy,pos_depot_xy,demand,capa=3.0)
    t2=time.time()
    time_run=t2-t1
    print('run time: ' + str(time_run))
    favasel=distancess(customer_xy,pos_depot_xy)
    print("favasel"+str(favasel))

    print("labels: "+str(codes_label))
    #print("codes_index: " +str(codes_index))
    print("final depots: "+str(depots))
#    print("final depots_xy: "+str(depot_xy))
    print("load:" + str(load))
    print("chosen_cost: "+str(chosen_cost+200*len(load)))
    #plt.plot(depot_xy[:,0],depot_xy[:,1],'X')
    #plt.plot(centers[:,0],centers[:,1],'o')
    plt.plot(pos_depot_xy[:,0],pos_depot_xy[:,1],'s',markerfacecolor='none',markersize=10)
    
    colors = ['green','blue','cyan','purple','black','orange','red','brown']

    plt.scatter(customer_xy[:,0], customer_xy[:,1], c=codes_label, cmap=matplotlib.colors.ListedColormap(colors),s=5)
    plt.scatter(depot_xy[:,0],depot_xy[:,1],c=depots,cmap=matplotlib.colors.ListedColormap(colors),s=35,marker='X')
    
    plt.show()
