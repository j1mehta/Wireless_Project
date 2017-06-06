import pandas as pd
import math
from operator import add


fname = "/Users/jatin/Documents/MATLAB/LocalMNormalization/mixed_vid_pkt_frame_map.txt"
pktLossTrace_path = "/Users/jatin/Documents/MATLAB/LocalMNormalization/results_mu_100/mixed_corrupted_mu_100.csv"
nb_frames = 26

with open(fname) as f:
    content = f.readlines()
    
packetTrace = pd.read_csv(pktLossTrace_path, header = None, sep=';')

content = [int(x.strip()) for x in content] 

lst = [0] * 26
for i in content:
    lst[i] += 1
#List of start packet and end packet sq. number tuple
sP_eP_tuple_list = []
count_prev = 0
flag_change = 0
start = 0
k = 0
try:
    for i in range(len(content)+1):
        pkt = content[i]
        if(pkt != content[i+1]):
            t = []
            k += lst[count_prev]
            end = k
            pkt_range_per_frame = [start, end]
            t.append(pkt_range_per_frame)
            sP_eP_tuple_list.append(t)
            start = i+2
            count_prev += 1
except:
    print("Damn")
    
abc = sP_eP_tuple_list[-1][0][1]
t1 = [[abc+1, i+1]]
sP_eP_tuple_list.append(t1)


packet_lost_per_frame = [0] * (nb_frames)
for p in range(1,len(packetTrace)):
        packet_lost_per_frame[int(packetTrace[1][p])] += 1

percentage_20 = [0,0,0,0,0]
input_vector = []
for i in range(nb_frames):
    l = []
    input_vector.append([])
count_p = 0
for i in range(1,len(packetTrace)):
    frame_no = int(packetTrace[1][i])
    sq_no_pkt_lost = int(packetTrace[0][i])
    print(frame_no, sq_no_pkt_lost, i)
    range_of_pkt_for_frame = sP_eP_tuple_list[frame_no]
    start = int(range_of_pkt_for_frame[0][0])
    end = int(range_of_pkt_for_frame[0][1])
    print(start, end)
    percentage = 100*((sq_no_pkt_lost-start)/(end-start))
    k = math.floor(percentage/20)
    percentage_20[k] += 1
    input_vector[frame_no].append(percentage_20)
    percentage_20 = [0,0,0,0,0]   
    print('\n')


frame_packet_trend = []
for i in range(nb_frames):
    l = []
    frame_packet_trend.append([])
count_fr = 0
for each in input_vector:
    if(len(each)):
        first = each[0]
        for i in (range(1,len(each))):
            first = list(map(add, first, each[i]))
        frame_packet_trend[count_fr].append(first)
    else:
        frame_packet_trend[count_fr].append(each)
    count_fr += 1
  
        
