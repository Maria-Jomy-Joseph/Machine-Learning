import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="File Name")
parser.add_argument("-e", "--eta", help="Learning Rate")
parser.add_argument("-t", "--iterations", help="iterations")
args = parser.parse_args()
#received values
eta=float(args.eta)
iterations=int(args.iterations)
file=args.data
"""
iterations=2
eta=0.2
file="/Users/maria/OneDrive/Documents/ML/Assignment/Programming Assignment 3/Gauss_new/Gauss3.csv"
"""
data=open(file,"r")

#initializing weights
w_bias_h1=0.2
w_a_h1=-0.3
w_b_h1=0.4
w_bias_h2=-0.5
w_a_h2=-0.1
w_b_h2=-0.4
w_bias_h3=0.3
w_a_h3=0.2
w_b_h3=0.1
w_bias_o=-0.1
w_h1_o=0.1
w_h2_o=0.3
w_h3_o=-0.4
#--------printing first two lines
#headers
print("a,b,h1,h2,h3,o,t,delta_h1,delta_h2,delta_h3,delta_o,w_bias_h1,w_a_h1,w_b_h1,w_bias_h2,w_a_h2,w_b_h2,w_bias_h3,w_a_h3,w_b_h3,w_bias_o,w_h1_o,w_h2_o,w_h3_o")
#printing initialized weights
print("-,-,-,-,-,-,-,-,-,-,-"+','+str(w_bias_h1)+','+str(w_a_h1)+','+str(w_b_h1)+','+str(w_bias_h2)
              +','+str(w_a_h2)+','+str(w_b_h2)+','+str(w_bias_h3)+','+str(w_a_h3)+','+str(w_b_h3)
              +','+str(w_bias_o)+','+str(w_h1_o)+','+str(w_h2_o)+','+str(w_h3_o))
#data processing
data_list=list(line[:-1] for line in data)
for i in range (0,len(data_list)):
    data_list[i]=list(data_list[i].split(","))
    data_list[i]=[elem for elem in data_list[i]]
import numpy as np
for j in range (0,iterations):
    for i in range (0,len(data_list)):
        a = float(data_list[i][0])
        b = float(data_list[i][1])
        t = int(data_list[i][2])
        #output calculation
        h1 = 1/(1+np.exp(-1 * ((a*w_a_h1) + (b*w_b_h1) + (w_bias_h1))))
        h2 = 1/(1+np.exp(-1 * ((a*w_a_h2) + (b*w_b_h2) + (w_bias_h2))))
        h3 = 1/(1+np.exp(-1 * ((a*w_a_h3) + (b*w_b_h3) + (w_bias_h3))))
        o = 1/(1+np.exp(-1 * ((h1*w_h1_o) + (h2*w_h2_o) + (h3*w_h3_o) + (w_bias_o))))
        #error calculation according to sigmoid activation function
        delta_o = o * (1-o) * (t-o)
        delta_h3 = h3 * (1-h3) * (w_h3_o*delta_o)
        delta_h2 = h2 * (1-h2) * (w_h2_o*delta_o)
        delta_h1 = h1 * (1-h1) * (w_h1_o*delta_o)
        #weight updation
        w_bias_h1 = w_bias_h1 + (eta*delta_h1*1)
        w_a_h1 = w_a_h1 + (eta*delta_h1*a)
        w_b_h1 = w_b_h1 + (eta*delta_h1*b)
        w_bias_h2 = w_bias_h2 + (eta*delta_h2*1)
        w_a_h2 = w_a_h2 + (eta*delta_h2*a)
        w_b_h2 = w_b_h2 + (eta*delta_h2*b)
        w_bias_h3 = w_bias_h3 + (eta*delta_h3*1)
        w_a_h3 = w_a_h3 + (eta*delta_h3*a)
        w_b_h3 = w_b_h3 + (eta*delta_h3*b)
        w_bias_o = w_bias_o + (eta*delta_o)
        w_h1_o = w_h1_o + (eta*delta_o*h1)
        w_h2_o = w_h2_o + (eta*delta_o*h2)
        w_h3_o = w_h3_o + (eta*delta_o*h3)
        #output printing
        print(str(a)+','+str(b)+','+str(h1)+','+str(h2)+','+str(h3)+','+str(o)+','+str(t)
              +','+str(delta_h1)+','+str(delta_h2)+','+str(delta_h3)+','+str(delta_o)
              +','+str(w_bias_h1)+','+str(w_a_h1)+','+str(w_b_h1)+','+str(w_bias_h2)
              +','+str(w_a_h2)+','+str(w_b_h2)+','+str(w_bias_h3)+','+str(w_a_h3)+','+str(w_b_h3)
              +','+str(w_bias_o)+','+str(w_h1_o)+','+str(w_h2_o)+','+str(w_h3_o))
        