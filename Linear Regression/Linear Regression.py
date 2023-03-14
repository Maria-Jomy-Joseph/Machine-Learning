import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="File Name")
parser.add_argument("-e", "--eta", help="Learning Rate")
parser.add_argument("-t", "--threshold", help="Threshold")
args = parser.parse_args()
#received values
eta=float(args.eta)
threshold=float(args.threshold)
file=args.data

data=open(file,"r")

#preprocessing data to x and y
x=[]
y_target=[]
weight=[]
sse=[]
data_list=list(line[:-1] for line in data)
for i in range (0,len(data_list)):
    data_list[i]=list(data_list[i].split(","))
    data_list[i]=[1]+[float(elem) for elem in data_list[i]]
    x.append(data_list[i][:-1])
    y_target.append(data_list[i][-1:])
#initiating weights to zero for n number of weights
weight=[0]*(len(x[0]))

iteration=0
while True:
    if iteration>2:
        sse_diff=sse[-2]-sse[-1]
        if sse_diff<threshold:
            break
            
    #calculating f(x)/y_predicted, w(i)*x(i)
    
    y_predicted=[]
    for i in range (0,len(x)):
        y=0
        for j in range (0,len(x[i])):
            y+=x[i][j]*weight[j]
        y_predicted.append(y)
        
    #calculating error y_target - y_predicted  and gradient
    
    sse_iteration=0
    gradient=[0]*(len(x[0]))
    for i in range (0,len(x)):
        e=y_target[i][0]-y_predicted[i]
        for j in range (0,len(x[i])):
            gradient[j]+=x[i][j]*e    
        sse_iteration+=(e ** 2)
    sse.append(sse_iteration)
    #output
    print(str(iteration),end=',')
    for j in range (0,len(x[i])):
        print(str(weight[j]),end=',')
    print(str(sse[-1]))
    #weight updation
    for j in range (0,len(x[i])):
        weight[j]=weight[j]+(eta*gradient[j])
    iteration+=1
