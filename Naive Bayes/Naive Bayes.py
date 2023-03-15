import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="File Name")
args = parser.parse_args()
file=args.data
#file="/Users/maria/OneDrive/Documents/ML/Assignment/Programming Assignment 4/nb_new/Example.csv"
import math
data=open(file,"r")
data_list=list(line[:-1] for line in data)
for i in range (0,len(data_list)):
    data_list[i]=list(data_list[i].split(","))
    data_list[i]=[elem for elem in data_list[i]]
second_count=0
mean11_sum=0
mean12_sum=0
mean21_sum=0
mean22_sum=0
variance11_sum=0
variance12_sum=0
variance21_sum=0
variance22_sum=0
for i in range (0,len(data_list)):
    if i==0:
        first_class=data_list[i][0]
        first_count=1
        mean11_sum+=float(data_list[i][1])
        mean12_sum+=float(data_list[i][2])
    else:
        if data_list[i][0]!=first_class:
            second_count+=1
            mean21_sum+=float(data_list[i][1])
            mean22_sum+=float(data_list[i][2])
        else:
            first_count+=1
            mean11_sum+=float(data_list[i][1])
            mean12_sum+=float(data_list[i][2])
#probability        
p1=first_count/(first_count+second_count)
p2=second_count/(first_count+second_count)
#mean
mean11=(1/first_count)*mean11_sum
mean12=(1/first_count)*mean12_sum
mean21=(1/second_count)*mean21_sum
mean22=(1/second_count)*mean22_sum
#variance
for i in range (0,len(data_list)):
    if i==0:
        first_class=data_list[i][0]
        variance11_sum+=(float(data_list[i][1])-mean11)**2
        variance12_sum+=(float(data_list[i][2])-mean12)**2
    else:
        if data_list[i][0]!=first_class:
            variance21_sum+=(float(data_list[i][1])-mean21)**2
            variance22_sum+=(float(data_list[i][2])-mean22)**2
        else:
            variance11_sum+=(float(data_list[i][1])-mean11)**2
            variance12_sum+=(float(data_list[i][2])-mean12)**2
variance11=(1/(first_count-1))*variance11_sum
variance12=(1/(first_count-1))*variance12_sum 
variance21=(1/(first_count-1))*variance21_sum
variance22=(1/(first_count-1))*variance22_sum
print(str(mean11)+','+str(variance11)+','+str(mean12)+','+str(variance12)+','+str(p1))
print(str(mean21)+','+str(variance21)+','+str(mean22)+','+str(variance22)+','+str(p2)) 
#calculation of misclassification
miscount=0
for i in range (0,len(data_list)):
    p11=(1/math.sqrt(2*math.pi*variance11))*math.exp(-1*(((float(data_list[i][1])-mean11)**2)/(variance11*2)))
    p12=(1/math.sqrt(2*math.pi*variance12))*math.exp(-1*(((float(data_list[i][2])-mean12)**2)/(variance12*2)))
    p21=(1/math.sqrt(2*math.pi*variance21))*math.exp(-1*(((float(data_list[i][1])-mean21)**2)/(variance21*2)))
    p22=(1/math.sqrt(2*math.pi*variance22))*math.exp(-1*(((float(data_list[i][2])-mean22)**2)/(variance22*2)))
    vnb1=(p1*p11*p12)
    vnb2=(p2*p21*p22)
    if(vnb1>vnb2):
        if data_list[i][0]!=first_class:
            miscount+=1
    else:
        if data_list[i][0]==first_class:
            miscount+=1
print(miscount)