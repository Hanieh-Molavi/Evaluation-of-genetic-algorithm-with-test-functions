import numpy as np
import math
import statistics
from geneticalgorithm import geneticalgorithm as ga
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def Var_and_Std(data, ddof=0):

    variance=[]
    std=[]

    n = len(data)
    mean = sum(data) / n
    variance.append(sum((x - mean) ** 2 for x in data) / (n - ddof))
    
    std.append(math.sqrt(variance[0][0]))
    std.append(math.sqrt(variance[0][1]))

    return variance, std


def tpl_to_lst(m):
    return list(m)



def convert_to_list(lst):

    resultList=[]

    for o in range(0,len(lst)):
        d=lst[o]
        resultList.append(d.get('variable'))
                        
    return resultList




def find_best(solution,mini):

    minimum1=math.inf
    minimum2=math.inf
    destination=[]
    best_solution=0

    for a in range(0,len(solution)):

        temp1 = solution[a]
        
        for b in range(0,len(temp1)):
            destination.append(temp1[b] - mini[b])

    destination = list(map(float, destination))
    destination = np.array_split(destination, len(solution)) 


    for c in range(0,len(solution)):
        
        temp2 = solution[a]
        if temp2[0] < minimum1 and temp2[1] < minimum2:

            minimum1 = temp2[0]
            minimum2 = temp2[1]

            best_solution=solution[a]


    return best_solution , destination



def epsilon_destince(destination,epsilon):

    ture_solutions=[]
    t=0
    ture_solution_count=0
    sum_=0
    mean_=0
    for c in range(0,len(destination)):

        temp2 = destination[c]
        
        for d in range(0,len(temp2)):

            if temp2[d] <= epsilon:
                t+=1

            if t==2:
                ture_solution_count+=1
                ture_solutions.append(temp2)

    return  ture_solution_count , ture_solutions



def convert(sec):
   sec = sec % (24 * 3600)
   hour = sec // 3600
   sec %= 3600
   min = sec // 60
   sec %= 60
   return "%02d:%02d:%02d" % (hour, min, sec) 



def eggholder(X):

    dim=len(X) 

    return (-(X[1] + 47) * np.sin(np.sqrt(abs(X[0]/2 + (X[1] + 47)))) -X[0] * np.sin(np.sqrt(abs(X[0] - (X[1] + 47)))))




sol=[]
mean_of_runtime=[]


print("\n========* Eggholder_model *========\n") 

for i in range(0,20):

    print('\n_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_iterations ',i,'-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-\n')
    
    st=time.time()
    ct=time.process_time()    

    varbound=np.array([[450,513]]*2)
    eggholder_model=ga(function=eggholder,dimension=2,variable_type='float',variable_boundaries=varbound)
                        
                        
    eggholder_model.run()
    sol.append(eggholder_model.output_dict)

    st1=time.time()
    ct1=time.process_time()

    mean_of_runtime.append(st1-st)

    print("\nRunTime :",convert(st1-st))
    print("\nCpuTime :",convert(ct1-ct))                     


print("\n=============================================================\n ")


min_=[512,404.2319]

solution=convert_to_list(sol)
best_, destination = find_best(solution , min_)

count, ture_sol = epsilon_destince(destination,0.00000001)
var, std = Var_and_Std(solution)

print('\n Mean of RunTime:',sum(mean_of_runtime)/len(mean_of_runtime),' -> round : ',convert(sum(mean_of_runtime)/len(mean_of_runtime)))
print('\n Mean of Error:',sum(destination)/len(destination))

print('\n Variance for x:',var[0][0],'[ round:',round(var[0][0]),']','\n variance for y:',var[0][1],'[ round:',round(var[0][1]),']')
print('\n Standard deviation for x:',var[0][0],'[ round:',round(var[0][0]),']','\n Standard deviation for y:',var[0][1],'[ round:',round(var[0][1]),']')

print("\n Optimum point beale function:",min_)
print("\n Best solution with GA :",best_)

print("\n Distance of best solution from optimum point:",best_ - min_)
print('\n Correct solutions with error 10e-7:', (count/20)*100 ,'%') 
