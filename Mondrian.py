# using Mondrian alg to create K-Anonymity table 
import pandas as pd
import numpy as np
import time
import random

# time record
time_start=time.time()

#QI={age, education_num}
# define age:1 ; education_num:0 
# para init
k = 10

# read adult as .csv , get data whose type is dataframe
data = pd.read_csv('adult.csv',encoding = 'GBK', engine="python",header = None)
data.columns = ["age", "work_class" ,"final_weight" ,"education" ,"education_num" ,"marital_status" ,"occupation" ,"relationship" ,"race" ,"sex" ,"capital_gain" ,"capital_loss" ,"hours_per_week" ,"native_country" ,"class"]
DataTupleNum = len(data)

# delData and Gendata after generalize,which will be write into .csv file as the final output
GenData = pd.DataFrame.copy(data)
minAge = data['age'].min()
maxAge = data['age'].max()
minEdu = data['education_num'].min()
maxEdu = data['education_num'].max()
partition = [((minAge,maxAge),(minEdu,maxEdu))]
GlobalAgeList = data['age'].values.tolist()
GlobalEduList = data['education_num'].values.tolist()
GlobalLen = len(data)

# implement Mondrian partition
def Mondrian(lAge:int,hAge:int,lEdu:int,hEdu:int):
    # find the group
    randomFlagForAge = False
    Agelist = []
    Edulist = []

    # M1: 0.3-2.0 S per time ( maybe exist CPU branch pridict) too slow
    ### bottle neck is create Agelist & Edulist 
    #for i in range(0,len(GenData)):
    #    if GenData.loc[i,"age"]>= lAge and GenData.loc[i,"age"]<= hAge and GenData.loc[i,"education_num"]>= lEdu and GenData.loc[i,"education_num"]<= hEdu:
    for i in range(0,GlobalLen):
        if GlobalAgeList[i]>= lAge and GlobalAgeList[i]<= hAge and GlobalEduList[i]>= lEdu and GlobalEduList[i]<= hEdu:
            Agelist.append(GlobalAgeList[i])
            Edulist.append(GlobalEduList[i])
    
    """
    # M2: 0.5S per time (more stable then M1) as slow as M1 
    ### bottle neck is create Agelist & Edulist
    data['ageTmp'] = data[['age','education_num']].apply(lambda x: x['age'] if x['age']>=lAge and x['age']<=hAge and x['education_num']>=lEdu and x['education_num']<=hAge else -1,axis=1)
    Agelist = data['ageTmp'].values.tolist()
    data['eduTmp'] = data[['age','education_num']].apply(lambda x: x['education_num'] if x['age']>=lAge and x['age']<=hAge and x['education_num']>=lEdu and x['education_num']<=hAge else -1,axis=1)
    Edulist = data['eduTmp'].values.tolist()    
    ### del -1 cost too much
    #while -1 in Agelist:
    #    Agelist.remove(-1)
    #while -1 in Edulist:
    #    Edulist.remove(-1)
    ### opt the "-1 del" 
    AgeCnt = pd.value_counts(Agelist)
    EduCnt = pd.value_counts(Edulist)
    tmpAgelist = list(set(Agelist))
    tmpEdulist = list(set(Edulist))
    Agelist.clear()
    Edulist.clear()
    if -1 in Agelist:
        tmpAgelist.remove(-1)
    if -1 in Edulist:
        tmpEdulist.remove(-1)
    Agelist = [val for val in tmpAgelist for i in range(AgeCnt[val])]
    Edulist = [val for val in tmpEdulist for i in range(EduCnt[val])]
    """
    
    # partition    
    AgeisAble = False
    median = int(np.nanmedian(Agelist))
    if sum(i > median  for i in Agelist) >= k and sum(i <= median  for i in Agelist) >= k:
        AgeisAble = True
    median = int(np.nanmedian(Edulist))
    EduisAble = False
    if sum(i > median  for i in Edulist) >= k and sum(i <= median  for i in Edulist) >= k:
        EduisAble = True
    if AgeisAble == False and EduisAble == False:
        return
    randomNum = random.randint(0,100)
    if AgeisAble == True and (EduisAble == False or randomNum%2):# Age
        if(not Agelist):
            return
        median = int(np.nanmedian(Agelist))
        if sum(i > median  for i in Agelist) >= k and sum(i <= median  for i in Agelist) >= k:
            partition.remove(((lAge,hAge),(lEdu,hEdu)))
            partition.append(((lAge,median),(lEdu,hEdu)))
            partition.append(((median+1,hAge),(lEdu,hEdu)))
            Mondrian(lAge,median,lEdu,hEdu)
            Mondrian(median+1,hAge,lEdu,hEdu)
            return             
    else:# Edu
        if(EduisAble == False):
            return
        if(not Edulist):
            return
        median = int(np.nanmedian(Edulist))
        if sum(i > median  for i in Edulist) >= k and sum(i <= median  for i in Edulist) >= k:
            partition.remove(((lAge,hAge),(lEdu,hEdu)))
            partition.append(((lAge,hAge),(lEdu,median)))
            partition.append(((lAge,hAge),(median+1,hEdu)))
            Mondrian(lAge,hAge,lEdu,median)
            Mondrian(lAge,hAge,median+1,hEdu)
            return 
    return 

# find the partition
Mondrian(minAge,maxAge,minEdu,maxEdu)

# generalization rule
Agedict = {}
for i in range(minAge,maxAge+1):
    for element in partition:
        if i >= element[0][0] and i <= element[0][1]:
            Agedict[i] = "("+str(element[0][0])+","+str(element[0][1])+")"
            break
Edudict = {}
for i in range(minEdu,maxEdu+1):
    for element in partition:
        if i >= element[1][0] and i <= element[1][1]:
            Edudict[i] = "("+str(element[1][0])+","+str(element[1][1])+")"
            break

# do Generalization
GenData['age'] = GenData['age'].map(lambda x:Agedict[x])
GenData['education_num'] = GenData['education_num'].map(lambda x:Edudict[x])

# groupby and contat
groups = GenData.groupby(['age','education_num'])
GroupGenData = pd.DataFrame(columns = ["age", "work_class" ,"final_weight" ,"education" ,"education_num" ,"marital_status" ,"occupation" ,"relationship" ,"race" ,"sex" ,"capital_gain" ,"capital_loss" ,"hours_per_week" ,"native_country" ,"class"])
for group in groups:
    GroupGenData = pd.concat([GroupGenData,group[1]])

# loss compute
ageLoss = 0.0
groups = GenData.groupby(['age'])
for group in groups:
    strNum = group[1]['age'].min()
    ageLoss += (int(strNum[strNum.index(',')+1:strNum.index(')')])-1 - int(strNum[1:strNum.index(',')]))*len(group[1])/(maxAge-minAge)
eduLoss = 0.0
groups = GenData.groupby(['education_num'])
for group in groups:
    strNum = group[1]['education_num'].min()
    eduLoss += (int(strNum[strNum.index(',')+1:strNum.index(')')])-1 - int(strNum[1:strNum.index(',')]))*len(group[1])/(maxEdu-minEdu)
totalloss = (ageLoss + eduLoss)/DataTupleNum

# write file
GroupGenData.to_csv("MondrianGendata.csv")

# time record
time_end=time.time()
print("loss:",totalloss)
print('time cost',time_end-time_start,'s')
