# using samarati alg to create K-Anonymity table 
import pandas as pd
import time

# time record
time_start=time.time()

# para init
k = 10
maxSup = 20

# read adult as .csv , get data whose type is dataframe
data = pd.read_csv('adult.csv',encoding = 'GBK', engine="python",header = None)
data.columns = ["age", "work_class" ,"final_weight" ,"education" ,"education_num" ,"marital_status" ,"occupation" ,"relationship" ,"race" ,"sex" ,"capital_gain" ,"capital_loss" ,"hours_per_week" ,"native_country" ,"class"]

# delData and Gendata after generalize,which will be write into .csv file as the final output
GenData = data
DelData = pd.DataFrame(columns = ["age", "work_class" ,"final_weight" ,"education" ,"education_num" ,"marital_status" ,"occupation" ,"relationship" ,"race" ,"sex" ,"capital_gain" ,"capital_loss" ,"hours_per_week" ,"native_country" ,"class"])
DataTupleNum = len(data)

# vecdict[vecsum] = veclist; veclist = [(#age,#gender,#marital_status,#race),(),()]
vecDict = {}
for ageGenLayer in range(0,5):
    for genderGenLayer in range(0,2):
        for raceGenLayer in range(0,2):
            for maritalStatusGenLayer in range(0,3):
                vecDict.setdefault(ageGenLayer+genderGenLayer+maritalStatusGenLayer+raceGenLayer,[]).append((ageGenLayer,genderGenLayer,raceGenLayer,maritalStatusGenLayer))

# generalization rules (key is a tuple like (attribute,GenLayer) ;value is genAttribute)
# Generalization rules for Race (only 2 layer)
RaceGenRule = {("Other",0):"Other",("Amer-Indian-Eskimo",0):"Amer-Indian-Eskimo",("Black",0):"Black",("White",0):"White",("Asian-Pac-Islander",0):"Asian-Pac-Islander",
("Other",1):"*",("Amer-Indian-Eskimo",1):"*",("Black",1):"*",("White",1):"*",("Asian-Pac-Islander",1):"*"}
# Generalization rules for Gender (only 2 layer)
GenderGenRule = {("Female",0):"Female",("Male",0):"Male",("Female",1):"*",("Male",1):"*"}
# Generalization rules for Age ( 5 layer)
AgeGenRule = {}
stride = (1,5,10,20)
for i in range(0,100):
    AgeGenRule[(i,0)] = i
    AgeGenRule[(i,4)] = "*"
    for j in range(1,4):
        for l in range(0,int(100/stride[j])):
            if i>=l*stride[j] and i<(l+1)*stride[j]:
                AgeGenRule[(i,j)] = "("+str(l*stride[j])+","+str((l+1)*stride[j]-1)+")"
# Generalization rules for Marital_staus (3 layer)
MaritalGenRule = {("Married-spouse-absent",0):"Married-spouse-absent",("Widowed",0):"Widowed",("Separated",0):"Separated",("Divorced",0):"Divorced",
("Married-AF-spouse",0):"Married-AF-spouse",("Married-civ-spouse",0):"Married-civ-spouse",("Never-married",0):"Never-married",
("Married-spouse-absent",1):"alone",("Widowed",1):"alone",("Separated",1):"leave",("Divorced",1):"leave",
("Married-AF-spouse",1):"Married",("Married-civ-spouse",1):"Married",("Never-married",1):"NM",
("Married-spouse-absent",2):"*",("Widowed",2):"*",("Separated",2):"*",("Divorced",2):"*",
("Married-AF-spouse",2):"*",("Married-civ-spouse",2):"*",("Never-married",2):"*"}

################generalization func###################################
#if isSatisfies == True then update GenData
def generalization(vec:tuple,whetherSup:bool):
# init 
    isSatisfies = False
    TmpGenData =  pd.DataFrame.copy(data)
    AlreadySup = 0
    genderLayer = vec[1]
    ageLayer = vec[0]
    raceLayer = vec[2]
    maritalStatusLayer = vec[3]
    SupGenData = pd.DataFrame(columns = ["age", "work_class" ,"final_weight" ,"education" ,"education_num" ,"marital_status" ,"occupation" ,"relationship" ,"race" ,"sex" ,"capital_gain" ,"capital_loss" ,"hours_per_week" ,"native_country" ,"class"])
    TmpDelData =  pd.DataFrame(columns = ["age", "work_class" ,"final_weight" ,"education" ,"education_num" ,"marital_status" ,"occupation" ,"relationship" ,"race" ,"sex" ,"capital_gain" ,"capital_loss" ,"hours_per_week" ,"native_country" ,"class"])

# generalization
# age,marital_status,sex,race
    TmpGenData['age'] = TmpGenData['age'].map(lambda x :AgeGenRule[(x,ageLayer)])
    TmpGenData['marital_status'] = TmpGenData['marital_status'].map(lambda x:MaritalGenRule[(x.replace(' ',''),maritalStatusLayer)])
    TmpGenData['sex'] = TmpGenData['sex'].map(lambda x :GenderGenRule[(x.replace(' ',''),genderLayer)])
    TmpGenData['race'] = TmpGenData['race'].map(lambda x :RaceGenRule[(x.replace(' ',''),raceLayer)])

# groupby
    if whetherSup!=True:
        groups = TmpGenData.groupby(['age','marital_status','sex','race'],as_index = False).size()
        for i in groups['size']:
            if(i < k):
                AlreadySup += i
# Suppression
    else:
        groups = TmpGenData.groupby(['age','marital_status','sex','race'])
        for group in groups:
            tmpDataFrame = pd.DataFrame.copy(group[1])
            if(len(tmpDataFrame) < k):
                TmpDelData = pd.concat([TmpDelData,tmpDataFrame])
                tmpDataFrame.loc[:,:] = "-"
            SupGenData = pd.concat([SupGenData,tmpDataFrame])

# judge isSatisfies
    if AlreadySup <= maxSup:
        isSatisfies = True
    if isSatisfies:
        global GenData
        global DelData
        GenData = SupGenData
        DelData = TmpDelData
    return isSatisfies;
#########################func finish################################

# prepare for compute Loss 
RaceLossDict = {"Other":0.0,"Amer-Indian-Eskimo":0.0,"Black":0.0,"White":0.0,"Asian-Pac-Islander":0.0,"*":1.0,"-":1.0}
MaritalLossDict = {"Married-spouse-absent":0.0,"Widowed":0.0,"Separated":0.0,"Divorced":0.0,"Married-AF-spouse":0.0,
"Married-civ-spouse":0.0,"Never-married":0.0,"Married":1.0/6,"NM":0.0,"alone":1.0/6,"leave":1.0/6,"*":1.0,"-":1.0}
GenderLossDict = {"Male":0.0,"Female":0.0,"*":1.0,"-":1.0}
# recognize age as categorical attribute
AgeSet = set(data['age'].values.tolist())
AgeNum = len(AgeSet)
AgeLossDict = {"*":1.0,"-":1.0}
for element in AgeSet:
    AgeLossDict[element] = 0.0
for j in range(1,4):
    for l in range(0,int(100/stride[j])):
        tmpcnt = 0
        for element in AgeSet:
            if element>=l*stride[j] and element<(l+1)*stride[j]:
                tmpcnt += 1
        AgeLossDict["("+str(l*stride[j])+","+str((l+1)*stride[j]-1)+")"] = (tmpcnt-1)/(AgeNum-1)

#####################Loss Compute Func################################
def LossMetric():
    agegroups = GenData.groupby(['age'])
    ageLoss = 0.0
    for group in agegroups:
        ageLoss += AgeLossDict[group[1]['age'].min()]*len(group[1])
    maritalgroups = GenData.groupby(['marital_status'])
    maritalLoss = 0.0
    for group in maritalgroups:
        maritalLoss += MaritalLossDict[group[1]['marital_status'].min()]*len(group[1])
    gendergroups = GenData.groupby(['sex'])
    genderLoss = 0.0
    for group in gendergroups:
        genderLoss += GenderLossDict[group[1]['sex'].min()]*len(group[1])
    racegroups = GenData.groupby(['race'])
    raceLoss = 0.0
    for group in racegroups:
        raceLoss += RaceLossDict[group[1]['race'].min()]*len(group[1])
    totalLoss = (ageLoss + maritalLoss + genderLoss + raceLoss)/DataTupleNum
    return (totalLoss, ageLoss , maritalLoss , genderLoss , raceLoss)
##########################Func Finish###############################

# find vector(samarati alg , Best-Fix)
low = 0
high = len(vecDict)-1
#init sol as completely generalization
sol = (4,1,2,1)
minLoss = 100000.0
minageLoss = 0.0
minmaritalLoss = 0.0
mingenderLoss = 0.0
minraceLoss = 0.0
while low < high:
    tryNum = int((low+high)/2)
    veclist = vecDict[tryNum]
    reach_k = False
    while len(veclist):
        vec = veclist.pop(0)
        if generalization(vec,False):
            generalization(vec,True)
            tmpLoss = LossMetric()
            sol = vec if tmpLoss[0] < minLoss else sol
            minageLoss = tmpLoss[1] if tmpLoss[0] < minLoss else minageLoss
            minmaritalLoss = tmpLoss[2] if tmpLoss[0] < minLoss else minmaritalLoss
            mingenderLoss = tmpLoss[3] if tmpLoss[0] < minLoss else mingenderLoss
            minraceLoss = tmpLoss[4] if tmpLoss[0] < minLoss else minraceLoss
            minLoss = tmpLoss[0] if tmpLoss[0] < minLoss else minLoss
            reach_k = True
    if reach_k :
        high = tryNum
    else:
        low = tryNum + 1

generalization(sol,True)

GenData.to_csv("SamaratiAddGendata.csv")
DelData.to_csv("SamaratiAddDelData.csv")

# time record
time_end=time.time()
print("(age,gender,race,ms)")
print(sol)
print("loss:",(minLoss,minageLoss,mingenderLoss,minraceLoss,minmaritalLoss))
print("DataNum:",DataTupleNum)
print('time cost:',time_end-time_start,'s')

