### Lab1 实验报告
#### 曾源 PB18111741
-------------------------------
* 该实验中本人并没有做交互模块，所有的参数只有`K``maxSup`且未写脚本进行自动测评。所有图像表格数据均在本地多次运行得到，故而无使用指南(有具体解释)，直接运行即可。
-------------------------------
### Samarati 算法实现

* $\textbf{实验要求:}$
  * 使用Adult数据集；
  * `QI={age, gender, race, marital _status}`(categorical 型), `S = {occupation}`；
  * 输入：`data`, `k`, `maxSup` (data是数据集，k是K-Anonymity的参数，maxSup表示最大suppression的个数)；
  * 输出：匿名后的数据集；
  * 评价指标： 运行时间和Loss-Metric。

* $\textbf{代码实现:}$(`Samarati.py`)
    * 数据读入：使用`pandas`库，将数据以`.csv`文件的形式读入并存于`dataframe`中。由于本人验证过`QI-cluster`的属性并不包含非法值，所以本人没有对含`?`数据进行删除，如若需要删除，也只需在读入部分加一条删除指令即可。
    
    ```python
    # read adult as .csv , get data whose type is dataframe
    data = pd.read_csv('adult.csv',encoding = 'GBK', engine="python",header = None)
    data.columns = ["age", "work_class" ,"final_weight" ,"education" ,"education_num" ,"marital_status" ,"occupation" ,"relationship" ,"race" ,"sex" ,"capital_gain" ,"capital_loss" ,"hours_per_week" ,"native_country" ,"class"]
    ```
    * 泛化规则生成：本人直接将泛化规则以`(属性，泛化层数)->(泛化属性)`的形式存于字典之中，并在后面的泛化函数中直接作用于临时表上。（由于该实验泛化结构较为简单，便没有专门为泛化生成写一个通用函数）
    ```python
    # Generalization rules for Race (only 2 layer)
    RaceGenRule = {("Other",0):"Other",("Amer-Indian-Eskimo",0):"Amer-Indian-Eskimo",("Black",0):"Black",("White",0):"White",("Asian-Pac-Islander",0):"Asian-Pac-Islander",
    ("Other",1):"*",("Amer-Indian-Eskimo",1):"*",("Black",1):"*",("White",1):"*",("Asian-Pac-Islander",1):"*"}
    # 其他代码不在此赘述
    ```
    * 泛化函数：该函数形式为`def generalization(vec:tuple,whetherSup:bool):`
    `vec`为泛化向量，用于指示各个属性的泛化层数，`whetherSup`用于指示是否泛化后需要进行删除操作，该函数只有在必要时对泛化表进行删除操作，以减少对表的读写操作，进而减少运行时间。该函数还进行了K匿名验证，并将验证结果以`bool`值返回。
    ```python
    # 此处只展示核心代码，使用dataframe的内置方法map(),结合lambda表达式可得到高效的整列替代的效果
        TmpGenData['age'] = TmpGenData['age'].map(lambda x :AgeGenRule[(x,ageLayer)])
        TmpGenData['marital_status'] = TmpGenData['marital_status'].map(lambda x:MaritalGenRule[(x.replace(' ',''),maritalStatusLayer)])
        TmpGenData['sex'] = TmpGenData['sex'].map(lambda x :GenderGenRule[(x.replace(' ',''),genderLayer)])
        TmpGenData['race'] = TmpGenData['race'].map(lambda x :RaceGenRule[(x.replace(' ',''),raceLayer)])

    ```
    * K匿名验证：对于泛化后的表单，本人使用了`dataframe`的`groupby`方法，对`QI-cluster`进行了分类，随后通过对每个`group`的大小进行判断，当不满足K匿名的`group`的累计条目数大于`maxSup`时，K匿名验证失败。
    ```python
        groups = TmpGenData.groupby(['age','marital_status','sex','race'],as_index = False).size()
        for i in groups['size']:
            if(i < k):
                AlreadySup += i
    ```
    * 二分寻找向量：寻找最佳泛化向量是`Samarati`算法的目标，参照`silde`的伪代码,本人实现了该算法的`python`版本,算法本身是`First-Fit`的，在高度一致的向量列表中，算法总会在找到第一个符合K匿名的泛化向量后便跳出循环，进行下一步二分。
    ```python
        low = 0
        high = len(vecDict)-1
        #init sol as completely generalization
        sol = (4,1,2,1)
        while low < high:
            tryNum = int((low+high)/2)
            veclist = vecDict[tryNum]
            reach_k = False
            while len(veclist) and reach_k != True:
                vec = veclist.pop(0)
                if generalization(vec,False):
                    sol = vec
                    reach_k = True
                if reach_k:
                    high = tryNum
                else:
                    low = tryNum + 1
    ```
    * 计算LM：同样的，本人使用以`泛化属性->损失值`为形式的字典进行`Loss-Metric`计算.值得注意的是，在`Samarati`算法中，`age`也是`categorical`型的数据，故而要按照`M-1/A-1`的形式来计算。其计算如下图所示：
    ```python
        # recognize age as categorical attribute
        AgeSet = set(data['age'].values.tolist())
        AgeNum = len(AgeSet)
        AgeLossDict = {"*":1,"-":1}
        for element in AgeSet:
            AgeLossDict[element] = 0
        for j in range(1,4):
            for l in range(0,int(100/stride[j])):
                for element in AgeSet:
                    tmpcnt = 0
                    if element>=l*stride[j] and element<(l+1)*stride[j]:
                        tmpcnt += 1
                AgeLossDict["("+str(l*stride[j])+","+str((l+1)*stride[j]-1)+")"] = (tmpcnt-1)/(AgeNum-1)
        
        # LM计算如下，以age为例 （元组总数在最后求和后再除）
        agegroups = GenData.groupby(['age'])
        ageLoss = 0.0
        for group in agegroups:
            ageLoss += AgeLossDict[group[1]['age'].min()]*len(group[1])
    ```
* $\textbf{实验结果分析:}$
    以`k`为行，`maxSup`为列，本人测试了25组数据，结果如下表所示，括号内为泛化向量，形式同`(gender,age,race,marital_status)`,每次测试用时均在2s以内。
    ||20|40|60|80|100|
    |---|---|---|---|---|---|
    |10|2.0545(0112)|2.0545(0112)|2.0545(0112)|2.0037(0012)|2.0037(0012)|
    |20|2.1207(0212)|2.0554(0112)|2.0554(0112)|2.0554(0112)|2.0554(0112)|
    |30|3.0543(1112)|2.0014(0410)|2.0014(0410)|2.0584(0112)|2.0584(0112)|
    |40|2.0(0402)|2.0014(0410)|2.0014(0410)|2.0584(0112)|2.0584(0112)|
    |50|2.0(0402)|2.0014(0410)|2.0014(0410)|2.0014(0410)|2.1238(0212)|
    ![](./figs/k=10S=20.png)
    由于`Samarati`算法`First-Fit`的特性，所得的数据表与向量列表的顺序有很大的相关性，但通过表大概可以确定：当`maxSup`固定时，`K`的增大会使得泛化向量的层数增高，因为苛刻的`maxSup`条件会使得泛化后分组过多的表不满足K匿名，进而使得泛化向量向高层发展;当`K`固定时，`maxSup`的增大会使得泛化向量的层数降低，这与前者的原因正好相反。宽松的`maxSup`条件使得泛化向量可以取得较低的值。同时，越低的泛化向量一般意味着更低的数据损失，但我认为这种关系不是绝对的，这与各个属性的泛化结构密切相关（这在下文附加题部分也得到了一定验证）。

* $\textbf{附加题Samarati部分:}$
    * 代码修改：
    为了获得更高的数据可用性，本人在`Samarati`的基础上，将`First-Fit`的策略变更为`Best-Fit`,在二分过程中会记录所有搜索到且复合K匿名的泛化向量，同时计算其`Loss-metric`并以此来判断泛化结果的可用性(越小越好)。在前文讨论的基础下，本人认为泛化向量的层次降低对`Loss-metric`的下降有一定的促进作用，因此在记录当前`Best vec`的同时，仍旧按照`Samarati`的算法，像低层次进行进一步二分搜索。主要修改如下(`SamaratiAdd.py`)
    ```python
    low = 0
    high = len(vecDict)-1
    #init sol as completely generalization
    sol = (4,1,2,1)
    minLoss = 100000.0
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
                minLoss = tmpLoss[0] if tmpLoss[0] < minLoss else minLoss
                reach_k = True
        if reach_k :
            high = tryNum
        else:
            low = tryNum + 1
    ```
    * 结果分析：
    同样的本人重新进行了25次测试，平均每场运行时间在7s左右
    ![](./figs/ADD1.png)

    ||20|40|60|80|100|
    |---|---|---|---|---|---|
    |10|2.0545(0112)|2.0545(0112)|1.2357(0211)|1.2357(0211)|1.0081(0400)|
    |20|2.0(0402)|2.0554(0112)|2.0554(0112)|2.0554(0112)|2.0554(0112)|
    |30|2.0(0402)|2.0(0402)|2.0(0402)|2.0584(0112)|2.0584(0112)|
    |40|2.0(0402)|2.0(0402)|2.0(0402)|2.0584(0112)|2.0584(0112)|
    |50|2.0(0402)|2.0(0402)|2.0(0402)|2.0(0402)|2.0(0402)|
    上述结果进一步验证了本人前文关于`K`,`maxSup`以及泛化层数的猜测，且出乎意料的是，在`K`逐渐增大过程中，高层但极端的泛化向量(0402)带来了更低的`Loss-metric`,而当`maxSup`逐渐增大时，宽松的删除条件会造成更多的不发布信息，在增大高层泛化向量`Loss-metric`的同时，让泛化向量进一步向低层发展。这也体现了泛化向量层次高低与该泛化带来的`Loss-metric`之间非绝对的正相关性。

### Mondrian 算法实现

* $\textbf{实验要求:}$
  * 使用Adult数据集；
  * `QI={age, education_num}`(数值型), `S = {occupation}`；
  * 输入：`data`, `k`, `maxSup` (data是数据集，k是K-Anonymity的参数；
  * 输出：匿名后的数据集；
  * 评价指标： 运行时间和Loss-Metric。

* $\textbf{代码实现:}$(`Mondrian.py`)
  * 数据读入:与`Samarati`同，在此不赘述。
  * `Mondrian`算法:该算法本质是不断的更新二维表的划分,直至所有划分区间都满足K匿名且不可继续划分下去。本人采用以两个元组为元素的元组构成的列表，即`[((lowAge,highAge)(lowEdu,highEdu))]`来描述划分的情形，并在划分结束后再根据该列表进行泛化规则生成与数据泛化。由于该算法有着明显的分治思想，于是本人使用递归的方法来实现。下面放出划分的核心代码：先判断两种属性划分的可行性并给相应的布尔变量赋值，随后使用随机的方法选择一个属性，并结合先前的布尔变量，进行某一属性的递归函数调用，进而实现分治划分的效果。
    ```python
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
    ```
  * 泛化规则生成：泛化规则根据`Mondrian`函数产生的`Partition`列表生成。由于递归时边界的不重叠性，每个数值对应的泛化区间是唯一的。本人采用字典的形式来描述这一规则，下面以`Age`这一属性为例：
    ```python
    Agedict = {}
    for i in range(minAge,maxAge+1):
        for element in partition:
            if i >= element[0][0] and i <= element[0][1]:
                Agedict[i] = "("+str(element[0][0])+","+str(element[0][1])+")"
                break
    ```
  * 泛化：这一步与`Samarati`相同,在此不重复解释
    ```python
        GenData['age'] = GenData['age'].map(lambda x:Agedict[x])
        GenData['education_num'] = GenData['education_num'].map(lambda x:Edudict[x])
    ```
  * `Loss-Metric`计算：数值型的泛化损失计算在于确认泛化区间的长度，落实到代码中便是从字符串间提取出区间左右界的问题。该部分我同样使用了`dataframe.groupby()`来协助计算`Loss-Metric`,具体实现如下：
    ```python
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
    ```
* $\textbf{实验结果分析:}$
    由于使用了随机的属性选取方法，为了保证结果的可靠性，本人将`Mondrian.py`改写并另存为`MondrianPlot.py`,在其中本人对`K`进行了`(10,200)`的步长为1的遍历，同时对于每一个`K`,都进行了5次计算并平均。具体的实验结果如下图所示(x:K,y:LM)
    ![](./figs/MondrianTrue.jpg)
    由于属性挑选的随机性，得到的二维表是震荡的。但是可以看到大体的趋势是`LM`随着`K`的增大缓慢地升高。而且由于`Mondrian`划分的较为精细，使得得到的`LM`貌似更低一些，平均值基本在0.1到0.3之间。此外，我并不能得到其他的结论或猜测。
    ![](./figs/MondrianTime.png)
    除此之外，包括写文件操作在内，`Modrian.py`的单次泛化时间也在4s内。对于`Python`而言，我任务这个时间也是可以接受。
    

* $\textbf{附加题Mondrian部分：}$
    我认为对于`categorical`型属性的`Mondrian`算法应用，应该根据本身其`categorical`的泛化结构，为每个叶子(即`R0`)属性进行排列与赋值，而且对于每个叶子结点，其应该满足：属于同一子树的叶子间数值差距要小，且子树根节点层级越低，该子树的叶子的数值应该更接近；相反的，不属于同一子树的叶子间数值差距要大，且子树根节点层级越高，不同子树的叶子的数值应该相差更大。随后将`Mondrian`算法中的取中位数操作更改为取平均数，并在该基础上进行`Mondrian`划分。进一步考虑，针对每个泛化结点对应子树拥有的叶子数目，应该赋予其相应权值，并将平均数计算转换为加权平均数计算，使得以及加权平均数的划分能更好地将属性值分开。例如我们为`gender = "Male"`赋予在`[0,10]`均匀分布的整数，为`gender = "FeMale"`赋予在`[20,30]`均匀分布的整数。假设`"Male":"FeMale" = 1:2`,则我们在计算加权平均数时，为`"FeMale"`赋予`1/2`的权值，于是期望上的加权平均数便是`(1*5+2*25*1/2)/2 = 15`,这使得划分的主元数值正好在两个性别对应的数值区间的中间，进而保证划分的合理性。
    由于这部分代码的修改并不复杂，只需进行属性赋值以及修改取中位数操作即可，故而在提交代码中并没有体现这一部分。


### 讨论与总结
* 总结(收获)
    * 更为深刻地理解了`K-Anonymity`的算法实现，并对参数`K`,`maxSup`有了进一步的认知。同时对泛化层数以及数据可用性的关系也有了一定的理解。
    * 对于`Python`的使用更为熟练，尤其对`Pandas`库相关的操作有了更深刻的理解。在本次实验，我探索了不少`dataframe`的遍历方法(`Mondrian.py`的注释中有我保留的历史遍历代码)，为此查阅了不少资料，同时也体会到了`lambda`表达式的精妙之处。
    * 资料的查询与理解，外文阅读能力也得到了锻炼。
* 讨论
    * 对于数据的可用性方面，本人只是采用了`Loss-Metric`指标进行评价。不知其他的评价指标(如CM,DM,AM)与LM相比有何优劣。
    * 对于本实验提供的两个算法，我认为其效率还是不高的，不知是否有更为高效的算法。以`Samarati`为例子，能否利用已经计算的其他层次的泛化结果，在其基础上得到一定的信息来优化自己的泛化向量搜素路线。