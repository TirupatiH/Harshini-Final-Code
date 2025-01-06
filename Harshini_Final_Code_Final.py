def DataRead():
    dr=os.getcwd()
    if "CKD_Data.csv" not in os.listdir():
        data = fetch_ucirepo(id=336) 
        dtck=pandas.concat([data.data.features,data.data.targets],axis=1)
        dtck.to_csv("CKD_Data.csv")
        return dtck
    else:
        dtck=pandas.read_csv("CKD_Data.csv")
        if 'Unnamed: 0' in dtck.columns:
            dtck=dtck.drop('Unnamed: 0',axis=1)
        return dtck
KDS=DataRead()
KDS.head()
def InfoData(dtck):
    resval1=len(dtck)
    print(dtck.info(),"\n")
    display(HTML(dtck.describe().to_html()))
    return dtck
KDS=InfoData(KDS)
def DataCleaning(dtck):
    data_length_init=len(dtck)
    if sum(dtck.isna().sum())>0:
        print(dtck.isna().sum())
        dtck=dtck.dropna()
        data_length_after=len(dtck)
        dtck=utils.resample(dtck,replace = True, n_samples = int(len(dtck)*(data_length_init/data_length_after)), random_state = 10)
        print(dtck.isna().sum())
        print(dtck.info())
        dtck=dtck.reset_index(drop=True)
    return dtck
KDS=DataCleaning(KDS)
KDS.head()
def RectifyFet(dtck):
    dtck['class']=dtck['class'].replace("ckd\t","ckd")
    return dtck
KDS=RectifyFet(KDS)
print(KDS['class'].value_counts())
KDS.head()
def CatGraph(dtck,ft,nm):
    pandas.crosstab(dtck[ft],dtck['class']).plot(kind='barh',figsize=(4,2), color=['m','g'],title="Kidney Disease by {}".format(nm))
cat_fets=["rbc","pc","pcc","ane","ba","appet"]
names=["Red blood cells", "Pus cell", "Clumps of Pus cell", "Anaemia Possibility","Bacteria Infection", "Level Of Appetite"]
for cf in range(len(cat_fets)):
    CatGraph(KDS,cat_fets[cf],names[cf])
def NumGraph(dtck,ft, nm):
    kdnyclass=dtck[dtck['class']=='ckd']
    hltyclass=dtck[dtck['class']=='notckd']
    ckvs.figure(figsize=(4,4))
    ckvs.title("Kidney Disease by {}".format(nm))
    ckvs.pie([kdnyclass[ft].mean(),hltyclass[ft].mean()],labels=["CKD","NOT CKD"],
                         colors=cksb.color_palette('Set3'), autopct='%1.0f%%',pctdistance=0.5, labeldistance=0.2)
    ckvs.show()
num_fets=["bgr","bu","sc","sod","hemo"]
names=["Amount of Glucose", "Amount of Urea", "Amount of Serum Creatinine", "Sodium Level", "Amount of Haemoglobin"]
for nf in range(len(num_fets)):
    NumGraph(KDS,num_fets[nf], names[nf])
def DataEncoding(dtck):
    dtcktg=dtck['class']
    dtck1=dtck.drop('class',axis=1)
    dtckcat=dtck1.dtypes[dtck1.dtypes=='object'].index.tolist()
    for k in range(len(dtckcat)):
        dtck1[dtckcat[k]]=dtck1[dtckcat[k]].replace(dtck1[dtckcat[k]].unique(),[x for x in range(len(dtck1[dtckcat[k]].unique()))])
    dtck2=pandas.concat([dtck1,dtcktg],axis=1)
    return dtck2
ECKDS=DataEncoding(KDS)
ECKDS.head()
def PCACK(dtck,n,w,h,col,TX):   
    arrdtck=numpy.array(dtck.iloc[:,:-1]) 
    pcdtck = decomposition.PCA(n_components=n) 
    pcdtck.fit(arrdtck) 
    pcdtckcm=["Comp-{}".format(i+1) for i in range(len(pcdtck.explained_variance_ratio_.tolist()))]
    ckvs.figure(figsize=(w,h))  
    ckvs.title("Variance(PCA={})\n{} Normalization Data\nMaximum Variance Value: {}".format(n,TX,round(max(pcdtck.explained_variance_ratio_),8)),fontsize=18)
    ckvs.bar(pcdtckcm,pcdtck.explained_variance_ratio_.tolist(),width=0.5,color=col) 
    ckvs.xlabel("PCA",fontsize=14)
    ckvs.ylabel("Variance",fontsize=14)
    ckvs.grid()
    ckvs.show()
    return pcdtck.explained_variance_ratio_

def DataScale(dtck): 
    ssnorm = preprocessing.StandardScaler() 
    ckscl=ssnorm.fit_transform(dtck) 
    return ckscl
pcvl=[]
pcvl.append(PCACK(ECKDS.drop('class',axis=1),2,6,3,"#FF00FF","Before"))

outvl=[]  
for pv in pcvl: 
    for p in pv:
        if p>0.6:
            outvl.append(True)
if len(outvl)==1 and True in outvl:
    ScKDSdt=DataScale(ECKDS.drop('class',axis=1))   
ScKDS=pandas.DataFrame(ScKDSdt,columns=ECKDS.drop('class',axis=1).columns.tolist())     
ScKDS['class']=ECKDS['class']
PCACK(ScKDS[::-1],2,6,3,"#6AFB92","After")
ScKDS.head()
def CKF1(dtck):
    Xdtck=dtck.drop([dtck.columns.tolist()[-1]],axis=1)
    Ydtck=dtck[dtck.columns.tolist()[-1]]
    nmft=Xdtck.columns.tolist()
    ckensm = ensemble.RandomForestClassifier(random_state=0)
    ckensm.fit(Xdtck, Ydtck)
    impck = ckensm.feature_importances_
    ftsckdf=pandas.DataFrame({"Feature":nmft,"Importance":impck})
    ftsckdf1=ftsckdf[ftsckdf['Importance']>0.01]
    ckvs.figure(figsize=(7,3))
    ckvs.bar(ftsckdf1['Feature'],ftsckdf1['Importance'])
    ckvs.title("Feature Importance",fontsize=20,color="b")
    ckvs.xlabel("Features",fontsize=17,color="b")
    ckvs.ylabel("Importance",fontsize=17,color="b")
    ckvs.xticks(rotation=90)
    ckvs.grid()
    ckvs.show()
    display(HTML(ftsckdf1.to_html()))
    print("Total Features Selecetd Using Ensemble: {}".format(len(ftsckdf1)))
    return ftsckdf1['Feature'].tolist()
def CKF2(dtck):
    Xdtck=dtck.drop([dtck.columns.tolist()[-1]],axis=1)
    Ydtck=dtck[dtck.columns.tolist()[-1]]
    Ydtck=Ydtck.replace(Ydtck.unique(),[x for x in range(len(Ydtck.unique()))]) 
    M2 = feature_selection.RFE(estimator=linear_model.LogisticRegression(),n_features_to_select = int(len(Xdtck.columns)*0.6), step = 0.7)
    M2Trnd=M2.fit(Xdtck,Ydtck)
    ftrfe=pandas.DataFrame({"Feature":Xdtck.columns,"Ranking":M2Trnd.ranking_})
    ftrfe2=ftrfe[ftrfe['Ranking']==1]
    print(len(ftrfe2))
    display(HTML(ftrfe2.to_html()))
    return ftrfe2['Feature'].tolist()
kdsfets=[]  
ensft=CKF1(ScKDS) 
rfft=CKF2(ScKDS) 
for x in ensft:   
    if x in rfft:   
        kdsfets.append(x) 
print("Features Selected Using Ensemble\n",*ensft, sep="\n")
print("Features Selected Using RFE\n",*rfft, sep="\n")
print("Features Selected Using Hybrid\n",*kdsfets, sep="\n")
print("Total Features Selecetd Using Hybrid: ",len(kdsfets))
X=ECKDS.drop('class',axis=1)
X=X[kdsfets]
y=ECKDS['class']
x_train,x_test,y_train,y_test=model_selection.train_test_split(X,y, train_size=0.8, random_state=10)
print(y_test.value_counts())
print("\n")
print(y_train.value_counts())
CkModelInit=[
    ensemble.RandomForestClassifier(),
    neural_network.MLPClassifier(),
    naive_bayes.MultinomialNB(),
    svm.LinearSVC(),
    tree.DecisionTreeClassifier(),
    neighbors.KNeighborsClassifier()
]
ackmdnm=[
    "Random Forest",
    "MLP Classifier",
    "Naive Bayes",
    "Decision Tree Classifier",
    "Support Vector Classifier",
    "K-Neighbors"
]
CkModelInitopt=CkModelInit.copy()
optmkd=[
    [{"min_weight_fraction_leaf":[0.4,0.5,0.6]}],
    [{"hidden_layer_sizes":[1,2,3],"max_iter":[1,2]}],
    [{"alpha":[0.6,0.8,1.0],"fit_prior":[False]}],
    [{"tol":[0.01,0.001,0.0001,0.00001],'C':[0.2,0.4,0.6,0.8,1.0],"max_iter":[2,3,4]}],
    [{"criterion":['entropy'],"min_weight_fraction_leaf":[0.3,0.4,0.5]}],
    [{"n_neighbors":[3,4,5],"algorithm":['auto', 'ball_tree', 'kd_tree']}]
]
optparams=[]
for ci in range(len(CkModelInitopt)):
    optgs = model_selection.GridSearchCV(CkModelInitopt[ci], optmkd[ci], cv = 5, scoring='accuracy')
    optgs.fit(x_train,y_train)
    CkModelInitopt[ci]=optgs.best_estimator_
    optparams.append(optgs.best_estimator_)
dataoptm=pandas.DataFrame({"Model":ackmdnm,"Optimized Version":optparams})
dataoptm.to_csv("OptModels.csv")
CkModelInitopt
MetCKInit=[[],[],[],[],[]]
for i in range(len(CkModelInit)):
    ScMetLp=[[],[],[],[],[],[]]
    for ts in range(10):
        t1 = datetime.datetime.now()
        CkModelInit[i].fit(x_train,y_train)
        kidprd=CkModelInit[i].predict(x_test)
        t2 = datetime.datetime.now()
        delta = t2 - t1
        time_pred=delta.total_seconds()
        ScMetLp[0].append(round(metrics.accuracy_score(y_test,kidprd)*100,2))
        ScMetLp[1].append(round(metrics.precision_score(y_test, kidprd, average='weighted'),2)*100)
        ScMetLp[2].append(round(metrics.recall_score(y_test, kidprd, average='weighted'),2)*100)
        ScMetLp[3].append(round(metrics.f1_score(y_test, kidprd, average='weighted'),2)*100)
        cm=pandas.crosstab(y_test, kidprd, rownames=['True'], colnames=['Predicted'], margins=True)
        ScMetLp[4].append(cm.iloc[:2,:2])
        ScMetLp[5].append(time_pred)
    opt_idx=ScMetLp[0].index(max(ScMetLp[0]))
    MetCKInit[0].append(ScMetLp[0][opt_idx])
    MetCKInit[1].append(ScMetLp[1][opt_idx])
    MetCKInit[2].append(ScMetLp[2][opt_idx])
    MetCKInit[3].append(ScMetLp[3][opt_idx])
    MetCKInit[4].append(ScMetLp[5][opt_idx])
ResInitCKD=pandas.DataFrame({
    "Classifiers":ackmdnm,
    "Accuracy":MetCKInit[0],
    "Precision":MetCKInit[1],
    "Recall":MetCKInit[2],
    "F1-Score":MetCKInit[3],
    "Execution Time":MetCKInit[4]
})

for i in ResInitCKD.columns.tolist()[1:]:
    ResInitCKD=ResInitCKD.sort_values(by=i,ascending=False)
    fig = express.bar(ResInitCKD, y=i, x="Classifiers",text=i,color="Classifiers",
                 title="{}(Unfiltered)".format(i),height=400,width=600)
    fig.show()
ResInitCKD1=ResInitCKD[ResInitCKD['Accuracy']<100].reset_index(drop=True)
for i in ResInitCKD1.columns.tolist()[1:]:
    ResInitCKD1=ResInitCKD1.sort_values(by=i,ascending=False)
    fig = express.bar(ResInitCKD1, y=i, x="Classifiers",text=i,color="Classifiers",
                 title="{}(Filtered)".format(i),height=400,width=600)
    fig.show()
pickle.dump(CkModelInitopt[3], open("KIDOPT.sav", 'wb'))
CkModelInitopt[3]
ResInitCKD
ResInitCKD1
kdopt=CkModelInitopt[3]
kdopt.fit(X,y)
print("Optimum Model\n",ResInitCKD1['Classifiers'][4])
optmd = 'kdopt.sav'
pickle.dump(kdopt, open(optmd, 'wb'))
print("\nOptimum Model Structure:\n")
kdopt
expdata=pandas.concat([X,y],axis=1)
expdata.to_csv("expdata.csv")
expdata