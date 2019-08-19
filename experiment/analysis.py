import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn import preprocessing


import database as db

##########################################################################
## PCA Analysis and find out the contribution made by individual components
##########################################################################
components = 10
dates = pd.read_csv("stormlist.csv")
dates.dates = pd.to_datetime(dates.dates)
_o, xparams, yparam_clf = db.load_data()
pca = decomposition.PCA(n_components=components)
df = _o[xparams]
data_scaled = pd.DataFrame(preprocessing.scale(df),columns = df.columns)
pca.fit_transform(data_scaled)
comp_desc = pd.DataFrame(pca.components_,columns=data_scaled.columns,index = ['PC-'+str(i) for i in range(components)])
print pca.explained_variance_ratio_,data_scaled.columns
print np.cumsum(pca.explained_variance_ratio_)
desc_code = []
for i,row in comp_desc.iterrows():
    code = ""
    for r,v in zip(row.tolist(),comp_desc.columns):
        if r<0 and len(code)==0: 
            code = code[:-2] + str(np.round(r,2))
        elif r<0 and len(code)>0: 
            code = code[:-2] + str(np.round(r,2))
        elif r>0: 
            code = code + str(np.round(r,2))
        code = code + " X %s + "%v
        pass
    code = code[:-2]
    desc_code.append("%s=%s"%(i,code))
    pass
#comp_desc["code"] = [np.round()]
print desc_code
