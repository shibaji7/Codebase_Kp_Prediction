import numpy as np
import pandas as pd
import datetime as dt

import database as db
from models import DetRegressor,LSTMClassifier,LSTMDataSource,GPRegressor

import os
#os.remove("../out/storm/prediction.csv")
os.remove("../out/storm/prediction_gpr.csv")

name = "regression"
_o, xparams, yparam_clf = db.load_data()
dataset = db.load_data(case=1)
source = LSTMDataSource(_o, xparams, ["K_P_LT_delay"], yparam_clf)
clf = LSTMClassifier(source=source)

days = 10
trws = np.arange(1,10).astype(int)
dates = [dt.datetime(1995,7,1) + dt.timedelta(hours=3*i) for i in range(days*8)]

k_type = "Matern"

for trw in trws:
    for d in dates:
        #det_reg = DetRegressor(name, d, dataset, clf, window=trw, alt_window=trw, is_mix_model=True, source=source)
        #det_reg.run_model()
        gp_reg = GPRegressor(d, dataset, clf, k_type, window=trw, source=source, alt_window=trw)
        gp_reg.run_model()
        pass
    pass

o = pd.read_csv("../out/storm/prediction_gpr.csv")
for trw in trws:
    x = o[o.window==trw]
    print trw,"-", np.abs(x.kp-x.kp_pred).mean()
    pass
