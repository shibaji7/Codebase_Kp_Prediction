import numpy as np
import pandas as pd
import datetime as dt

import database as db
from models import DetRegressor,Classifier,LSTMClassifier,LSTMRegressor,LSTMDataSource,GPRegressor,DeepGPR,DeepGPRegressor


run_without_goes_machine = True

if run_without_goes_machine:
###########################################################################################################
## Main body of the program to run storm events "without goes data".
###########################################################################################################
    case = -1
    reg_name_list = ["regression","elasticnet","bayesianridge","dtree","etree","knn","ada","bagging","etrees","gboost","randomforest"]
    training_winows = range(1,108,7)#[14, 27, 54, 81, 108, 135, 162, 189, 216, 243, 270, 297, 324, 351, 378]
    is_mix_model = False
    dates = pd.read_csv("stormlist.csv")
    dates.dates = pd.to_datetime(dates.dates)
    _o, xparams, yparam_clf = db.load_data()
    dataset = db.load_data(case=1)
    #reg_name_list = ["dtree"]
    
    ## Case for all deterministic models
    if case == 0:
        source = None
        if is_mix_model:
            source = LSTMDataSource(_o, xparams, ["K_P_LT_delay"], yparam_clf)
            clf = LSTMClassifier(source=source)
        else: clf = Classifier()
    
        for name in reg_name_list:
            for trw in training_winows:
                for date in dates.dates.tolist():
                    if date >= dt.datetime(2000,1,1) and date < dt.datetime(2014,1,1):
                        for h in range(8):
                            hour_date = date.to_pydatetime() + dt.timedelta(hours=h*3)
                            print "Execute outputs for one model DateTime - ",name,",",hour_date
                            try:
                                det_reg = DetRegressor(name, hour_date, dataset, clf, window=trw, alt_window=trw,#np.random.randint(28,56)+trw, 
                                        is_mix_model=is_mix_model, source=source)
                                det_reg.run_model()
                            except: pass
                            pass
                        pass
                    pass
                pass
            pass
    
        pass
    ## Case for all deterministic LSTM model
    elif case == 1:
        source = LSTMDataSource(_o, xparams, ["K_P_LT_delay"], yparam_clf)
        clf = LSTMClassifier(source=source)
    
        loop_back = 3
        for trw in training_winows:
            for date in dates.dates.tolist():
                if date >= dt.datetime(2000,1,1) and date < dt.datetime(2014,1,1):
                    for h in range(8):
                        hour_date = date.to_pydatetime() + dt.timedelta(hours=h*3)
                        print "Execute outputs for one model DateTime - LSTM",",",hour_date
                        try:
                            lstm_reg = LSTMRegressor(hour_date, source, clf, window=trw, alt_window=np.random.randint(28,56)+trw)
                            lstm_reg.run_model()
                        except: pass
                        pass
                    pass
                pass
            pass
    
        pass
    ## Case for all GP model
    elif case == 2:
        source = LSTMDataSource(_o, xparams, ["K_P_LT_delay"], yparam_clf)
        clf = LSTMClassifier(source=source)
        k_type = "Matern"
    
        for trw in training_winows:
            for date in dates.dates.tolist():
                if date >= dt.datetime(2000,1,1) and date < dt.datetime(2014,1,1):
                    for h in range(8):
                        hour_date = date.to_pydatetime() + dt.timedelta(hours=h*3)
                        print "Execute outputs for one model DateTime - GPR",",",hour_date
                        try:
                            gp_reg = GPRegressor(hour_date, dataset, clf, k_type, window=trw, source=source,alt_window=np.random.randint(28,56)+trw)
                            gp_reg.run_model()
                        except: pass
                        pass
                    pass
                pass
            pass
    
        pass
    ## Case for all LSTM GP model
    elif case == 3:
        source = LSTMDataSource(_o, xparams, ["K_P_LT_delay"], yparam_clf)
        clf = LSTMClassifier(source=source)
        k_type = "RBF"
    
        for trw in training_winows:
            for date in dates.dates.tolist():
                if date >= dt.datetime(2000,1,1) and date < dt.datetime(2014,1,1):
                    for h in range(8):
                        hour_date = date.to_pydatetime() + dt.timedelta(hours=h*3)
                        print "Execute outputs for one model DateTime - GPR",",",hour_date
                       #try:
                        deep_gp = DeepGPRegressor(hour_date, source, clf, k_type, window=trw, alt_window=np.random.randint(28,56)+trw)
                        success = deep_gp.run_model()
                        #except: pass
                        pass
                    pass
                pass
            pass
        
        pass
    ## Case for all deep GPR
    elif case == 4:
        source = LSTMDataSource(_o, xparams, ["K_P_LT_delay"], yparam_clf)
        clf = LSTMClassifier(source=source)
    
        for name in reg_name_list:
            for trw in training_winows:
                for date in dates.dates.tolist():
                    if date >= dt.datetime(2000,1,1) and date < dt.datetime(2014,1,1):
                        for h in range(8):
                            hour_date = date.to_pydatetime() + dt.timedelta(hours=h*3)
                            print "Execute outputs for one model DateTime - ",name,",",hour_date
                            try:
                                dgp = DeepGPR(name, hour_date, dataset, clf, window=trw, alt_window=np.random.randint(28,56)+trw, source=source)
                                dgp.run_model()
                            except: pass
                            pass
                        pass
                    pass
                pass
            pass
    
        pass

    pass

else:
    ###########################################################################################################
    ## Main body of the program to run storm events "with goes data".
    ###########################################################################################################
    case = -1
    reg_name_list = ["regression","elasticnet","bayesianridge","dtree","etree","knn","ada","bagging","etrees","gboost","randomforest"]
    training_winows = [14, 27, 54, 81, 108, 135, 162, 189, 216, 243, 270, 297, 324, 351, 378]
    is_mix_model = True
    dates = pd.read_csv("stormlist.csv")
    dates.dates = pd.to_datetime(dates.dates)

    #clf = Classifier(is_goes=True)
    reg_name_list, training_winows = ["regression"], [14,27,54,81,108]

    ## Case for all deterministic models
    if case == 0:
        source = None
        if is_mix_model:
            _o, xparams, yparam_clf = db.load_data()
            source = LSTMDataSource(_o, xparams, ["K_P_LT_delay"], yparam_clf)
            clf = LSTMClassifier(source=source)
        else: clf = Classifier(is_goes=True)
        _o, xparams, yparam_clf = db.load_data_RB()
        dataset = db.load_data_RB(case=1)
        
        for name in reg_name_list:
            for trw in training_winows:
                for date in dates.dates.tolist():
                    if date >= dt.datetime(2000,1,1) and date < dt.datetime(2014,1,1):
                        for h in range(8):
                            hour_date = date.to_pydatetime() + dt.timedelta(hours=h*3)
                            print "Execute outputs for one model DateTime - ",name,",",hour_date
                            #try:
                            det_reg = DetRegressor(name, hour_date, dataset, clf, window=trw, alt_window=np.random.randint(28,56)+trw, 
                                        is_mix_model=is_mix_model, source=source)
                            det_reg.run_model()
                            #except: pass
                            pass
                        pass
                    pass
                pass
            pass

        pass
    ## Case for all mix regression GP models
    elif case == 1:
        _o, xparams, yparam_clf = db.load_data()
        source = LSTMDataSource(_o, xparams, ["K_P_LT_delay"], yparam_clf)
        clf = LSTMClassifier(source=source)
        
        _o, xparams, yparam_clf = db.load_data_RB()
        dataset = db.load_data_RB(case=1)
    
        for name in reg_name_list:
            for trw in training_winows:
                for date in dates.dates.tolist():
                    if date >= dt.datetime(2000,1,1) and date < dt.datetime(2014,1,1):
                        for h in range(8):
                            hour_date = date.to_pydatetime() + dt.timedelta(hours=h*3)
                            print "Execute outputs for one model DateTime - ",name,",",hour_date
                            try:
                                dgp = DeepGPR(name, hour_date, dataset, clf, window=trw, alt_window=np.random.randint(28,56)+trw, source=source)
                                dgp.run_model()
                            except: pass
                            pass
                        pass
                    pass
                pass
            pass


    pass #EOD



## Case for all deep GPR 2 month periods
case = -1
if case == 0:
    reg_name_list = ["regression","elasticnet","bayesianridge","dtree","etree","knn","ada","bagging","etrees","gboost","randomforest"]
    training_winows = [14, 27, 54, 81]
    dates = [dt.datetime(2004,7,1) + dt.timedelta(days=i) for i in range(62)]
    _o, xparams, yparam_clf = db.load_data()
    dataset = db.load_data_RB(case=1)
   
    source = LSTMDataSource(_o, xparams, ["K_P_LT_delay"], yparam_clf)
    clf = LSTMClassifier(source=source)

    for name in reg_name_list:
        for trw in training_winows:
            for date in dates:
                for h in range(8):
                    hour_date = date + dt.timedelta(hours=h*3)
                    print "Execute outputs for one model,trw DateTime - ",name,",",trw,",",hour_date
                    rnt = np.random.randint(28,56)+trw
                    det_reg = DetRegressor(name, hour_date, dataset, clf, window=trw, alt_window=rnt,
                            is_mix_model=True, source=source)
                    det_reg.run_model()
                    pass
                pass
            pass
        pass

    pass
elif case == 1:
    _o, xparams, yparam_clf = db.load_data()
    #dataset = db.load_data(case=1)
    dataset = db.load_data_RB(case=1)
    
    source = LSTMDataSource(_o, xparams, ["K_P_LT_delay"], yparam_clf)
    clf = LSTMClassifier(source=source)

    name= "regression"
    fname = "../out/storm/prediction_RB_lstm_mixgp.csv"
    training_winows = [27]
    sdate, edate = dt.datetime(2001,1,1), dt.datetime(2011,1,1)
    ndates = int((edate-sdate).total_seconds()/(60.*60.*24.))
    print ndates*8
    dates = [dt.datetime(2001,1,1) + dt.timedelta(days=i) for i in range(ndates)]
    for trw in training_winows:
        for date in dates:
            for h in range(8):
                hour_date = date + dt.timedelta(hours=h*3)
                print "Execute outputs for one model,trw DateTime - ",name,",",trw,",",hour_date
                rnt = np.random.randint(28,56) + trw
                try:
                    dgp = DeepGPR(name, hour_date, dataset, clf, window=trw, alt_window=rnt, source=source, fname=fname)
                    dgp.run_model()
                    pass
                except: pass
                pass
            pass
        pass
    pass

    pass


