import os
import pandas as pd
import numpy as np
import datetime as dt


def get_kp(f_csv="../data_store/kp.csv",stime=None, etime=None):
    '''
    get Kp values from CSV file and filter based on dates
    '''
    _k = pd.read_csv(f_csv)
    _k.dates = pd.to_datetime(_k.dates)
    if stime is not None: _k = _k[_k.dates >= stime]
    if etime is not None: _k = _k[_k.dates < etime]
    return _k

def get_omni_data(f_csv = "../data_store/omni_3h.csv"):
    '''
    get and filter OMNI data  from csv file
    '''
    def filterX(_o):
        _o = _o[(_o.Bx_m!=9999.99) & (_o.By_m!=9999.99) & (_o.Bz_m!=9999.99) & (_o.V_m!=99999.9) & (_o.Vx_m!=99999.9)
                & (_o.Vy_m!=99999.9) & (_o.Vz_m!=99999.9) & (_o.PR_d_m!=999.99) & (_o["T_m"]!=9999999.0) & (_o.P_dyn_m!=99.99)
                & (_o.E_m!=999.99) & (_o.beta_m!=999.99) & (_o.Ma_m!=999.9)]
        return _o

    _o = pd.read_csv(f_csv)
    _o.sdates = pd.to_datetime(_o.sdates)
    _o.edates = pd.to_datetime(_o.edates)
    _o = filterX(_o)
    return _o


def transform_variables(_df):
    '''
    Transform 13 solar wind variables to 10 solar wind variables
    '''
    B_x = np.array(_df["Bx_m"]).T
    B_T = np.sqrt(np.array(_df["By_m"])**2+np.array(_df["Bz_m"])**2).T
    theta_c = np.arctan(np.array(_df["Bz_m"])/np.array(_df["By_m"])).T
    #theta_c[np.isnan(theta_c)] = 0.
    sinetheta_c2 = np.sin(theta_c/2)
    V = np.array(_df["V_m"]).T
    n = np.array(_df["PR_d_m"]).T
    T = np.array(_df["T_m"]).T
    P_dyn = np.array(_df["P_dyn_m"]).T
    beta = np.array(_df["beta_m"]).T
    M_A = np.array(_df["Ma_m"]).T
    Kp = np.array(_df["Kp"]).T
    Kplt = np.array(_df["_kp_lt"]).T
    Kpd = np.array(_df["_dkp"]).T
    Kpdlt = np.array(_df["_dkp_lt"]).T
    sdates = _df["sdates"]
    fcdates = _df["delay_time"]
    columns = ["B_x","B_T","theta_c","sin_tc","V","n","T",
            "P_dyn","beta","M_A","Date_WS","K_P","K_P_LT",
            "Date_FC","K_P_delay","K_P_LT_delay"]
    _o = pd.DataFrame(np.array([B_x,B_T,theta_c,sinetheta_c2,V,n,T,P_dyn,beta,M_A,sdates,
        Kp,Kplt,fcdates,Kpd,Kpdlt]).T,columns=columns)
    return _o

def do_transform_Kp2lin(Kp):
    '''
    Do a liniear transform on Kp [This is not a very good idea. People have done that, that's why we are also doing it]
    '''
    _levels = ["0","0+","1-","1","1+","2-","2","2+","3-","3","3+","4-","4","4+",
            "5-","5","5+","6-","6","6+","7-","7","7+","8-","8","8+","9-","9"]
    _n = 0.33
    _lin_values = [0,0+_n,1-_n,1,1+_n,2-_n,2,2+_n,3-_n,3,3+_n,4-_n,4,4+_n,
            5-_n,5,5+_n,6-_n,6,6+_n,7-_n,7,7+_n,8-_n,8,8+_n,9-_n,9]
    _dict = dict(zip(_levels,_lin_values))
    _Kp_lin = []
    for _k in Kp: _Kp_lin.append(_dict[_k])
    _Kp_lin = np.array(_Kp_lin)
    return _Kp_lin

def load_data(case=0,threshold=4.5,delay_hours=3):
    '''
    1. Load data from Omni and Kp CSV files
    2. Delay the data for <h> hours
    3. Case denotes type of data <classifier/regressor>
    4. threshold defines Kp convertion threshold for binary classifier
    '''
    fname = "../data_store/master_data_store_%d_%.2f.csv"%(delay_hours,threshold)
    if not os.path.exists(fname):
        params = ["Bx_m","By_m","Bz_m","V_m","Vx_m","Vy_m","Vz_m","PR_d_m","T_m","P_dyn_m","E_m","beta_m",
                            "Ma_m","sdates","Kp","_kp_lt","delay_time","_dkp","_dkp_lt"]
        headers = ["B_x","B_T","sin_tc","V","n","T",
                "P_dyn","beta","M_A","Date_WS","K_P","K_P_LT",
                "Date_FC","K_P_delay","K_P_LT_delay"]
        _o = get_omni_data()
        _k = get_kp()
        _dkp = []
        _kp = []
        delay_time = []
        for I,rec in _o.iterrows():
            now = rec["sdates"]
            FC_time = now + dt.timedelta(hours=delay_hours)
            delay_time.append(FC_time)
            future_kp = _k[_k.dates == FC_time]
            now_kp = _k[_k.dates == now]
            if len(future_kp) == 0: _dkp.append(_dkp[-1])
            else: _dkp.append(future_kp.Kp.tolist()[0])
            if len(now_kp) == 0: _kp.append(_kp[-1])
            else: _kp.append(now_kp.Kp.tolist()[0])
            pass
        _dkp = np.array(_dkp)
        _kp = np.array(_kp)
        _o["_dkp"] = _dkp
        _o["Kp"] = _kp
        _o["delay_time"] = delay_time
        dkp_tx = do_transform_Kp2lin(_dkp)
        _o["_dkp_lt"] = dkp_tx
        _o["_kp_lt"] = do_transform_Kp2lin(_o.Kp)
        _o = transform_variables(_o)
        stormL = np.zeros(len(_dkp))
        stormL[dkp_tx > threshold] = 1
        _o["stormL"] = stormL
        _o.to_csv(fname, index=False, header=True)
        pass
    else:
        _o = pd.read_csv(fname)
        _o.Date_FC = pd.to_datetime(_o.Date_FC)
        _o.Date_WS = pd.to_datetime(_o.Date_WS)
        pass
    _xparams = ["B_x","B_T","sin_tc","V","n","T",
            "P_dyn","beta","M_A","K_P_LT"]
    if case == 0: _yparam = ["stormL"]
    elif case == 1: _yparam = ["K_P_LT_delay"]
    with pd.option_context('mode.use_inf_as_null', True):
        _o = _o.dropna()
    return _o, _xparams, _yparam

def load_data_RB(case=0,threshold=4.5,delay_hours=3):
    '''
    1. Load data from Omni and Kp CSV files
    2. Delay the data for <h> hours
    3. Case denotes type of data <classifier/regressor>
    4. threshold defines Kp convertion threshold for binary classifier
    '''
    fname = "../data_store/master_data_store_RB_%d_%.2f.csv"%(delay_hours,threshold)
    if not os.path.exists(fname):
        params = ["Bx_m","By_m","Bz_m","V_m","Vx_m","Vy_m","Vz_m","PR_d_m","T_m","P_dyn_m","E_m","beta_m",
                            "Ma_m","sdates","Kp","_kp_lt","delay_time","_dkp","_dkp_lt"]
        headers = ["B_x","B_T","sin_tc","V","n","T",
                "P_dyn","beta","M_A","Date_WS","K_P","K_P_LT",
                "Date_FC","K_P_delay","K_P_LT_delay"]
        _g = pd.read_csv("../data_store/goes_prep_3h.csv")
        _g.date = pd.to_datetime(_g.date)
        _o = get_omni_data()
        _k = get_kp()
        _dkp = []
        _kp = []
        _goes_R = []
        _goes_B = []
        delay_time = []
        for I,rec in _o.iterrows():
            now = rec["sdates"]
            FC_time = now + dt.timedelta(hours=delay_hours)
            delay_time.append(FC_time)
            future_kp = _k[_k.dates == FC_time]
            now_kp = _k[_k.dates == now]
    
            now_g = _g[_g.date == (now - dt.timedelta(hours=12))]
            if len(now_g) == 0  and len(_goes_R) > 0:
                _goes_R.append(_goes_R[-1])
                _goes_B.append(_goes_B[-1])
            elif len(now_g) > 0:
                _goes_R.append(now_g.R.tolist()[0])
                _goes_B.append(now_g.B.tolist()[0])
            else:
                _goes_R.append(1.0)
                _goes_B.append(1e-9)
                pass

            if len(future_kp) == 0: _dkp.append(_dkp[-1])
            else: _dkp.append(future_kp.Kp.tolist()[0])
            if len(now_kp) == 0: _kp.append(_kp[-1])
            else: _kp.append(now_kp.Kp.tolist()[0])
            pass
        _dkp = np.array(_dkp)
        _kp = np.array(_kp)
        _o["_dkp"] = _dkp
        _o["Kp"] = _kp
        _o["delay_time"] = delay_time
        dkp_tx = do_transform_Kp2lin(_dkp)
        _o["_dkp_lt"] = dkp_tx
        _o["_kp_lt"] = do_transform_Kp2lin(_o.Kp)
        _o = transform_variables(_o)
        _o["goes_R"] = _goes_R
        _o["goes_B"] = np.log10(_goes_B)
        stormL = np.zeros(len(_dkp))
        stormL[dkp_tx > threshold] = 1
        _o["stormL"] = stormL
        _o.to_csv(fname, index=False, header=True)
        pass
    else:
        _o = pd.read_csv(fname)
        _o.Date_FC = pd.to_datetime(_o.Date_FC)
        _o.Date_WS = pd.to_datetime(_o.Date_WS)
        pass
    _xparams = ["B_x","B_T","sin_tc","V","n","T",
            "P_dyn","beta","M_A","K_P_LT","goes_R","goes_B"]
    if case == 0: _yparam = ["stormL"]
    elif case == 1: _yparam = ["K_P_LT_delay"]
    with pd.option_context('mode.use_inf_as_null', True):
        _o = _o.dropna()
    return _o, _xparams, _yparam
