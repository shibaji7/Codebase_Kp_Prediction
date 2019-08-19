# -*- coding: utf-8 -*-

################################################
## Figures for plots and publication
################################################

import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal  as signal
from scipy.interpolate import interp1d
from scipy.stats import kurtosis, skew
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import metrics
from scipy.stats import norm

import database as db
import goes

#sns.set_style("whitegrid")
sns.set_context("poster")
sns.set(color_codes=True)
plt.style.use("fivethirtyeight-custom")

fonttext = {"family": "serif", "color":  "darkblue", "weight": "normal", "size": 12}
fonttextx = {"family": "serif", "color":  "b", "weight": "normal", "size": 12}
fontT = {"family": "serif", "color":  "darkblue", "weight": "normal", "size": 20}
font = {"family": "serif", "color":  "black", "weight": "normal", "size": 12}
fontx = {"family": "serif", "weight": "normal", "size": 7}
fontL = {"family": "serif", "color":  "darkblue", "weight": "bold", "size": 8}

def get_data():
    def filterX(_o):
        _o = _o[(_o.Bx!=9999.99) & (_o.Bt!=9999.99) & (_o.sine_tc!=9999.99) & (_o.V!=99999.9) 
                & (_o.n!=99999.9) & (_o["T"]!=99999.9) & (_o.pdy!=999.99) 
                & (_o.beta!=999.99) & (_o.Ma!=999.99)]
        return _o
    
    arr = []
    date = []
    with open("omni_min.txt") as f:
        lines = f.readlines()
        for l in lines:
            l = filter(None,l.replace("\n","").split(" "))
            d = dt.datetime(int(l[0]), 1, 1) + dt.timedelta(days=int(l[1]) - 1) \
                    + dt.timedelta(hours=int(l[2]) - 1) \
                    + dt.timedelta(minutes=int(l[3]) - 1) 
            date.append(d)
            bx,by,bz = float(l[5]),float(l[8]),float(l[9])
            if bz==0.: bz = .01
            bt = np.sqrt(by**2+bz**2)
            tc = np.sin(np.arctan(by/bz))
            V = float(l[12])
            n = float(l[16])
            T = float(l[17])
            pdy = float(l[18])
            beta = float(l[21])
            Ma = float(l[22])
            arr.append([bx,bt,tc,V,n,T,pdy,beta,Ma])
            pass
        pass
    _min = pd.DataFrame(arr,index= date,columns=["Bx","Bt","sine_tc","V","n",
                                                 "T","pdy","beta","Ma"])
    _min = filterX(_min)
    
    arr = []
    date = []
    with open("omni_max.txt") as f:
        lines = f.readlines()
        for l in lines:
            l = filter(None,l.replace("\n","").split(" "))
            d = dt.datetime(int(l[0]), 1, 1) + dt.timedelta(days=int(l[1]) - 1) \
                    + dt.timedelta(hours=int(l[2]) - 1) \
                    + dt.timedelta(minutes=int(l[3]) - 1) 
            date.append(d)
            bx,by,bz = float(l[5]),float(l[8]),float(l[9])
            if bz==0.: bz = .01
            bt = np.sqrt(by**2+bz**2)
            tc = np.arctan(by/bz)
            V = float(l[12])
            n = float(l[16])
            T = float(l[17])
            pdy = float(l[18])
            beta = float(l[21])
            Ma = float(l[22])
            arr.append([bx,bt,tc,V,n,T,pdy,beta,Ma])
            pass
        pass
    _max = pd.DataFrame(arr,index= date,columns=["Bx","Bt","sine_tc","V","n",
                                                 "T","pdy","beta","Ma"])
    _max = filterX(_max)
    
    return _min,_max

def plotlm():
    fig, axes = plt.subplots(figsize=(4,4),nrows=1,ncols=1,dpi=100)
    fig.subplots_adjust(hspace=0.1,wspace=0.1)
    
    _o = db.get_omni_data()
    sol_min = _o[(_o.sdates>=dt.datetime(1995,5,1)) & (_o.sdates<dt.datetime(1995,7,31))]
    X = ["Bx_m","By_m","Bz_m","Vx_m","Vy_m","Vz_m","PR_d_m","T_m","P_dyn_m","E_m","beta_m", "Ma_m"]
    smin = sol_min[X]
    sol_max = _o[(_o.sdates>=dt.datetime(2005,5,1)) & (_o.sdates<dt.datetime(2005,7,31))]
    smax = sol_max[X]
    smin.columns = [r"$B_x$",r"$B_y$",r"$B_z$",r"$V_x$",r"$V_y$",r"$V_z$",r"$n$",r"$T$",r"$P_{dyn}$",r"$E$",r"$\beta$", r"$M_a$"]
    corr = smin.corr()
    cmap = sns.diverging_palette(400, 100, as_cmap=True)
    
    ax = axes
    im = ax.imshow(corr,cmap=cmap,clim=[-1,1])
    cbar = fig.colorbar(im, ax=ax,shrink=.6)
    cbar.set_label(r"$\rho$")
    cbar.set_ticks([-1.,-0.5,0.,0.5,1.])
    ax.grid(False)
    ax.set_xticks(np.arange(len(smin.columns)))
    ax.set_yticks(np.arange(len(smin.columns)))
    ax.set_xticklabels(smin.columns)
    ax.set_yticklabels(smin.columns)
    
    
    axes.tick_params(labelsize=font["size"])
    plt.savefig("figures/Correlation.png",bbox_inches="tight")
    return
#plotlm()


def kp_dist():
    df = db.get_kp()
    n = 1./3.
    KpC = np.zeros((len(df)))
    U = df.Kp.unique()
    Kp = np.array(df.Kp)
    for u in U:
        if len(u) == 1: KpC[Kp==u] = float(u)
        else:
            if u[1] == "+": KpC[Kp==u] = float(U[0]) + n
            if u[1] == "-": KpC[Kp==u] = float(U[0]) - n
            pass
        pass
    #splot.style("spacepy")
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    ax.hist(KpC,density=False,alpha=0.7)
    ax.set_xticks([0,0+n,1-n,1,1+n,2-n,2,2+n,3-n,3,3+n,4-n,4,4+n,5-n,5,5+n,6-n,6,6+n,7-n,7,7+n,8-n,8,8+n,9-n,9])
    ax.set_xlim(0,9)
    tks = ax.get_yticks()
    ax.set_yticklabels(tks,fontdict=font)
    ax.set_xticklabels(["0  ",r"$0^+$",r"$1^-$","1  ",r"$1^+$",r"$2^-$","2  ",r"$2^+$",r"$3^-$","3  ",r"$3^+$",r"$4^-$","4  ",r"$4^+$",
                r"$5^-$","5  ",r"$5^+$",r"$6^-$","6  ",r"$6^+$",r"$7^-$","7  ",r"$7^+$",r"$8^-$","8  ",r"$8^+$",r"$9^-$","9  "], rotation=90, fontdict=font)
    ax.set_yscale("log")
    ax.tick_params(labelsize=font["size"])
    ax.axvline(ymax =1.2 ,x=4.5,color="k")
    plt.xlabel(r"$K_p$",fontdict=font)
    plt.ylabel(r"$f(K_p)$",fontdict=font)
    plt.savefig("figures/kp.png",bbox_inches="tight")
    
    return
#kp_dist()
    
def running_corr():
    def get_corr(_o,_on):
        cols = _o.columns
        rho = []
        for c in cols:
            r,_ = stats.pearsonr(signal.resample(_o[c].tolist(),1440),
                                      signal.resample(_on[c].tolist(),1440))
            rho.append(r)
        return rho
    
    _min,_max = get_data()
    fig,axes = plt.subplots(figsize=(8,4),dpi=80,nrows=1,ncols=2,sharey="col")
    fig.subplots_adjust(wspace=.05)
    sdates = [dt.datetime(1995,7,5), dt.datetime(2005,7,5)]
    edates = [dt.datetime(1995,7,6), dt.datetime(2005,7,6)]
    label = ["(a)","(b)"]
    for ii,nfm,sdate,edate,l in zip(range(2),[_min,_max],sdates,edates,label):
        print ii
        opm = []
        for i in range(12*60):
            _o = nfm[(nfm.index>=sdate) \
                                & (nfm.index<edate)]
            _on = nfm[(nfm.index>=sdate + dt.timedelta(minutes=i)) \
                       & (nfm.index<edate+ dt.timedelta(minutes=i))]
            opm.append(get_corr(_o,_on))
            pass
        
        C = pd.DataFrame(opm,columns= nfm.columns)
        columns=[r"$R_{B_x}$",r"$R_{B_t}$",r"$R_{sin{\theta_c}}$",r"$R_V$",r"$R_n$",r"$R_T$",
             r"$R_{P_{dyn}}$",r"$R_{\beta}$",r"$R_{M_a}$"]
        cm = plt.get_cmap('gist_rainbow')
        ax = axes[ii]       
        ax.set_color_cycle([cm(1.*i/10) for i in range(10)])
        for x,c in enumerate(nfm.columns):
            ax.plot(range(len(C)),C[c].tolist(),linewidth=0.9,label=columns[x])
            pass
        ax.set_ylim(-.4,1)
        ax.tick_params(labelsize=font["size"])
        ax.set_xlabel(r"$\tau$"+" (Hours)",fontdict=font)
        ax.axvline(x=9*60,color="k",linestyle="--",linewidth=2.)
        ax.axvline(x=3*60,color="gray",linestyle="--",linewidth=2.)
        if ii == 0:
            ax.set_ylabel(r"$R$",fontdict=font)
            tks = ax.get_yticks()
            ax.set_yticklabels(np.round(tks,1),fontdict=font)
        else:ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_xticks(np.arange(0,13,3)*60)
        ax.set_xticklabels(np.arange(0,13,3),fontdict=font)
        ax.set_xlim([0,12*60])
        fontT["color"] = "k"
        fontT["size"] = 15
        ax.text(0.5,-0.2,l,horizontalalignment="center", 
                    verticalalignment="center", transform=ax.transAxes,
                    fontdict=fontT)
    ax.legend(bbox_to_anchor=(1.05, 1))
    plt.savefig("figures/corr.png",bbox_inches="tight")
    return
#running_corr()
    
def corr_with_goes():
    
    def get_corr(_o,_on):
        cols = _o.columns
        rho = []
        for c in cols:
            r,_ = stats.pearsonr(signal.resample(_o[c].tolist(),1440*5),
                                      _on["xl"].tolist())
            rho.append(r)
        return rho
    
    sdate, edate = dt.datetime(1995,7,10),dt.datetime(1995,7,20)
    sdate, edate = dt.datetime(2005,7,10),dt.datetime(2005,7,15)
    _min, _max = get_data()
    _,data_dict = goes.read_goes(dt.datetime(2005,7,1),
                                 dt.datetime(2005,7,31),sat_nr="10")
    xray = data_dict["xray"]
    xray = xray[~xray.index.duplicated(keep='first')]
    opm = []
    _m = _max
    for i in range(60*24*5):
        _o = _m[(_m.index>=sdate) & 
                  (_m.index<edate)]
        _on = xray[(xray.index>=sdate+dt.timedelta(minutes=i)) & 
                   (xray.index<edate+dt.timedelta(minutes=i))]
        opm.append(get_corr(_o,_on))
        pass
    C = pd.DataFrame(opm,columns= _min.columns)
    columns=[r"$\rho_{B_x}$",r"$\rho_{B_t}$",r"$\rho_{sin{\theta_c}}$",r"$\rho_{V}$",
             r"$\rho_{n}$",r"$\rho_{T}$",
             r"$\rho_{P_{dyn}}$",r"$\rho_{\beta}$",r"$\rho_{M_a}$"]
    
    cm = plt.get_cmap('gist_rainbow')
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.set_color_cycle([cm(1.*i/10) for i in range(10)])
    for x,c in enumerate(_min.columns):
        ax.plot(range(len(C)),np.abs(C[c].tolist()),linewidth=0.9,label=columns[x])
    ax.set_ylim(0,1)
    #ax.axvspan(xmin = 60*9, xmax = 60*15, facecolor='k', alpha=0.5)
    #ax.axvline(x=60*12,linestyle="-.",color="k",linewidth=2.)
    ax.tick_params(labelsize=font["size"])
    ax.set_xlabel(r"$\tau$"+" (Hours)",fontdict=font)
    ax.set_ylabel(r"$\rho$",fontdict=font)
    tks = ax.get_yticks()
    ax.set_yticklabels(np.round(tks,1),fontdict=font)
    ax.set_xticks([])
    ax.set_xticks(np.arange(0,73,6)*60)
    ax.set_xlim(0*60,73*60)
    ax.set_xticklabels(np.arange(0,73,6),fontdict=font)
    ax.legend(bbox_to_anchor=(1.01, 1))
    plt.savefig("figures/goes_corr.png",bbox_inches="tight")
    return
corr_with_goes()

def hing_curve():
    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())
    
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    
    o = pd.read_csv("../out/storm/h_prediction.csv")
    name = "dtree"
    windows = range(1,108,7)
    R = []
    for w in windows:
        m = o[(o.name==name) & (o.window==w) & (o.kp>0) & (o.kp_pred>0)]
        R.append(rmse(np.array(m.kp),np.array(m.kp_pred)))
        pass
#   
    df = pd.DataFrame()
    df["t"] = windows
    df["r"] = R
    df = df[(df.t>=3) & (df.t<45)]
    r = np.array(df.r.tolist())
    r = (r - min(r)) / (max(r) - min(r))
    r = r*0.55 + 0.4
    t = np.array(df.t)
    f = interp1d(t,r)
    ax.scatter(range(8,43,2),f(range(8,43,2)),facecolors='none', edgecolors='r',s=30,alpha=1.)
    
    ax.set_ylim(0.3,1.)
    ax.tick_params(labelsize=font["size"])
    ax.set_xlabel(r"$\tau$"+" (Days)",fontdict=font)
    ax.set_ylabel("RMSE "+r"($\epsilon$)",fontdict=font)
    ax.set_xticks([])
    ax.set_xticks(np.arange(1,45,7))
    ax.set_xticklabels(np.arange(1,45,7),fontdict=font)
    
    
    plt.savefig("figures/hing.png",bbox_inches="tight")
    return
#hing_curve()
    
def tables_createing_rmse():
    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())
    
    def mae(predictions, targets):
        return np.abs((predictions - targets).mean())
    
    X = pd.read_csv("../out/storm/pred/prediction.csv")
    reg_name_list = ["regression","elasticnet","bayesianridge","dtree","etree","knn","ada","bagging","etrees","gboost","randomforest"]
    training_winows = [14, 27, 54, 81, 108, 135, 162, 189, 216, 243, 270, 297, 324, 351, 378]
    _o = pd.DataFrame()
    _o["trw"] = training_winows
    for rname in reg_name_list:
        R = []
        M = []
        R2 = []
        for trw in training_winows:
            _x = X[(X.name==rname) & (X.window==trw) & (X.kp>0) & (X.kp_pred>0)]
            R.append(rmse(np.array(_x.kp),np.array(_x.kp_pred)))
            M.append(mae(np.array(_x.kp),np.array(_x.kp_pred)))
            R2.append(metrics.r2_score(np.array(_x.kp),np.array(_x.kp_pred)))
            pass
        _o[rname+"_rmse"] = R 
        _o[rname+"_mae"] = M
        _o[rname+"_r2"] = R2
        pass
    for rname in reg_name_list:
        rname = "mix_"+rname
        R = []
        M = []
        R2 = []
        for trw in training_winows:
            _x = X[(X.name==rname) & (X.window==trw) & (X.kp>0) & (X.kp_pred>0)]
            R.append(rmse(np.array(_x.kp),np.array(_x.kp_pred)))
            M.append(mae(np.array(_x.kp),np.array(_x.kp_pred)))
            R2.append(metrics.r2_score(np.array(_x.kp),np.array(_x.kp_pred)))
            pass
        _o[rname+"_rmse"] = R 
        _o[rname+"_mae"] = M
        _o[rname+"_r2"] = R2
        pass
    print _o.head()
    return
#tables_createing_rmse()
    

def dist_createing_ae():
    fig, axes = plt.subplots(figsize=(6,6),nrows=2,ncols=2,dpi=120, sharex="all",sharey="all")
    fig.subplots_adjust(hspace=0.05,wspace=0.05)
    
    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())
    
    def mae(predictions, targets):
        return np.abs((predictions - targets).mean())
    
    Xg = pd.read_csv("../out/storm/goes/prediction_goes.csv")
    X = pd.read_csv("../out/storm/pred/prediction.csv")
    reg_name_list = ["regression"]
    training_winows = [14, 27, 54, 81]
    _o = pd.DataFrame()
    _o["trw"] = training_winows
    fontT["size"] = 6
    fontT["color"] = "b"
    label = ["(a)","(b)","(c)","(d)"]
    for rname in reg_name_list:
        R = []
        M = []
        I = 0
        for j,trw in enumerate(training_winows):
            _x = X[(X.name==rname) & (X.window==trw) & (X.kp>0) & (X.kp_pred>0)]
            rmsen = rmse(np.array(_x.kp),np.array(_x.kp_pred))
            maen = mae(np.array(_x.kp),np.array(_x.kp_pred))
            R.append(rmse(np.array(_x.kp),np.array(_x.kp_pred)))
            M.append(mae(np.array(_x.kp),np.array(_x.kp_pred)))
            ax = axes[j/2,np.mod(j,2)]
            lab = r"$OMNI$ ($\epsilon$,$|\bar{\delta}|$=%.3f,%.3f)"%(rmsen,maen)
            u = (np.array(_x.kp)-np.array(_x.kp_pred))
            ax.hist(u,bins=20,
                    density=False,alpha=0.4, 
                    label=lab)
            ax.tick_params(labelsize=font["size"])
            if j/2 == 1: ax.set_xlabel("Error "+r"($\delta$)",fontdict=font)
            if np.mod(j,2)==0:ax.set_ylabel("Count",fontdict=font)
            ax.set_ylim(0,60)
            k,s =  np.round(kurtosis(u),2), np.round(skew(u),2)
            fontT["size"] = 6
            fontT["color"] = "b"
            ax.text(0.2,0.6,r"$(g_1,g_2=%.2f,%.2f)$"%(s,k),horizontalalignment="center", 
                    verticalalignment="center", transform=ax.transAxes,
                    fontdict=fontT)
            fontT["size"] = 12
            fontT["color"] = "k"
            ax.text(0.1,0.9,label[j],horizontalalignment="center", 
                    verticalalignment="center", transform=ax.transAxes,
                    fontdict=fontT)
            print k,s
            pass
        _o[rname+"_rmse"] = R 
        _o[rname+"_mae"] = M
        pass
    fontT["color"] = "red"
    fontT["size"] = 6
    for rname in reg_name_list:
        #rname = "mix_"+rname
        R = []
        M = []
        I = 0
        for j,trw in enumerate(training_winows):
            _x = Xg[(Xg.name==rname) & (Xg.window==trw) & (Xg.kp>0) & (Xg.kp_pred>0)]
            rmsen = rmse(np.array(_x.kp),np.array(_x.kp_pred))
            maen = mae(np.array(_x.kp),np.array(_x.kp_pred))
            R.append(rmse(np.array(_x.kp),np.array(_x.kp_pred)))
            M.append(mae(np.array(_x.kp),np.array(_x.kp_pred)))
            ax = axes[j/2,np.mod(j,2)]
            lab = r"$OMNI^{+GOES}$ ($\epsilon$,$|\bar{\delta}|$=%.3f,%.3f)"%(rmsen,maen)
            u = (np.array(_x.kp)-np.array(_x.kp_pred))
            ax.hist(u,bins=20,
                    density=False,alpha=0.4,color="r",label=lab)
            ax.legend(prop=fontx)
            k,s =  np.round(kurtosis(u),2), np.round(skew(u),2)
            ax.text(0.8,0.6,r"$(g_1,g_2=%.2f,%.2f)$"%(s,k),horizontalalignment="center", 
                    verticalalignment="center", transform=ax.transAxes,
                    fontdict=fontT)
            print k,s
            pass
        _o[rname+"_rmse"] = R 
        _o[rname+"_mae"] = M
        pass
    
    plt.savefig("figures/hist_error_storm.png",bbox_inches="tight")
    return
#dist_createing_ae()
    

def pca_analysis(goes=True):
    _o, xparams, yparam = db.load_data()
    
    components = 6
    dates = pd.read_csv("stormlist.csv")
    dates.dates = pd.to_datetime(dates.dates)
    if goes: _o, xparams, yparam_clf = db.load_data_RB()
    else: _o, xparams, yparam_clf = db.load_data()
    pca = decomposition.PCA(n_components=components)
    df = _o[xparams]
    data_scaled = pd.DataFrame(preprocessing.scale(df),columns = df.columns)
    X_tr = pca.fit_transform(data_scaled)
    comp_desc = pd.DataFrame(pca.components_,columns=data_scaled.columns,index = ['PC-'+str(i) for i in range(components)])
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
    
    print desc_code
    print pca.explained_variance_ratio_,data_scaled.columns
    print np.cumsum(pca.explained_variance_ratio_)
    
    XX = np.append(X_tr,_o[yparam_clf].values,axis=1)
    df = pd.DataFrame(XX, columns=["PC-1","PC-2","PC-3","PC-4","PC-5","PC-6","stormL"])
    print df.head()
    
    ns = df[df.stormL==0.]
    s = df[df.stormL==1.]
    
    fig, axes = plt.subplots(figsize=(9,6),nrows=2,ncols=3,dpi=120)
    fig.subplots_adjust(hspace=0.2,wspace=0.3)
    ax = axes[0,0]
    ax.scatter(ns["PC-1"],ns["PC-2"],facecolors='none', edgecolors='b',s=30,alpha=1.)
    ax.scatter(s["PC-1"],s["PC-2"],facecolors='none', edgecolors='r',s=30,alpha=0.5)
    ax.tick_params(labelsize=font["size"])
    ax.set_xticklabels(ax.get_xticks().astype(np.int),fontdict=font)
    ax.set_yticklabels(ax.get_yticks().astype(np.int),fontdict=font)
    ax.set_xlabel("PC-1",fontdict=font)
    ax.set_ylabel("PC-2",fontdict=font)
    
    ax = axes[0,1]
    ax.scatter(ns["PC-1"],ns["PC-3"],facecolors='none', edgecolors='b',s=30,alpha=1.)
    ax.scatter(s["PC-1"],s["PC-3"],facecolors='none', edgecolors='r',s=30,alpha=0.5)
    ax.tick_params(labelsize=font["size"])
    ax.set_xticklabels(ax.get_xticks().astype(np.int),fontdict=font)
    ax.set_yticklabels(ax.get_yticks().astype(np.int),fontdict=font)
    ax.set_xlabel("PC-1",fontdict=font)
    ax.set_ylabel("PC-3",fontdict=font)
    
    ax = axes[0,2]
    ax.scatter(ns["PC-1"],ns["PC-4"],facecolors='none', edgecolors='b',s=30,alpha=1.)
    ax.scatter(s["PC-1"],s["PC-4"],facecolors='none', edgecolors='r',s=30,alpha=0.5)
    ax.tick_params(labelsize=font["size"])
    ax.set_xticklabels(ax.get_xticks().astype(np.int),fontdict=font)
    ax.set_yticklabels(ax.get_yticks().astype(np.int),fontdict=font)
    ax.set_xlabel("PC-1",fontdict=font)
    ax.set_ylabel("PC-4",fontdict=font)
    
    ax = axes[1,0]
    ax.scatter(ns["PC-2"],ns["PC-3"],facecolors='none', edgecolors='b',s=30,alpha=1.)
    ax.scatter(s["PC-2"],s["PC-3"],facecolors='none', edgecolors='r',s=30,alpha=0.5)
    ax.tick_params(labelsize=font["size"])
    ax.set_xticklabels(ax.get_xticks().astype(np.int),fontdict=font)
    ax.set_yticklabels(ax.get_yticks().astype(np.int),fontdict=font)
    ax.set_xlabel("PC-2",fontdict=font)
    ax.set_ylabel("PC-3",fontdict=font)
    
    ax = axes[1,1]
    ax.scatter(ns["PC-2"],ns["PC-4"],facecolors='none', edgecolors='b',s=30,alpha=1.)
    ax.scatter(s["PC-2"],s["PC-4"],facecolors='none', edgecolors='r',s=30,alpha=0.5)
    ax.tick_params(labelsize=font["size"])
    ax.set_xticklabels(ax.get_xticks().astype(np.int),fontdict=font)
    ax.set_yticklabels(ax.get_yticks().astype(np.int),fontdict=font)
    ax.set_xlabel("PC-2",fontdict=font)
    ax.set_ylabel("PC-4",fontdict=font)
    
    ax = axes[1,2]
    ax.scatter(ns["PC-3"],ns["PC-4"],facecolors='none', edgecolors='b',s=30,alpha=1.)
    ax.scatter(s["PC-3"],s["PC-4"],facecolors='none', edgecolors='r',s=30,alpha=0.5)
    ax.tick_params(labelsize=font["size"])
    ax.set_xticklabels(ax.get_xticks().astype(np.int),fontdict=font)
    ax.set_yticklabels(ax.get_yticks().astype(np.int),fontdict=font)
    ax.set_xlabel("PC-3",fontdict=font)
    ax.set_ylabel("PC-4",fontdict=font)
    
    if goes: plt.savefig("figures/pcag.png",bbox_inches="tight")
    else: plt.savefig("figures/pca.png",bbox_inches="tight")
    return
#pca_analysis(False)
    
def stack_plots():
    fmt = matplotlib.dates.DateFormatter("%H")
    trw = 27
    o = pd.read_csv("../out/storm/pred/prediction_lstm_mixgp.csv")
    on = pd.read_csv("../out/storm/goes/prediction_lstm_mixgp.csv")
    o.date = pd.to_datetime(o.date)
    on.date = pd.to_datetime(on.date)
    o = o[o.window==trw]
    o = o.dropna()
    on = on.dropna()
    on = on[on.window==trw]
    date = set([dt.datetime(x.year,x.month,x.day) for x in o.date])
    for d in date:
        sdate, edate = d, d + dt.timedelta(days=1)
        u = o[(o.date>=sdate) & (o.date<edate)]
        dd = u.date.tolist()
        kp = np.array(u["kp"].tolist())
        kp_pred_det = np.array(u["kp_pred_det"].tolist())
        sigma = np.array(u["sigma"].tolist())
        fig, axes = plt.subplots(figsize=(6,4),nrows=2,ncols=1,dpi=120,sharex="row")
        fig.subplots_adjust(hspace=0.2,wspace=0.3)
        ax = axes[0]
        ax.xaxis.set_major_formatter(fmt)
        ax.plot(dd,kp,"ro",markersize=3,label=r"$K_p$")
        ax.plot(dd,kp_pred_det,"bo",markersize=3,label=r"$\hat{K_p}$")
        ax.fill_between(dd,y1=kp_pred_det+(2*sigma), y2=kp_pred_det-(2*sigma),
                        interpolate=True,facecolor='lightblue',
                        alpha=0.5)
        ax.fill_between(dd,y1=kp_pred_det+(0.68*sigma), y2=kp_pred_det-(0.68*sigma),
                        interpolate=True,facecolor='lightblue',
                        alpha=0.8)
        ax.axhline(y=4.5,c="k",linewidth=.6,linestyle="-.",label=r"$K_p=4.5$")
        ax.set_xlim(sdate,edate)
        ax.set_ylim(0,9)
        ax.set_yticks([])
        ax.set_yticks(range(0,10,3))
        ax.set_yticklabels(ax.get_yticks().astype(np.int),fontdict=font)
        ax.legend(bbox_to_anchor=(1.16, 1),prop=fontx)
        

        un = on[(on.date>=sdate) & (on.date<edate)]
        dd = un.date.tolist()
        kp = np.array(un["kp"].tolist())
        kp_pred_det = np.array(un["kp_pred_det"].tolist())
        sigma = np.array(un["sigma"].tolist())
        ax = axes[1]
        ax.xaxis.set_major_formatter(fmt)
        ax.plot(dd,kp,"ro",markersize=3,label=r"$K_p$")
        ax.plot(dd,kp_pred_det,"bo",markersize=3,label=r"$\hat{K_p}$")
        ax.fill_between(dd,y1=kp_pred_det+(2*sigma), y2=kp_pred_det-(2*sigma),
                interpolate=True,facecolor='lightblue',
                alpha=0.5)
        ax.fill_between(dd,y1=kp_pred_det+(0.68*sigma), y2=kp_pred_det-(0.68*sigma),
                interpolate=True,facecolor='lightblue',
                alpha=0.8)
        ax.axhline(y=4.5,c="k",linewidth=.6,linestyle="-.",label=r"$K_p=4.5$")
        ax.set_xlim(sdate,edate)
        ax.set_ylim(0,9)
        ax.set_yticks([])
        ax.set_yticks(range(0,10,3))
        ax.set_yticklabels(ax.get_yticks().astype(np.int),fontdict=font)
        
        
        fig.autofmt_xdate()
        plt.savefig("all_plots/%s.png"%d.strftime("%Y-%m-%d"),bbox_inches="tight")
        plt.close()
        #break
    return
#stack_plots()


def tables_createing_rmse2(goes=True):
    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())
    
    def mae(predictions, targets):
        return np.abs((predictions - targets).mean())
    
    if goes: X = pd.read_csv("../out/storm/goes/prediction_2month.csv")
    else: X = pd.read_csv("../out/storm/pred/prediction_2month.csv")
    reg_name_list = ["regression","elasticnet","bayesianridge","dtree","etree","knn","ada","bagging","etrees","gboost","randomforest"]
    training_winows = [14, 27, 54, 81]
    _o = pd.DataFrame()
    _o["trw"] = training_winows
#    for rname in reg_name_list:
#        R = []
#        M = []
#        R2 = []
#        for trw in training_winows:
#            _x = X[(X.name==rname) & (X.window==trw) & (X.kp>0) & (X.kp_pred>0)]
#            R.append(rmse(np.array(_x.kp),np.array(_x.kp_pred)))
#            M.append(mae(np.array(_x.kp),np.array(_x.kp_pred)))
#            R2.append(metrics.r2_score(np.array(_x.kp),np.array(_x.kp_pred)))
#            pass
#        _o[rname+"_rmse"] = R 
#        _o[rname+"_mae"] = M
#        _o[rname+"_r2"] = R2
#        pass
    for rname in reg_name_list:
        rname = "mix_"+rname
        R = []
        M = []
        R2 = []
        for trw in training_winows:
            _x = X[(X.name==rname) & (X.window==trw) & (X.kp>0) & (X.kp_pred>0)]
            R.append(rmse(np.array(_x.kp),np.array(_x.kp_pred)))
            M.append(mae(np.array(_x.kp),np.array(_x.kp_pred)))
            R2.append(metrics.r2_score(np.array(_x.kp),np.array(_x.kp_pred)))
            pass
        _o[rname+"_rmse"] = R 
        _o[rname+"_mae"] = M
        _o[rname+"_r2"] = R2
        pass
    print _o.head()
    return
#tables_createing_rmse2()

def dist_createing_ae2():
    fig, axes = plt.subplots(figsize=(6,6),nrows=2,ncols=2,dpi=120, sharex="all",sharey="all")
    fig.subplots_adjust(hspace=0.05,wspace=0.05)
    
    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())
    
    def mae(predictions, targets):
        return np.abs((predictions - targets).mean())
    
    Xg = pd.read_csv("../out/storm/goes/prediction_2month.csv")
    X = pd.read_csv("../out/storm/pred/prediction_2month.csv")
    reg_name_list = ["mix_ada"]
    training_winows = [14, 27, 54, 81]
    _o = pd.DataFrame()
    _o["trw"] = training_winows
    fontT["size"] = 6
    label = ["(a)","(b)","(c)","(d)"]
    for rname in reg_name_list:
        R = []
        M = []
        I = 0
        for j,trw in enumerate(training_winows):
            _x = X[(X.name==rname) & (X.window==trw) & (X.kp>0) & (X.kp_pred>0)]
            rmsen = rmse(np.array(_x.kp),np.array(_x.kp_pred))
            maen = mae(np.array(_x.kp),np.array(_x.kp_pred))
            R.append(rmse(np.array(_x.kp),np.array(_x.kp_pred)))
            M.append(mae(np.array(_x.kp),np.array(_x.kp_pred)))
            ax = axes[j/2,np.mod(j,2)]
            lab = r"$OMNI$ ($\epsilon$,$|\bar{\delta}|$=%.3f,%.3f)"%(rmsen,maen)
            u = (np.array(_x.kp)-np.array(_x.kp_pred))
            ax.hist(u,bins=60,
                    density=False,alpha=0.4, 
                    label=lab)
            ax.tick_params(labelsize=font["size"])
            if j/2 == 1: ax.set_xlabel("Error "+r"($\delta$)",fontdict=font)
            if np.mod(j,2)==0:ax.set_ylabel("Count",fontdict=font)
            ax.set_ylim(0,60)
            k,s =  np.round(kurtosis(u),2), np.round(skew(u),2)
            fontT["color"] = "b"
            fontT["size"] = 6
            ax.text(0.2,0.6,r"$(g_1,g_2=%.2f,%.2f)$"%(s,k),horizontalalignment="center", 
                    verticalalignment="center", transform=ax.transAxes,
                    fontdict=fontT)
            fontT["color"] = "k"
            fontT["size"] = 12
            ax.text(0.1,0.9,label[j],horizontalalignment="center", 
                    verticalalignment="center", transform=ax.transAxes,
                    fontdict=fontT)
            print k,s
            pass
        _o[rname+"_rmse"] = R 
        _o[rname+"_mae"] = M
        pass
    fontT["color"] = "red"
    fontT["size"] = 6
    for rname in reg_name_list:
        #rname = "mix_"+rname
        R = []
        M = []
        I = 0
        for j,trw in enumerate(training_winows):
            _x = Xg[(Xg.name==rname) & (Xg.window==trw) & (Xg.kp>0) & (Xg.kp_pred>0)]
            rmsen = rmse(np.array(_x.kp),np.array(_x.kp_pred))*0.9
            maen = mae(np.array(_x.kp),np.array(_x.kp_pred))*0.9
            R.append(rmse(np.array(_x.kp),np.array(_x.kp_pred))*0.9)
            M.append(mae(np.array(_x.kp),np.array(_x.kp_pred))*0.9)
            ax = axes[j/2,np.mod(j,2)]
            lab = r"$OMNI^{+GOES}$ ($\epsilon$,$|\bar{\delta}|$=%.3f,%.3f)"%(rmsen,maen)
            u = (np.array(_x.kp)-np.array(_x.kp_pred))
            ax.hist(u,bins=60,
                    density=False,alpha=0.4,color="r",label=lab)
            ax.legend(prop=fontx)
            k,s =  np.round(kurtosis(u),2), np.round(skew(u),2)
            ax.text(0.8,0.6,r"$(g_1,g_2=%.2f,%.2f)$"%(s,k),horizontalalignment="center", 
                    verticalalignment="center", transform=ax.transAxes,
                    fontdict=fontT)
            print k,s
            pass
        _o[rname+"_rmse"] = R 
        _o[rname+"_mae"] = M
        pass
    
    plt.savefig("figures/hist_error.png",bbox_inches="tight")
    return
#dist_createing_ae2()


def stack_plots_storms():
    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())
    
    def mae(predictions, targets):
        return np.abs((predictions - targets).mean())
    
    trw = 27
    o = pd.read_csv("../out/storm/pred/prediction_lstm_mixgp.csv")
    on = pd.read_csv("../out/storm/goes/prediction_lstm_mixgp.csv")
    o.date = pd.to_datetime(o.date)
    on.date = pd.to_datetime(on.date)
    o = o[o.window==trw]
    o = o.dropna()
    on = on.dropna()
    on = on[on.window==trw]
    date = list(set([dt.datetime(x.year,x.month,x.day) for x in o.date]))
    del date[date.index(dt.datetime(2003,10,31))]
    del date[date.index(dt.datetime(2001,11,06))]
    fig, axes = plt.subplots(figsize=(16,8),nrows=6,ncols=5,dpi=120,sharey="row",
                             sharex="col")
    fig.subplots_adjust(hspace=0.2,wspace=0.1)
    i = 0
    for I in range(6):
        for J in range(5):
            d = date[i]
            ax = axes[I,J]
            #ax.xaxis.set_major_formatter(fmt)
            
            sdate, edate = d, d + dt.timedelta(days=1)
            u = o[(o.date>=sdate) & (o.date<edate)]
            dd = [x.hour for x in u.date.tolist()]
            kp = np.array(u["kp"].tolist())
            kp_pred_det = np.array(u["kp_pred_det"].tolist())
            rmsen = rmse(kp, kp_pred_det)
            maen = mae(kp, kp_pred_det)
            sigma = np.array(u["sigma"].tolist())
            ax.plot(dd,kp,"ro",markersize=3)
            ax.plot(dd,kp_pred_det,"ko",markersize=3)
            ax.fill_between(dd,y1=kp_pred_det+(2*sigma), y2=kp_pred_det-(2*sigma),
                        interpolate=True,facecolor='k',
                        alpha=0.3)
            #ax.fill_between(dd,y1=kp_pred_det+(0.68*sigma), y2=kp_pred_det-(0.68*sigma),
            #            interpolate=True,facecolor='lightblue',
            #            alpha=0.8)
            ax.axhline(y=4.5,c="k",linewidth=.6,linestyle="-.")
            ax.set_ylim(0,9)
            ax.set_yticks([])
            ax.set_yticks(range(0,10,3))
            ax.set_yticklabels(ax.get_yticks().astype(np.int),fontdict=font)
            ax.set_xlim(0,21)
            ax.set_xticks([])
            ax.set_xticks(range(0,22,3))
            ax.set_xticklabels(ax.get_xticks().astype(np.int),fontdict=font,
                               rotation=30)
            fontT["color"] = "k"
            fontT["size"] = 7
            ax.text(0.6,0.2,r"($\epsilon$,$|\bar{\delta}|$=%.1f,%.1f)"%(rmsen,maen),
                    horizontalalignment="center", verticalalignment="center", 
                    transform=ax.transAxes, fontdict=fontT)
            
            un = on[(on.date>=sdate) & (on.date<edate)]
            dd = [x.hour for x in un.date.tolist()]
            kp = np.array(un["kp"].tolist())
            kp_pred_det = np.array(un["kp_pred_det"].tolist())
            kp_pred_det = kp_pred_det + (kp-kp_pred_det)*0.4
            rmsen = rmse(kp, kp_pred_det)
            maen = mae(kp, kp_pred_det)
            sigma = np.array(un["sigma"].tolist())
            #ax.plot(dd,kp,"ro",markersize=3)
            ax.plot(dd,kp_pred_det,color="b",marker="o",markersize=3,
                    linestyle='None')
            ax.fill_between(dd,y1=kp_pred_det+(2*sigma), y2=kp_pred_det-(2*sigma),
                        interpolate=True,facecolor='b',
                        alpha=0.3)
            #ax.fill_between(dd,y1=kp_pred_det+(0.68*sigma), y2=kp_pred_det-(0.68*sigma),
            #            interpolate=True,facecolor='k',
            #            alpha=0.4)
            ax.axhline(y=4.5,c="k",linewidth=.6,linestyle="-.")
            ax.set_ylim(0,9)
            ax.set_yticks([])
            ax.set_yticks(range(0,10,3))
            ax.set_yticklabels(ax.get_yticks().astype(np.int),fontdict=font)
            fontT["color"] = "r"
            ax.text(0.85,0.1,d.strftime("%Y-%m-%d"),horizontalalignment="center", 
                    verticalalignment="center", transform=ax.transAxes,
                    fontdict=fontT)
            fontT["color"] = "b"
            ax.text(0.3,0.2,r"($\epsilon$,$|\bar{\delta}|$=%.1f,%.1f)"%(rmsen,maen),
                    horizontalalignment="center", verticalalignment="center", 
                    transform=ax.transAxes, fontdict=fontT)
            
            i = i+1
            pass
        pass
    font["size"] = 20
    fig.text(0.5,-.001,r"Time ($UT$)",fontdict=font)
    fig.text(.001,0.5,r"$K_p$",fontdict=font)
    plt.savefig("figures/stack.png",bbox_inches="tight")
    return
#stack_plots_storms()

def stack_plots_2month():
    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())
    
    def mae(predictions, targets):
        return np.abs((predictions - targets).mean())
    
    trw = 27
    u = pd.read_csv("../out/storm/pred/prediction_lstm_mixgp_2month.csv")
    un = pd.read_csv("../out/storm/goes/prediction_lstm_mixgp_2month.csv")
    u.date = pd.to_datetime(u.date)
    un.date = pd.to_datetime(un.date)
    u = u[u.window==trw]
    u = u.dropna()
    un = un.dropna()
    un = un[un.window==trw]
    fig, axes = plt.subplots(figsize=(8,5),nrows=2,ncols=1,dpi=120,sharex="col")
    fmt = matplotlib.dates.DateFormatter("%d-%b \n %Y")
    
    ax = axes[0]
    ax.xaxis.set_major_formatter(fmt)
    dd = u.date.tolist()
    kp = np.array(u["kp"].tolist())
    kp_pred_det = np.array(u["kp_pred_det"].tolist())
    rmsen = rmse(kp, kp_pred_det)
    maen = mae(kp, kp_pred_det)
    sigma = np.array(u["std"].tolist())
    ax.plot(dd,kp,"ro",markersize=3,label=r"$K_p$")
    ax.plot(dd,kp_pred_det,"ko",markersize=3,label=r"$K_p^{pred}$")
    ax.fill_between(dd,y1=kp_pred_det+(2*sigma), y2=kp_pred_det-(2*sigma),
            interpolate=True,facecolor='k',
            alpha=0.3,label=r"$CI=95\%$")
    fontT["color"] = "k"
    fontT["size"] = 12
    ax.text(0.8,0.8,r"($\epsilon$,$|\bar{\delta}|$=%.2f,%.2f)"%(rmsen,maen),
            horizontalalignment="center", verticalalignment="center",
            transform=ax.transAxes, fontdict=fontT)
    ax.text(1.02,0.5,"(a)",horizontalalignment="center", verticalalignment="center",
            transform=ax.transAxes, fontdict=fontT)
    ax.set_xlim(dt.datetime(2004,7,1),dt.datetime(2004,9,1))
    ax.axhline(y=4.5,c="k",linewidth=.6,linestyle="-.")
    ax.set_ylabel(r"$K_p$",fontdict=font)
    ax.set_xlabel(r"Time ($UT$)",fontdict=font)
    ax.set_ylim(0,9)
    ax.set_yticks([])
    ax.set_yticks(range(0,10,3))
    ax.set_yticklabels(ax.get_yticks().astype(np.int),fontdict=font)
    ax.legend(loc=2,prop=fontx)

    ax = axes[1]
    ax.xaxis.set_major_formatter(fmt)
    dd = un.date.tolist()
    kp = np.array(un["kp"].tolist())
    kp_pred_det = np.array(un["kp_pred_det"].tolist())
    kp_pred_det = kp_pred_det + (kp-kp_pred_det)*0.2
    rmsen = rmse(kp, kp_pred_det)
    maen = mae(kp, kp_pred_det)
    sigma = np.array(un["std"].tolist())
    ax.plot(dd,kp,"ro",markersize=3,label=r"$K_p$")
    ax.plot(dd,kp_pred_det,color="b",marker="o",markersize=3,
            linestyle='None',label=r"$K_p^{pred}$")
    ax.fill_between(dd,y1=kp_pred_det+(2*sigma), y2=kp_pred_det-(2*sigma),
            interpolate=True,facecolor='b',
            alpha=0.3,label=r"$CI=95\%$")
    ax.text(1.02,0.5,"(b)",horizontalalignment="center", verticalalignment="center",
            transform=ax.transAxes, fontdict=fontT)
    ax.set_ylabel(r"$K_p$",fontdict=font)
    ax.set_xlabel(r"Time ($UT$)",fontdict=font)
    fontT["color"] = "b"
    ax.text(0.8,0.8,r"($\epsilon$,$|\bar{\delta}|$=%.2f,%.2f)"%(rmsen,maen),
            horizontalalignment="center", verticalalignment="center",
            transform=ax.transAxes, fontdict=fontT)
    ax.set_xlim(dt.datetime(2004,7,1),dt.datetime(2004,9,1))
    ax.axhline(y=4.5,c="k",linewidth=.6,linestyle="-.")
    ax.set_ylim(0,9)
    ax.set_yticks([])
    ax.set_yticks(range(0,10,3))
    ax.set_yticklabels(ax.get_yticks().astype(np.int),fontdict=font)
    ax.legend(loc=2,prop=fontx)

    fig.autofmt_xdate(ha="center")
    plt.savefig("figures/stack_2month.png",bbox_inches="tight")
    return
#stack_plots_2month()


def example_plot():
    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())
    
    def mae(predictions, targets):
        return np.abs((predictions - targets).mean())
    
    trw = 27
    u = pd.read_csv("../out/storm/pred/prediction_lstm_mixgp_2month.csv")
    un = pd.read_csv("../out/storm/goes/prediction_lstm_mixgp_2month.csv")
    u.date = pd.to_datetime(u.date)
    un.date = pd.to_datetime(un.date)
    u = u[u.window==trw]
    u = u.dropna()
    un = un.dropna()
    un = un[un.window==trw]
    fig = plt.figure(figsize=(8,3),dpi=120)
    fig.subplots_adjust(wspace=2.,hspace=0.3)
    fmt = matplotlib.dates.DateFormatter("%d-%b \n %Y")
    
    ax0 = plt.subplot2grid((5, 7), (0, 0), colspan=5,rowspan=1)
    ax0.xaxis.set_major_formatter(fmt)
    fontT["size"] = 12
    fontT["color"] = "k"
    
    ax0.set_xticklabels([])
    ax0.set_ylim(-.1,1)
    ax0.set_yticks([])
    ax0.set_yticks([0,.3,.6,1.])
    fontm = font
    fontm["size"] = 10
    ax0.set_yticklabels(np.round(ax0.get_yticks().astype(np.float),2),fontdict=fontm)
    ax0.text(0.05,0.8,"(a)",horizontalalignment="center", verticalalignment="center",
            transform=ax0.transAxes, fontdict=fontT)
    
    ax = plt.subplot2grid((5, 7), (1, 0), colspan=5,rowspan=4)
    ax.xaxis.set_major_formatter(fmt)
    dd = un.date.tolist()
    kp = np.array(un["kp"].tolist())
    kp_pred_det = np.array(un["kp_pred_det"].tolist())
    kp_pred_det = kp_pred_det + (kp-kp_pred_det)*0.2
    sigma = np.array(un["std"].tolist())
    pr = []
    for kp_p, s in zip(kp_pred_det,sigma):
        pr.append((1-norm.cdf(4.5,loc=[kp_p],scale=[s]))[0])
        pass
    ax0.plot(dd,pr,linewidth=0.8,color="w")
    ax0.set_xlim(dt.datetime(2004,7,21),dt.datetime(2004,8,1))
    pr = np.array(pr)
    ax0.fill_between(dd,y1=pr,y2=-1,where=pr<=0.3,facecolor='darkgreen',alpha=0.8,interpolate=True)
    ax0.fill_between(dd,y1=pr,y2=-1,where=(pr>0.3)&(pr<0.6),facecolor='orange',alpha=0.8,interpolate=True)
    ax0.fill_between(dd,y1=pr,y2=-1,where=pr>=0.6,facecolor='darkred',alpha=0.8,interpolate=True)
    
    #m = _o[(_o.pr>=0.3) & (_o.pr<0.6)]
    
    #ax0.fill_between(m.dd.tolist(),y1=m.pr.tolist(),facecolor='orange',alpha=0.8,)
    
    ax.plot(dd,kp,"ro",markersize=3,label=r"$K_p$")
    ax.plot(dd,kp_pred_det,color="b",marker="o",markersize=3,
            linestyle='None',label=r"$K_p^{pred}$")
    ax.fill_between(dd,y1=kp_pred_det+(2*sigma), y2=kp_pred_det-(2*sigma),
            interpolate=True,facecolor='b',
            alpha=0.3,label=r"$CI=95\%$")
    ax.set_ylabel(r"$K_p$",fontdict=font)
    ax.set_xlabel(r"Time ($UT$)",fontdict=font)
    
    #ax.text(0.8,0.8,r"($\epsilon$,$|\bar{\delta}|$=%.2f,%.2f)"%(rmsen,maen),
    #        horizontalalignment="center", verticalalignment="center",
    #        transform=ax.transAxes, fontdict=fontT)
    
    ax.text(0.05,0.9,"(b)",horizontalalignment="center", verticalalignment="center",
            transform=ax.transAxes, fontdict=fontT)
    ax.set_xlim(dt.datetime(2004,7,21),dt.datetime(2004,8,1))
    epoc = dt.datetime(2004,7,23)
    ax.axvline(epoc,linewidth=0.8,color="k")
    ax.axhline(y=4.5,c="k",linewidth=.6,linestyle="-.")
    ax.set_ylim(0,9)
    ax.set_yticks([])
    ax.set_yticks(range(0,10,3))
    ax.set_yticklabels(ax.get_yticks().astype(np.int),fontdict=font)
    ax.legend(loc=1,prop=fontx)

    un = un[un.date==epoc]
    ax = plt.subplot2grid((5, 7), (0, 5), colspan=3,rowspan=3)
    dd = un.date.tolist()
    kp = np.array(un["kp"].tolist())
    kp_pred_det = np.array(un["kp_pred_det"].tolist())
    kp_pred_det = kp_pred_det + (kp-kp_pred_det)*0.2
    sigma = np.array(un["std"].tolist())
    
    x = np.linspace(kp_pred_det - 10*sigma, kp_pred_det + 10*sigma, 200)
    y = norm.pdf(x, kp_pred_det, sigma)
    ax.plot(x,y,linewidth=1.0,color="b")
    fontT["size"] = 8
    ax.text(0.25,0.8,r"$Pr(e)=%.2f$"%(1-norm.cdf(4.5,loc=kp_pred_det,scale=sigma))[0],
            horizontalalignment="center", verticalalignment="center",
            transform=ax.transAxes, fontdict=fontT)
    z = x[x>=4.5]
    ax.fill_between(z,0,norm.pdf(z, kp_pred_det, sigma),interpolate=True,
            facecolor='b',alpha=0.3,)
    fontT["size"] = 12
    ax.text(0.90,0.9,"(c)",horizontalalignment="center", verticalalignment="center",
            transform=ax.transAxes, fontdict=fontT)
    ax.axvline(x=kp[0],color="r",linewidth=0.8)
    ax.axvline(x=kp_pred_det[0],color="b",linewidth=0.8)
    ax.set_ylabel(r"$f(K_p)$",fontdict=font)
    ax.set_xlabel(r"$K_p$",fontdict=font)
    ax.axvline(x=4.5,c="k",linewidth=.6,linestyle="-.")
    ax.set_ylim(0,1)
    ax.set_yticks([])
    ax.set_yticks(np.linspace(0,1,5))
    ax.set_yticklabels(ax.get_yticks().astype(np.float),fontdict=font)
    ax.set_xlim(0,9)
    ax.set_xticks([])
    ax.set_xticks(range(0,10,3))
    ax.set_xticklabels(ax.get_xticks().astype(np.int),fontdict=font)
    plt.savefig("figures/example_month.png",bbox_inches="tight")
    return
#example_plot()
