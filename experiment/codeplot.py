# -*- coding: utf-8 -*-

################################################
## Figures for plots and publication
################################################

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal  as signal
from scipy.interpolate import interp1d
from scipy.stats import kurtosis, skew
from scipy import stats
from sklearn.metrics import r2_score
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import metrics
from scipy.stats import norm
import verify

import spacepy.plot as splot

import database as db
import goes

## ['polar', 'altgrid', 'default', 'spacepy_altgrid', 'spacepy_polar', 'spacepy']
splot.style("spacepy_altgrid")

fonttext = {"family": "serif", "color":  "darkblue", "weight": "normal", "size": 12}
fonttextx = {"family": "serif", "color":  "b", "weight": "normal", "size": 12}
fontT = {"family": "serif", "color":  "darkblue", "weight": "normal", "size": 10}
font = {"family": "serif", "color":  "black", "weight": "normal", "size": 12}
fontx = {"family": "serif", "weight": "normal", "size": 9}
fontL = {"family": "serif", "color":  "darkblue", "weight": "bold", "size": 8}


fontT = {"family": "serif", "color":  "darkblue", "weight": "normal", "size": 8}
font = {"family": "serif", "color":  "black", "weight": "normal", "size": 10}
fontx = {"family": "serif", "weight": "normal", "size": 8}
fontL = {"family": "serif", "color":  "darkblue", "weight": "bold", "size": 8}
from matplotlib import font_manager
ticks_font = font_manager.FontProperties(family="serif", size=10, weight="normal")
matplotlib.rcParams['xtick.color'] = "k"
matplotlib.rcParams['ytick.color'] = "k"
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['mathtext.default'] = "default"

def get_data():
    def filterX(_o):
        _o = _o[(_o.Bx!=9999.99) & (_o.Bt!=9999.99) & (_o.sine_tc!=9999.99) & (_o.V!=99999.9)
                & (_o.n!=99999.9) & (_o["T"]!=99999.9) & (_o.pdy!=999.99)
                & (_o.beta!=999.99) & (_o.Ma!=999.99)]
        return _o

    arr = []
    date = []
    with open("../CodeBase/omni_min.txt") as f:
        lines = f.readlines()
        for l in lines:
            l = list(filter(None,l.replace("\n","").split(" ")))
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
    with open("../CodeBase/omni_max.txt") as f:
        lines = f.readlines()
        for l in lines:
            l = list(filter(None, l.replace("\n", "").split(" ")))
            d = dt.datetime(int(l[0]), 1, 1) + dt.timedelta(days=int(l[1]) - 1) \
                + dt.timedelta(hours=int(l[2]) - 1) \
                + dt.timedelta(minutes=int(l[3]) - 1)
            date.append(d)
            bx, by, bz = float(l[5]), float(l[8]), float(l[9])
            if bz == 0.: bz = .01
            bt = np.sqrt(by ** 2 + bz ** 2)
            tc = np.arctan(by / bz)
            V = float(l[12])
            n = float(l[16])
            T = float(l[17])
            pdy = float(l[18])
            beta = float(l[21])
            Ma = float(l[22])
            arr.append([bx, bt, tc, V, n, T, pdy, beta, Ma])
            pass
        pass
    _max = pd.DataFrame(arr, index=date, columns=["Bx", "Bt", "sine_tc", "V", "n",
                                                  "T", "pdy", "beta", "Ma"])
    _max = filterX(_max)

    return _min, _max

def plotlm():
    #matplotlib.rcParams['axes.labelweight'] = "k"
    font = {"family": "serif", "color":  "black", "weight": "normal", "size": 10}
    #matplotlib.rcParams['axes.labelweight'] = "normal"
    matplotlib.rcParams['mathtext.default'] = "default"


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
    cbar.set_label(r"$\rho$", fontdict=font)
    cbar.set_ticks([-1.,-0.5,0.,0.5,1.])
    cbar.ax.set_yticklabels([-1.,-0.5,0.,0.5,1.], fontdict=font)
    ax.grid(False)
    ax.set_xticks(np.arange(len(smin.columns)))
    ax.set_yticks(np.arange(len(smin.columns)))
    ax.set_xticklabels(smin.columns, fontdict=font)
    ax.set_yticklabels(smin.columns, fontdict=font)

    plt.savefig("../Correlation.png",bbox_inches="tight")
    return
#plotlm()

def kp_dist():
    font = {"family": "serif", "color":  "black", "weight": "normal", "size": 10}
    matplotlib.rcParams['mathtext.default'] = "default"

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
    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(111)
    ax.hist(KpC,alpha=0.7)
    ax.set_xticks([0,0+n,1-n,1,1+n,2-n,2,2+n,3-n,3,3+n,4-n,4,4+n,5-n,5,5+n,6-n,6,6+n,7-n,7,7+n,8-n,8,8+n,9-n,9])
    ax.set_xlim(0,9)
    tks = ax.get_yticks()
    ax.set_yticklabels(tks,fontdict=font)
    ax.set_xticklabels(["0  ",r"$0^+$",r"$1^-$","1  ",r"$1^+$",r"$2^-$","2  ",r"$2^+$",r"$3^-$","3  ",r"$3^+$",r"$4^-$","4  ",r"$4^+$",
                r"$5^-$","5  ",r"$5^+$",r"$6^-$","6  ",r"$6^+$",r"$7^-$","7  ",r"$7^+$",r"$8^-$","8  ",r"$8^+$",r"$9^-$","9  "], rotation=90, fontdict=font)
    ax.set_yscale("log")
    ax.tick_params(labelsize=font["size"])
    ax.axvline(ymax =1.2 ,x=5-n,color="k")
    plt.xlabel(r"$K_p$",fontdict=font)
    plt.ylabel(r"$f(K_p)$",fontdict=font)
    plt.savefig("../kp.png",bbox_inches="tight")
    return
#kp_dist()

def running_corr():
    fontT = {"family": "serif", "color":  "darkblue", "weight": "normal", "size": 8}
    font = {"family": "serif", "color":  "black", "weight": "normal", "size": 10}
    fontx = {"family": "serif", "weight": "normal", "size": 8}
    fontL = {"family": "serif", "color":  "darkblue", "weight": "bold", "size": 8}
    from matplotlib import font_manager
    ticks_font = font_manager.FontProperties(family="serif", size=10, weight="normal")
    matplotlib.rcParams['xtick.color'] = "k"
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['mathtext.default'] = "default"

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
        print(ii)
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
        for x, c in enumerate(nfm.columns):
            ax.plot(range(len(C)), C[c].tolist(), linewidth=0.9, label=columns[x])
            pass
        ax.set_ylim(-.4, 1)
        ax.tick_params(labelsize=font["size"])
        ax.set_xlabel(r"$\tau$" + " (Hours)", fontdict=font)
        #ax.axvline(x=9 * 60, color="k", linestyle="--", linewidth=2.)
        ax.axvline(x=3 * 60, color="k", linestyle="--", linewidth=2.)
        if ii == 0:
            ax.set_ylabel(r"$R$", fontdict=font)
            tks = ax.get_yticks()
            ax.set_yticklabels(np.round(tks, 1), fontdict=font)
        else:
            ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_xticks(np.arange(0, 13, 3) * 60)
        ax.set_xticklabels(np.arange(0, 13, 3), fontdict=font)
        ax.set_xlim([0, 12 * 60])
        fontT["color"] = "k"
        fontT["size"] = 15
        ax.text(0.5, -0.2, l, horizontalalignment="center",
                verticalalignment="center", transform=ax.transAxes,
                fontdict=fontT)
    ax.legend(bbox_to_anchor=(1.05, 1))
    plt.savefig("../corr.png", bbox_inches="tight")
    return
#running_corr()

def corr_with_rb():

    fontT = {"family": "serif", "color":  "darkblue", "weight": "normal", "size": 8}
    font = {"family": "serif", "color":  "black", "weight": "normal", "size": 10}
    fontx = {"family": "serif", "weight": "normal", "size": 8}
    fontL = {"family": "serif", "color":  "darkblue", "weight": "bold", "size": 8}
    from matplotlib import font_manager
    ticks_font = font_manager.FontProperties(family="serif", size=10, weight="normal")
    matplotlib.rcParams['xtick.color'] = "k"
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['mathtext.default'] = "default"

    cols = ['B_x', 'B_T', 'sin_tc', 'V', 'n', 'T', 'P_dyn', 'beta', 'M_A']
    d = 5
    def get_corr(_o,_on,k):
        rho = []
        for c in cols:
            r,_ = stats.pearsonr(signal.resample(_o[c].tolist(),8*d),
                    signal.resample(_on[k].tolist(), 8*d))
            rho.append(r)
        return rho

    _X, _xparams, _yparam = db.load_data_RB(1, 4.5, 3)
    sdate = dt.datetime(2003, 7, 5)
    edate = dt.datetime(2003, 7, 5+d)
    
    columns = [r"$\rho_{B_x}$", r"$\rho_{B_t}$", r"$\rho_{sin{\theta_c}}$", r"$\rho_{V}$",
            r"$\rho_{n}$", r"$\rho_{T}$",
            r"$\rho_{P_{dyn}}$", r"$\rho_{\beta}$", r"$\rho_{M_a}$"]
    cm = plt.get_cmap('gist_rainbow')
    fig = plt.figure(figsize=(6, 3),dpi=80)
    fig.subplots_adjust(wspace=0.3)

    o = 0
    labes = ["(a)","(b)"]
    for k,u,l in zip(["goes_B","goes_R"],["B","R"],labes):
        opm = []
        ax = fig.add_subplot(121+o)
        ax.set_color_cycle([cm(1. * i / 10) for i in range(10)])
        for i in range(8*4):
            _o = _X[(_X.Date_FC >= sdate) &
                    (_X.Date_FC < edate)]
            _on = _X[(_X.Date_FC >= sdate+dt.timedelta(hours=3*i)) &
                    (_X.Date_FC < edate+dt.timedelta(hours=3*i))]
            opm.append(get_corr(_o,_on,k))
            pass
        C = pd.DataFrame(opm,columns=cols)
        print C.head()
        for x, c in enumerate(cols):
            if o==0:  ax.plot(np.arange(len(C))*3, C[c].tolist(), linewidth=0.9)
            else:  ax.plot(np.arange(len(C))*3, C[c].tolist(), linewidth=0.9, label=columns[x])
        ax.set_ylim(-1, 1)
    
        ax.tick_params(labelsize=font["size"])
        ax.set_xlabel(r"$\tau$" + " (Hours)", fontdict=font)
        ax.set_ylabel(r"$\rho^%s$"%u, fontdict=font)
        tks = ax.get_yticks()
        ax.set_yticklabels(np.round(tks, 1), fontdict=font)
        ax.set_xlim(0,72)
        #ax.set_xticks([])
        #ax.set_xticks(np.arange(0, 73, 6) * 60)
    #ax.set_xlim(0 * 60, 73 * 60)
    #ax.set_xticklabels(np.arange(0, 73, 6), fontdict=font)
        if o==1: fig.legend(bbox_to_anchor=(1.1, 1.))
        o = o+1
        fontT["size"] = 12
        fontT["color"] = "k"
        ax.text(0.9,0.9, l, horizontalalignment="center", verticalalignment="center",
                             transform=ax.transAxes, fontdict=fontT)
        pass

    fig.savefig("../corr_goes_rb.png", bbox_inches="tight")
    return
#corr_with_rb()

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
    columns = [r"$\rho_{B_x}$", r"$\rho_{B_t}$", r"$\rho_{sin{\theta_c}}$", r"$\rho_{V}$",
               r"$\rho_{n}$", r"$\rho_{T}$",
               r"$\rho_{P_{dyn}}$", r"$\rho_{\beta}$", r"$\rho_{M_a}$"]

    cm = plt.get_cmap('gist_rainbow')
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.set_color_cycle([cm(1. * i / 10) for i in range(10)])
    for x, c in enumerate(_min.columns):
        ax.plot(range(len(C)), np.abs(C[c].tolist()), linewidth=0.9, label=columns[x])
    ax.set_ylim(0, 1)
    # ax.axvspan(xmin = 60*9, xmax = 60*15, facecolor='k', alpha=0.5)
    # ax.axvline(x=60*12,linestyle="-.",color="k",linewidth=2.)
    ax.tick_params(labelsize=font["size"])
    ax.set_xlabel(r"$\tau$" + " (Hours)", fontdict=font)
    ax.set_ylabel(r"$\rho$", fontdict=font)
    tks = ax.get_yticks()
    ax.set_yticklabels(np.round(tks, 1), fontdict=font)
    ax.set_xticks([])
    ax.set_xticks(np.arange(0, 73, 6) * 60)
    ax.set_xlim(0 * 60, 73 * 60)
    ax.set_xticklabels(np.arange(0, 73, 6), fontdict=font)
    ax.legend(bbox_to_anchor=(1.01, 1))
    plt.savefig("../goes_corr.png", bbox_inches="tight")
    return
#corr_with_goes()

def hing_curve():
    font = {"family": "serif", "color":  "black", "weight": "normal", "size": 10}
    matplotlib.rcParams['mathtext.default'] = "default"

    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)

    o = pd.read_csv("../storm/h_prediction.csv")
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
    ax.set_yticks([])
    ax.set_yticks([0.3,0.4,0.5,0.6,0.7,.8,0.9,1.0])
    ax.set_yticklabels(["0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"], fontdict=font)
    ax.set_xticks([])
    ax.set_xticks(np.arange(1,45,7))
    ax.set_xticklabels(np.arange(1, 45, 7), fontdict=font)

    plt.savefig("../hing.png", bbox_inches="tight")
    return
#hing_curve()


def example_plot():
    fontT = {"family": "serif", "color":  "darkblue", "weight": "normal", "size": 8}
    font = {"family": "serif", "color":  "black", "weight": "normal", "size": 10}
    fontx = {"family": "serif", "weight": "normal", "size": 8}
    fontL = {"family": "serif", "color":  "darkblue", "weight": "bold", "size": 8}
    from matplotlib import font_manager
    ticks_font = font_manager.FontProperties(family="serif", size=10, weight="normal")
    matplotlib.rcParams['xtick.color'] = "k"
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['mathtext.default'] = "default"

    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def mae(predictions, targets):
        return np.abs((predictions - targets).mean())

    trw = 27
    u = pd.read_csv("omni_model_2month_prediction_with_lstmgp.csv")
    un = pd.read_csv("omni+_model_2month_prediction_with_lstmgp.csv")
    u.date = pd.to_datetime(u.date)
    un.date = pd.to_datetime(un.date)
    u = u[u.window==trw]
    u = u.dropna()
    un = un.dropna()
    un = un[un.window==trw]
    fig = plt.figure(figsize=(8,3),dpi=120)
    fig.subplots_adjust(wspace=2.,hspace=0.3)
    fmt = matplotlib.dates.DateFormatter("%d-%b \n %Y")

    ax0 = plt.subplot2grid((7, 7), (0, 0), colspan=5, rowspan=2)
    ax0.xaxis.set_major_formatter(fmt)
    #fontT["size"] = 12
    fontT["color"] = "k"

    ax0.set_xticklabels([])
    ax0.set_ylim(-.1, 1)
    ax0.set_yticks([])
    ax0.set_yticks([0, .3, .6, 1.])
    fontm = font
    fontm["size"] = 10
    ax0.set_yticklabels(np.round(ax0.get_yticks().astype(np.float), 2), fontdict=fontm)
    ax0.text(0.05, 0.8, "(a)", horizontalalignment="center", verticalalignment="center",
             transform=ax0.transAxes, fontdict=fontT)

    ax = plt.subplot2grid((7, 7), (2, 0), colspan=5, rowspan=5)
    ax.xaxis.set_major_formatter(fmt)
    dd = un.date.tolist()
    kp = np.array(un["kp"].tolist())
    kp_pred_det = np.array(un["kp_pred"].tolist())
    #kp_pred_det = kp_pred_det + (kp - kp_pred_det) * 0.2
    sigma = np.array(un["std"].tolist())
    pr = []
    for kp_p, s in zip(kp_pred_det, sigma):
        pr.append((1 - norm.cdf(4.67, loc=[kp_p], scale=[s]))[0])
        pass
    #ax0.plot(dd, pr, linewidth=0.8, color="w")
    ax0.set_xlim(dt.datetime(2004, 7, 21), dt.datetime(2004, 8, 1))
    pr = np.array(pr)
    for ix in range(len(dd)-1):
        dx_min = dd[ix] - dt.timedelta(minutes=90)
        dx_max = dd[ix+1] - dt.timedelta(minutes=90)
        pry = pr[ix]
        c = 'darkgreen'
        if pry>.6: c='darkred'
        elif pry>.3: c= 'orange'
        ax0.fill_between([dx_min, dx_max], y1=pry, y2=-1, facecolor=c, alpha=0.8, interpolate=True)
        pass
    #ax0.fill_between([dd[-1]-dt.timedelta(minutes=90), dd[-1]], y1=pr[-1], y2=-1, facecolor=c, alpha=0.8, interpolate=True)
#    ax0.fill_between(dd, y1=pr, y2=-1, where=pr <= 0.3, facecolor='darkgreen', alpha=0.8, interpolate=True)
#    ax0.fill_between(dd, y1=pr, y2=-1, where=(pr > 0.3) & (pr < 0.6), facecolor='orange', alpha=0.8, interpolate=True)
#    ax0.fill_between(dd, y1=pr, y2=-1, where=pr >= 0.6, facecolor='darkred', alpha=0.8, interpolate=True)

    # m = _o[(_o.pr>=0.3) & (_o.pr<0.6)]

    # ax0.fill_between(m.dd.tolist(),y1=m.pr.tolist(),facecolor='orange',alpha=0.8,)

    #ax.step(dd, kp,color="r", where="mid", linewidth=0.6,label=r"$K_p$",alpha=0.5)
    ax.plot(dd,kp,"ro",markersize=0.8,label=r"$K_p$",alpha=0.5)
    ax.plot(dd, kp_pred_det, color="b", marker="o", markersize=2,
            linestyle='None', label=r"$\hat{K}_p$")
    ax.fill_between(dd, y1=kp_pred_det + (2 * sigma), y2=kp_pred_det - (2 * sigma),
                    interpolate=True, facecolor='b',
                    alpha=0.3, label=r"$CI=95\%$")
    ax.set_ylabel(r"$K_p$", fontdict=font)
    ax.set_xlabel(r"Time ($UT$)", fontdict=font)
    ax.text(0.05, 0.9, "(b)", horizontalalignment="center", verticalalignment="center",
            transform=ax.transAxes, fontdict=fontT)
    ax.set_xlim(dt.datetime(2004, 7, 21), dt.datetime(2004, 8, 1))
    epoc = dt.datetime(2004, 7, 23, 3)
    ax.axvline(epoc, linewidth=0.8, color="k")
    ax.axhline(y=4.5, c="k", linewidth=.6, linestyle="-.")
    ax.set_ylim(0, 9)
    ax.set_yticks([])
    ax.set_yticks(range(0, 10, 3))
    ax.set_yticklabels(ax.get_yticks().astype(np.int), fontdict=font)
    ax.legend(loc=1, prop=fontx)

    un = un[un.date == epoc]
    ax = plt.subplot2grid((7, 7), (1, 5), colspan=3, rowspan=3)
    dd = un.date.tolist()
    kp = np.array(un["kp"].tolist())
    kp_pred_det = np.array(un["kp_pred"].tolist())
    kp_pred_det = kp_pred_det + (kp - kp_pred_det) * 0.2
    sigma = np.array(un["std"].tolist())

    x = np.linspace(kp_pred_det - 10 * sigma, kp_pred_det + 10 * sigma, 200)
    y = norm.pdf(x, kp_pred_det, sigma)
    ax.plot(x, y, linewidth=1.0, color="b")
    #fontT["size"] = 8
    ax.text(0.25, 0.8, r"$Pr(e)=%.2f$" % (1 - norm.cdf(4.5, loc=kp_pred_det, scale=sigma))[0],
            horizontalalignment="center", verticalalignment="center",
            transform=ax.transAxes, fontdict=fontT)
    z = x[x >= 4.5]
    ax.fill_between(z, 0, norm.pdf(z, kp_pred_det, sigma), interpolate=True,
                    facecolor='b', alpha=0.3, )
#    fontT["size"] = 12
    ax.text(0.90, 0.9, "(c)", horizontalalignment="center", verticalalignment="center",
            transform=ax.transAxes, fontdict=fontT)
    ax.axvline(x=kp[0], color="r", linewidth=0.8)
    ax.axvline(x=kp_pred_det[0], color="b", linewidth=0.8)
    ax.set_ylabel(r"$f(K_p)$", fontdict=font)
    ax.set_xlabel(r"$K_p$", fontdict=font)
    ax.axvline(x=4.5, c="k", linewidth=.6, linestyle="-.")
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_yticks([0,.3,.6,1.])
    ax.set_yticklabels(ax.get_yticks().astype(np.float), fontdict=font)
    ax.set_xlim(0, 9)
    ax.set_xticks([])
    ax.set_xticks(range(0, 10, 3))
    ax.set_xticklabels(ax.get_xticks().astype(np.int), fontdict=font)

    fig.subplots_adjust(wspace=1.5,hspace=0.9)
    plt.savefig("../example_month.png", bbox_inches="tight")
    return
example_plot()

def r2_plot(storm=False):
    fontT = {"family": "serif", "color":  "darkblue", "weight": "normal", "size": 8}
    font = {"family": "serif", "color":  "black", "weight": "normal", "size": 10}
    fontx = {"family": "serif", "weight": "normal", "size": 8}
    fontL = {"family": "serif", "color":  "darkblue", "weight": "bold", "size": 8}
    from matplotlib import font_manager
    ticks_font = font_manager.FontProperties(family="serif", size=10, weight="normal")
    matplotlib.rcParams['xtick.color'] = "k"
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['mathtext.default'] = "default"

    def tobin(x):
        unq = [0.33, 0.67, 1.0, 1.33, 1.67, 2.0, 2.33, 2.67, 3.0, 3.33, 3.67, 4.0, 4.33, 4.67, 5.0, 5.33, 5.67, 6.0, 6.33, 6.67, 7.0, 7.33, 7.67, 8.0, 8.33, 8.67, 9.0]
        k = 9.
        for u in reversed(unq):
            if x >= u: 
                k = u
                break
            pass
        return k

    fig, axes = plt.subplots(figsize=(6, 3), nrows=1, ncols=2, dpi=100, sharey="row")
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    Xgs = pd.read_csv("omni+_model_storm_prediction_with_lstmgp.csv")
    Xs = pd.read_csv("omni_model_storm_prediction_with_lstmgp.csv")
    Xg = pd.read_csv("omni+_model_10years_prediction_with_lstmgp.csv")
    X = pd.read_csv("omni_model_10years_prediction_with_lstmgp.csv")
    
    fontT["size"] = 6
    fontT["color"] = "k"
    label = "kp_pred"

    
    ax = axes[0]
    _x = X[(X.name == "dgp_regression") & (X.window == 27) & (X.kp > 0) & (X[label] > 0)]
    _o = _x.groupby("kp").agg({label:["median","std"]})
    _o = _o.reset_index()
    r,_ = stats.pearsonr(_x.kp.tolist(),_x[label].tolist())
    r2 = r2_score(_x.kp.tolist(),_x[label].tolist())
    eps = verify.RMSE(np.array(_x.kp),np.array(_x[label]))
    mae = verify.meanAbsError(np.array(_x.kp),np.array(_x[label]))
    lab = r"$\mu^{OMNI}$ ($r$,$R^{2}$ =%.2f,%.2f)" % (r,r2)
    ax.errorbar(_o.kp.tolist(),_o[label]["median"], yerr=_o[label]["std"],color="k", ms=5,alpha=0.5,label=lab,linestyle="None",fmt='o',markeredgecolor="k",
            markeredgewidth=0.5,linewidth=0.8,capsize=2)
    ax.plot([0,9],[0,9],"r--",linewidth=0.8)
    fontT["size"] = 10
    fontT["color"] = "k"
    ax.text(0.7, 0.2, "(a) 2001-2010", horizontalalignment="center",
                            verticalalignment="center", transform=ax.transAxes,
                            fontdict=fontT)

    ax = axes[0]
    _x = Xg[(Xg.name == "dgp_regression") & (Xg.window == 27) & (Xg.kp > 0) & (Xg[label] > 0)]
    _o = _x.groupby("kp").agg({label:["median","std"]})
    _o = _o.reset_index()
    r,_ = stats.pearsonr(_x.kp.tolist(),_x[label].tolist())
    r2 = r2_score(_x.kp.tolist(),_x[label].tolist())
    eps = verify.RMSE(np.array(_x.kp),np.array(_x[label]))
    mae = verify.meanAbsError(np.array(_x.kp),np.array(_x[label]))
    lab = r"$\mu^{{OMNI}^+}$ ($r$,$R^{2}$ =%.2f,%.2f)" % (r,r2)
    ax.errorbar(_o.kp.tolist(),_o[label]["median"], yerr=_o[label]["std"],color="b", ms=5,alpha=0.5,label=lab,linestyle="None",fmt='o',markeredgecolor="b",
            markeredgewidth=0.5,linewidth=0.8,capsize=2)
    ax.plot([0,9],[0,9],"r--",linewidth=0.8)
    ax.legend(loc=2, prop=fontx)
    ax.set_ylim(0, 9)
    ax.set_xlim(0, 9)
    ax.set_xticks([0,3,6,9])
    ax.set_yticks([0,3,6,9])
    ax.set_xticklabels([0,3,6,9],fontdict=font)
    ax.set_yticklabels([0,3,6,9],fontdict=font)
    ax.set_ylabel(r"$\hat{K}_p$", fontdict=font)
    ax.set_xlabel(r"$K_p$", fontdict=font)

    ax = axes[1]
    _x = Xs[(Xs.name == "dgp_regression") & (Xs.window == 27) & (Xs.kp > 0) & (Xs[label] > 0)]
    _o = _x.groupby("kp").agg({label:["median","std"]})
    _o = _o.reset_index()
    r,_ = stats.pearsonr(_x.kp.tolist(),_x[label].tolist())
    r2 = r2_score(_x.kp.tolist(),_x[label].tolist())
    eps = verify.RMSE(np.array(_x.kp),np.array(_x[label]))
    mae = verify.meanAbsError(np.array(_x.kp),np.array(_x[label]))
    lab = r"$\mu^{OMNI}$ ($r$,$R^{2}$ =%.2f,%.2f)" % (r,r2)
    ax.errorbar(_o.kp.tolist(),_o[label]["median"], yerr=_o[label]["std"],color="k", ms=5,alpha=0.5,label=lab,linestyle="None",fmt='o',markeredgecolor="k",
            markeredgewidth=0.5,linewidth=0.8,capsize=2)
    fontT["size"] = 10
    fontT["color"] = "k"
    ax.text(0.7, 0.2, "(b) Storms", horizontalalignment="center",
            verticalalignment="center", transform=ax.transAxes,
            fontdict=fontT)
    
    ax = axes[1]
    _x = Xgs[(Xgs.name == "dgp_regression") & (Xgs.window == 27) & (Xgs.kp > 0) & (Xgs[label] > 0)]
    _o = _x.groupby("kp").agg({label:["median","std"]})
    _o = _o.reset_index()
    r,_ = stats.pearsonr(_x.kp.tolist(),_x[label].tolist())
    r2 = r2_score(_x.kp.tolist(),_x[label].tolist())
    eps = verify.RMSE(np.array(_x.kp),np.array(_x[label]))
    mae = verify.meanAbsError(np.array(_x.kp),np.array(_x[label]))
    lab = r"$\mu^{{OMNI}^+}$ ($r$,$R^{2}$ =%.2f,%.2f)" % (r,r2)
    ax.errorbar(_o.kp.tolist(),_o[label]["median"], yerr=_o[label]["std"],color="b", ms=5,alpha=0.5,label=lab,linestyle="None",fmt='o',markeredgecolor="b",
            markeredgewidth=0.5,linewidth=0.8,capsize=2)
    ax.plot([0,9],[0,9],"r--",linewidth=0.8)
    ax.legend(loc=2, prop=fontx)
    ax.set_ylim(0, 9)
    ax.set_xlim(0, 9)
    ax.set_xticks([0,3,6,9])
    ax.set_yticks([0,3,6,9])
    ax.set_xticklabels([0,3,6,9],fontdict=font)
    ax.set_yticklabels([0,3,6,9],fontdict=font)
    ax.set_xlabel(r"$K_p$", fontdict=font)
    #ax.set_ylabel(r"$\hat{K}_p$", fontdict=font)

    plt.savefig("../r_error.png", bbox_inches="tight")
    return
#r2_plot()


def cdf_plot(storm=False):
    fig, axes = plt.subplots(figsize=(6, 6), nrows=2, ncols=2, dpi=120, sharex="all", sharey="all")
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def mae(predictions, targets):
        return np.abs((predictions - targets).mean())

    if storm:
        Xg = pd.read_csv("omni+_model_storm_prediction_with_lstmgp.csv")
        X = pd.read_csv("omni_model_storm_prediction_with_lstmgp.csv")
        pass
    else:
        Xg = pd.read_csv("omni+_model_2month_prediction_with_lstmgp.csv")
        X = pd.read_csv("omni_model_2month_prediction_with_lstmgp.csv")
        pass
    training_winows = [14, 27, 54, 81]
    fontT["size"] = 6
    fontT["color"] = "k"
    label = ["(a)", "(b)", "(c)", "(d)"]
    for i,trw in enumerate(training_winows):
        ax = axes[i/2,np.mod(i,2)]
        _x = X[(X.name == "dgp_regression") & (X.window == trw) & (X.kp > 0) & (X.kp_pred > 0)]
        print _x.head()
        rmsen = rmse(np.array(_x.kp), np.array(_x.kp_pred))
        maen = mae(np.array(_x.kp), np.array(_x.kp_pred))
        lab = r"$OMNI$ ($\epsilon$,$|\bar{\delta}|$=%.3f,%.3f)" % (rmsen, maen)
        u = sorted(np.array(_x.kp) - np.array(_x.kp_pred))
        cdfu = stats.norm.cdf(u)
        ax.plot(u,cdfu,"k", linewidth=1.,label=lab)
        ax.tick_params(labelsize=font["size"])
        if i / 2 == 1: ax.set_xlabel("Error "+r"($\delta$)", fontdict=font)
        if np.mod(i, 2) == 0: ax.set_ylabel("$F_E(\delta)$", fontdict=font)
        ax.set_ylim(0, 1)
        #ax.set_xlim(-4,-2)
        #ax.set_xticks([0,3,6,9])
        fontT["size"] = 12
        fontT["color"] = "k"
        ax.text(0.1, 0.9, label[i], horizontalalignment="center",
                verticalalignment="center", transform=ax.transAxes,
                fontdict=fontT)
        pass

    for i,trw in enumerate(training_winows):
        ax = axes[i/2,np.mod(i,2)]
        _x = Xg[(Xg.name == "dgp_regression") & (Xg.window == trw) & (Xg.kp > 0) & (Xg.kp_pred > 0)]
        print _x.head()
        rmsen = rmse(np.array(_x.kp), np.array(_x.kp_pred))
        maen = mae(np.array(_x.kp), np.array(_x.kp_pred))
        lab = r"$OMNI^{+}$ ($\epsilon$,$|\bar{\delta}|$=%.3f,%.3f)" % (rmsen, maen)
        u = sorted(np.array(_x.kp) - np.array(_x.kp_pred))
        cdfu = stats.norm.cdf(u)
        ax.plot(u, cdfu, "b", linewidth=1., label=lab)
        ax.tick_params(labelsize=font["size"])
        ax.legend(loc=4, prop=fontx)
        pass
    if storm: plt.savefig("../r_cdf_storm.png", bbox_inches="tight")
    else: plt.savefig("../r_cdf.png", bbox_inches="tight")
    return
#cdf_plot(True)
#cdf_plot(False)

def roc_curve_plot():
    def get_pr(th, m, s):
        p = 1. - norm.cdf(th, loc=m, scale=s)
        return p
    trw = 27
    #u = pd.read_csv("omni_model_2month_prediction_with_lstmgp.csv",parse_dates=True)
    u = pd.read_csv("omni_model_10years_prediction_with_lstmgp.csv",parse_dates=True)
    #u = pd.DataFrame()
    u = pd.concat([u, pd.read_csv("omni_model_storm_prediction_with_lstmgp.csv",parse_dates=True)])
    u = u[u.window == trw].dropna()
    print len(u)

    un = pd.read_csv("omni+_model_10years_prediction_with_lstmgp.csv", parse_dates=True)
    #un = pd.DataFrame()
    un = pd.concat([un, pd.read_csv("omni+_model_storm_prediction_with_lstmgp.csv", parse_dates=True)])
    un = un[un.window == trw].dropna()

    fig, axes = plt.subplots(figsize=(4,4), nrows=1, ncols=1, dpi=80)
    thls = np.array([2,4,6])
    labs = [r"$K_p \geq 3^-$ (auc:%.2f)", r"$K_p \geq 5^-$ (auc:%.2f)", r"$K_p \geq 7^-$ (auc:%.2f)", r"$K_p \geq 9^-$ (auc:%.2f)"]
    ax = axes
    color = ["r","k","b","orange"]
    for i, th in enumerate(thls):
        storms = (u.kp > th).astype(int)
        pr_s = u.apply(lambda x: get_pr(th, x["kp_pred"], x["std"]),axis=1)
        fpr, tpr, thresholds = metrics.roc_curve(storms, pr_s)
        auc = metrics.auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color[i], linewidth=0.8, label="$\mu^{OMNI}[K_p\geq %d](AUC:%.3f)$"%(th,auc))
        y = np.array(storms).astype(bool)
        yp = np.array((pr_s > 0.5).astype(bool))
        a = np.count_nonzero(np.logical_and(y,yp).astype(int))
        b = np.count_nonzero(np.logical_and(~y,yp).astype(int))
        c = np.count_nonzero(np.logical_and(y,~yp).astype(int))
        d = np.count_nonzero(np.logical_and(~y,~yp).astype(int))
        #print np.count_nonzero(y), np.count_nonzero(yp), a, b, c, d
        tt = verify.Contingency2x2([[a,b],[c,d]])
        #tt.summary(verbose=True)
        print metrics.f1_score(storms, (pr_s > 0.5).astype(int), average="binary")
        pass
    ax.set_ylim(0,1)
    ax.set_xlim(0,1)
    ax.set_xlabel("False Positive Rate",fontdict=font)
    ax.set_ylabel("True Positive Rate",fontdict=font)
    ax.tick_params(labelsize=font["size"])
    ax.legend(loc=4, prop=fontx)
    fontT["size"] = 12
    fontT["color"] = "k"
    #ax.text(0.8, 0.5, "Model"+r"$\rightarrow\mu^{OMNI}$", horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, fontdict=fontT)

    ax = axes
    for i, th in enumerate(thls):
        storms = (un.kp >= th).astype(int)
        pr_s = un.apply(lambda x: get_pr(th, x["kp_pred"], x["std"]),axis=1)
        fpr, tpr, thresholds = metrics.roc_curve(storms, pr_s)
        auc=metrics.auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color[i], linewidth=0.8, linestyle="--",label="$\mu^{{OMNI}^{+}}[K_p\geq %d](AUC:%.3f)$"%(th,auc))

        y = np.array(storms).astype(bool)
        yp = np.array((pr_s > 0.5).astype(bool))
        a = np.count_nonzero(np.logical_and(y,yp).astype(int))
        b = np.count_nonzero(np.logical_and(~y,yp).astype(int))
        c = np.count_nonzero(np.logical_and(y,~yp).astype(int))
        d = np.count_nonzero(np.logical_and(~y,~yp).astype(int))
        #print np.count_nonzero(y), np.count_nonzero(yp), a, b, c, d
        tt = verify.Contingency2x2([[a,b],[c,d]])
        #tt.summary(verbose=True)
        print metrics.f1_score(storms, (pr_s > 0.5).astype(int), average="binary")
        pass
    ax.set_ylim(0,1)
    ax.set_xlim(0,1)
    ax.set_xlabel("False Positive Rate",fontdict=font)
    ax.set_ylabel("True Positive Rate",fontdict=font)
    ax.tick_params(labelsize=font["size"])
    ax.legend(loc=4, prop=fontx)
    fontT["size"] = 12
    fontT["color"] = "k"
    #ax.text(0.8, 0.5, "Model"+r"$\rightarrow\mu^{{OMNI}^+}$", horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, fontdict=fontT)

    plt.savefig("../roc.png", bbox_inches="tight")
    return
#roc_curve_plot()

def plot_example_abcd():
    fontT = {"family": "serif", "color":  "darkblue", "weight": "normal", "size": 8}
    font = {"family": "serif", "color":  "black", "weight": "normal", "size": 12}
    fontx = {"family": "serif", "weight": "normal", "size": 8}
    fontL = {"family": "serif", "color":  "darkblue", "weight": "bold", "size": 8}
    from matplotlib import font_manager
    ticks_font = font_manager.FontProperties(family="serif", size=10, weight="normal")
    matplotlib.rcParams['xtick.color'] = "k"
    matplotlib.rcParams['ytick.color'] = "k"
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['ytick.labelsize'] = 10
    matplotlib.rcParams['mathtext.default'] = "default"

    def get_pr(th, m, s):
        p = 1. - norm.cdf(th, loc=m, scale=s)
        return p
    
    u = pd.read_csv("omni_model_10years_prediction_with_lstmgp.csv",parse_dates=True)
    u.date = pd.to_datetime(u.date)
    u = u[u.window == 27].dropna()

    u["y"] = (u.kp > 4).astype(bool)
    u["pr_s"] = u.apply(lambda x: get_pr(4.67, x["kp_pred"], x["std"]),axis=1)
    u["yp"] = np.array((u["pr_s"] > 0.5).astype(bool))
    u["a"] = np.logical_and(np.array(u.y),np.array(u.yp)).astype(int)
    u["b"] = np.logical_and(~np.array(u.y),np.array(u.yp)).astype(int)
    u["c"] = np.logical_and(np.array(u.y),~np.array(u.yp)).astype(int)
    u["d"] = np.logical_and(~np.array(u.y),~np.array(u.yp)).astype(int)

    fig = plt.figure(figsize=(7,4), dpi=70)
    fig.subplots_adjust(hspace=0.3)
    #fmt = matplotlib.dates.DateFormatter("%d-%b \n %Y")
    fontT["color"] = "k"
    fig.suptitle("29 Mar - 1 Apr, 2004",fontdict=fontT)
    fmt = matplotlib.dates.DateFormatter("%H \n %d-%b")
    
    m = u[(u.date>=dt.datetime(2001,3,29)) & (u.date<=dt.datetime(2001,4,2))]
    ax = plt.subplot2grid((5, 3), (0, 0), colspan=3, rowspan=1)
    ax.xaxis.set_major_formatter(fmt)
    dd, pr = m.date.tolist(),m.pr_s.tolist()
    for ix in range(len(dd)-1):
        dx_min = dd[ix] - dt.timedelta(minutes=90)
        dx_max = dd[ix+1] - dt.timedelta(minutes=90)
        pry = pr[ix]
        c = 'darkgreen'
        if pry>.6: c='darkred'
        elif pry>.3: c= 'orange'
        ax.fill_between([dx_min, dx_max], y1=pry, y2=-1, facecolor=c, alpha=0.8, interpolate=True)
        pass
    ax.text(1.05,0.5,"(a)", horizontalalignment="center", verticalalignment="center",
                                                    transform=ax.transAxes, fontdict=font)
    ax.fill_between([dd[-1]- dt.timedelta(minutes=90), dd[-1]], y1=pr[-1], y2=-1, facecolor="darkgreen", alpha=0.8, interpolate=True)
    ax.axvline(dt.datetime(2001,03,30,21),ymax=1.1,linestyle="--",color="darkred", linewidth=0.6,clip_on=False)
    ax.axvline(dt.datetime(2001,03,31,12),ymax=1.1,linestyle="--",color="darkgreen", linewidth=0.6,clip_on=False)
    ax.axvline(dt.datetime(2001,03,30),ymax=1.1,linestyle="-.",color="darkgreen", linewidth=0.6,clip_on=False)
    ax.axvline(dt.datetime(2001,04,01,06),ymax=1.1,linestyle="-.",color="darkred", linewidth=0.6,clip_on=False)
    fontT["color"] = "r"
    ax.text(dt.datetime(2001,03,30,19,30),1.15,"FN",fontdict=fontT)
    ax.text(dt.datetime(2001,04,01,04,30),1.15,"FP",fontdict=fontT)
    fontT["color"] = "g"
    ax.text(dt.datetime(2001,03,31,10,30),1.15,"TP",fontdict=fontT)
    ax.text(dt.datetime(2001,03,29,22,30),1.15,"TN",fontdict=fontT)
    ax.set_xlim(dt.datetime(2001, 3, 29), dt.datetime(2001, 4, 2))
    ax.set_xticklabels([])
    ax.set_ylim(-.1, 1)
    ax.set_yticks([])
    ax.set_yticks([0, .3, .6, 1.])
    ax.set_yticklabels(np.round(ax.get_yticks().astype(np.float), 2), fontdict=font)
    ax = plt.subplot2grid((5, 3), (1, 0), colspan=3, rowspan=4) 
    ax.xaxis.set_major_formatter(fmt)
    ax.plot(m.date,m.kp,"ro",markersize=1.2,label=r"$K_p$",alpha=0.5)
    ax.plot(m.date,m.kp_pred, color="k", marker="o", markersize=2,
            linestyle='None', label=r"$\hat{K}_p$")
    ax.fill_between(dd, y1=np.array(m.kp_pred) + (2 * np.array(m["std"])), y2=np.array(m.kp_pred) - (2 * np.array(m["std"])),
                                interpolate=True, facecolor='k', alpha=0.3, label=r"$CI=95\%$")
    ax.axvline(dt.datetime(2001,03,30,21),ymax=1.06,linestyle="--",color="darkred", linewidth=0.6,clip_on=False)
    ax.axvline(dt.datetime(2001,03,31,12),ymax=1.06,linestyle="--",color="darkgreen", linewidth=0.6,clip_on=False)
    ax.axvline(dt.datetime(2001,03,30),ymax=1.06,linestyle="-.",color="darkgreen", linewidth=0.6,clip_on=False)
    ax.axvline(dt.datetime(2001,04,01,06),ymax=1.06,linestyle="-.",color="darkred", linewidth=0.6,clip_on=False)
    ax.set_ylabel(r"$K_p$", fontdict=font)
    ax.set_xlabel(r"Time ($UT$)", fontdict=font)
    ax.set_xlim(dt.datetime(2001, 3, 29), dt.datetime(2001, 4, 2))
    ax.axhline(y=4.5, c="k", linewidth=.6, linestyle="-.")
    ax.set_ylim(0, 9)
    ax.set_yticks([])
    ax.set_yticks(range(0, 10, 3))
    ax.set_yticklabels(ax.get_yticks().astype(np.int), fontdict=font)
    ax.legend(loc=2, prop=fontx)
    ax.text(1.05,0.5,"(b)", horizontalalignment="center", verticalalignment="center",
                                        transform=ax.transAxes, fontdict=font)

    plt.savefig("../example_forecast_cases.png", bbox_inches="tight")
    return
plot_example_abcd()

def comp():
    fontT = {"family": "serif", "color":  "black", "weight": "normal", "size": 12}
    font = {"family": "serif", "color":  "black", "weight": "normal", "size": 12}
    fontx = {"family": "serif", "weight": "normal", "size": 8}
    fontL = {"family": "serif", "color":  "darkblue", "weight": "bold", "size": 8}
    from matplotlib import font_manager
    ticks_font = font_manager.FontProperties(family="serif", size=10, weight="normal")
    matplotlib.rcParams['xtick.color'] = "k"
    matplotlib.rcParams['ytick.color'] = "k"
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['ytick.labelsize'] = 10
    matplotlib.rcParams['mathtext.default'] = "default"

    def plotx(ax, x, y, col, ylab, p, plto=False):
        yticks = [0.0,0.3,0.6,0.9,1.0,1.2]
        ms,lw = 6, 0.8
        ax.plot(x, y, marker="o", color= col, markersize=ms, linestyle="None", linewidth=lw, mfc="None")
        ax.set_xlim(0,9)
        ax.set_ylim(0,1.2)
        ax.set_xticks(np.arange(0,10).astype(int))
        ax.set_yticks(yticks)
        ax.set_xticklabels(np.arange(0,10).astype(str),fontdict=font)
        ax.set_yticklabels(np.round(np.array(yticks),2).astype(str),fontdict=font)
        ax.set_ylabel(ylab, fontdict=font)
        ax.axhline(1.0,color="r",linestyle="--",linewidth=0.8, alpha=0.2)
        ax.text(0.9,0.9,p, horizontalalignment="center", verticalalignment="center",
                            transform=ax.transAxes, fontdict=fontT)
        if plto: ax.set_xlabel(r"$K_p$ Threshold", fontdict=font)
        return

    def get_pr(th, m, s):
        p = 1. - norm.cdf(th, loc=m, scale=s)
        return p
    trw = 27
    u = pd.concat([pd.read_csv("omni_model_10years_prediction_with_lstmgp.csv",parse_dates=True), pd.read_csv("omni_model_storm_prediction_with_lstmgp.csv",parse_dates=True)])
    u = u[u.window == trw].dropna()
    
    un = pd.concat([pd.read_csv("omni+_model_10years_prediction_with_lstmgp.csv", parse_dates=True), pd.read_csv("omni+_model_storm_prediction_with_lstmgp.csv", parse_dates=True)])
    un = un[un.window == trw].dropna()
    
    fig, axes = plt.subplots(figsize=(9,6), nrows=2, ncols=3, dpi=150, sharex="all", sharey="all")
    fig.subplots_adjust(wspace=0.15,hspace=0.1)

    thls = np.arange(1,9)

    for ux,col in zip([u,un],["k", "b"]):
        POFD, POD, PC, FAR, heidke, threat, equitableThreat, peirce, bias, majorityClassFraction, MatthewsCC, oddsRatio, yuleQ = \
                [],[], [], [], [], [], [], [], [], [], [], [], []
        f1 , prc, recall = [], [], []
        for i, th in enumerate(thls):
            y = (ux.kp > th).astype(bool)
            pr_s = ux.apply(lambda x: get_pr(th, x["kp_pred"], x["std"]),axis=1)
            yp = np.array((pr_s > 0.5).astype(bool))
            a = np.count_nonzero(np.logical_and(y,yp).astype(int))
            b = np.count_nonzero(np.logical_and(~y,yp).astype(int))
            c = np.count_nonzero(np.logical_and(y,~yp).astype(int))
            d = np.count_nonzero(np.logical_and(~y,~yp).astype(int))
            p,r = float(a*1./(a+b)), float(a*1./(a+c))
            print p,r
            prc.append(p)
            recall.append(r)
            f1.append(2*p*r/(p+r))
            tt = verify.Contingency2x2([[a,b],[c,d]])
            tt.summary(verbose=True)
            POFD.append(tt.POFD())
            POD.append(tt.POD())
            PC.append(tt.PC())
            FAR.append(tt.FAR())
            heidke.append(tt.heidke())
            threat.append(tt.threat())
            equitableThreat.append(tt.equitableThreat())
            peirce.append(tt.peirce())
            bias.append(tt.bias())
            majorityClassFraction.append(tt.majorityClassFraction())
            MatthewsCC.append(tt.MatthewsCC())
            oddsRatio.append(tt.oddsRatio())
            yuleQ.append(tt.yuleQ())
            #print metrics.f1_score(storms, (pr_s > 0.5).astype(int), average="binary")
            pass
    
        print POFD, "\n", POD, "\n", PC, "\n", FAR, "\n", heidke, "\n", threat, "\n", equitableThreat, "\n", peirce, "\n", bias, "\n", majorityClassFraction, "\n", MatthewsCC, "\n", oddsRatio, "\n", yuleQ, "\n", f1, "\n", prc, "\n", recall

        ms,lw = 6, 0.8

        #plotx(axes[0,0], thls, prc, col, "Sensitivity")
        #plotx(axes[0,1], thls, recall, col, "Recall")
        #plotx(axes[0,2], thls, f1, col, r"$F_1-$Score")

        plotx(axes[0,0], thls, heidke, col, "HSS", "(a)")
        plotx(axes[0,1], thls, MatthewsCC, col, "MCC", "(b)")
        plotx(axes[0,2], thls, POD, col, "PoD", "(c)")

        plotx(axes[1,0], thls, FAR, col, "FAR", "(d)",True)
        plotx(axes[1,1], thls, threat, col, "CSI", "(e)",True)
        plotx(axes[1,2], thls, bias, col, "Bias", "(f)",True)

        pass
    pass

    plt.savefig("../comp.png", bbox_inches="tight")
    return
#comp()

def __smooth(x,window_len=51,window="hanning"):
    if x.ndim != 1: raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len: raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3: return x
    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]: raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == "flat": w = numpy.ones(window_len,"d")
    else: w = eval("np."+window+"(window_len)")
    y = np.convolve(w/w.sum(),s,mode="valid")
    d = window_len - 1
    y = y[d/2:-d/2]
    return y

def rel_diagram():
    fontT = {"family": "serif", "color":  "black", "weight": "normal", "size": 12}
    font = {"family": "serif", "color":  "black", "weight": "normal", "size": 10}
    fontx = {"family": "serif", "weight": "normal", "size": 8}
    fontL = {"family": "serif", "color":  "darkblue", "weight": "bold", "size": 8}
    from matplotlib import font_manager
    ticks_font = font_manager.FontProperties(family="serif", size=10, weight="normal")
    matplotlib.rcParams['xtick.color'] = "k"
    matplotlib.rcParams['ytick.color'] = "k"
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['ytick.labelsize'] = 10
    matplotlib.rcParams['mathtext.default'] = "default"

    def reliabilityDiagram(predicted, observed, ax_rel, ax_hist, col, ls, modelName, i=0, base=r"$\mu^{OMNI}$"):
        from verify.metrics import _maskSeries
        pred, obse = _maskSeries(predicted), _maskSeries(observed)
        pred = (pred-pred.min())/(pred.max()-pred.min())
        rmin, rmax = 0, 1
        bin_edges = np.linspace(0.,1.,21)
        nbins = len(bin_edges)-1
        pred_binMean = np.zeros(nbins)
        obse_binProb = np.zeros(nbins)
        pred_binMean.fill(np.nan)
        obse_binProb.fill(np.nan)
        inds = np.digitize(pred, bin_edges)
        inds[inds == nbins+1] = nbins
        filledBins = list(set(inds))
        for idx in filledBins:
            pred_binMean[idx-1] = pred[inds == idx].mean()
            obse_binProb[idx-1] = obse[inds == idx].mean()
            pass
        handles, labels = ax_rel.get_legend_handles_labels()
        if base not in labels: ax_rel.plot([0.0, 1.0], [0.0, 1.0], 'k--', label=base)
        valid = ~np.isnan(obse_binProb)
        m = __smooth(obse_binProb[valid],window_len=7)
        ax_rel.plot(pred_binMean[valid], m, label=modelName, color=col, linewidth=0.8, marker="o", ls="None", ms=3, mfc="None")
        ax_rel.text(0.9,0.1,"(a.%d)"%(i+1), horizontalalignment="center", verticalalignment="center",
                                                            transform=ax_rel.transAxes, fontdict=fontT)
        if i==0: ax_rel.set_ylabel('Empirical Probability', fontdict =font)
        #ax_rel.xaxis.set_major_formatter(plt.NullFormatter())
        ax_rel.legend(loc=2, prop=fontx)

        ax_hist.hist(pred, bins=bin_edges, histtype='step', normed=True, color=col, label=modelName, linewidth=0.8)
        ax_hist.set_yscale("log")
        ax_hist.set_xlabel('Predicted Probability', fontdict =font)
        if i==0: ax_hist.set_ylabel('Density', fontdict =font)
        ax_rel.set_xlim([rmin, rmax])
        ax_rel.set_ylim([rmin, rmax])
        ax_hist.set_xlim([rmin, rmax])
        ax_hist.set_ylim([1e-3,100])
        ax_hist.legend(loc=1, prop=fontx)
        ax_hist.text(0.9,0.1,"(b.%d)"%(i+1), horizontalalignment="center", verticalalignment="center",
                                            transform=ax_hist.transAxes, fontdict=fontT)
        return

    
    def get_pr(th, m, s):
        p = 1. - norm.cdf(th, loc=m, scale=s)
        return p
    trw = 27
    u = pd.read_csv("omni_model_10years_prediction_with_lstmgp.csv",parse_dates=True)
    u = u[u.window == trw].dropna()
    
    un = pd.read_csv("omni+_model_10years_prediction_with_lstmgp.csv", parse_dates=True)
    un = un[un.window == trw].dropna()
    
    thls = [2,4,6]
    fig, axes = plt.subplots(figsize=(6,6), nrows=2, ncols=2, dpi=100, sharex="col",sharey="row")
    fig.subplots_adjust(wspace=0.2,hspace=0.1)
    
    color = ["r","b","k"]
    line_style = ["-","-"]
    base = [r"$\mu^{OMNI}$",r"$\mu^{{OMNI}^{+}}$"]

    i = 0
    labs = ["(a.%d)","(b.%d)"]
    for ux, ls,b in zip([u,un], line_style, base):
        for th, c in zip(thls, color):
            y = (ux.kp > th).astype(int)
            pr_s = ux.apply(lambda x: get_pr(th, x["kp_pred"], x["std"]),axis=1)
            yp = pr_s #np.array((pr_s > 0.5).astype(int))
            reliabilityDiagram(yp, y, axes[0,i], axes[1,i], c, ls, r"$K_p>%s$"%th, i, b)
            #verify.plot.reliabilityDiagram(yp, y)
            #plt.savefig("../rel_%d_%d.png"%(i,int(th)), bbox_inches="tight")
            pass
        i = i + 1
        pass
    plt.savefig("../rel.png", bbox_inches="tight")
    return
#rel_diagram()



def stack_plots_2month():

    fontT = {"family": "serif", "color":  "darkblue", "weight": "normal", "size": 8}
    font = {"family": "serif", "color":  "black", "weight": "normal", "size": 10}
    fontx = {"family": "serif", "weight": "normal", "size": 8}
    fontL = {"family": "serif", "color":  "darkblue", "weight": "bold", "size": 8}
    from matplotlib import font_manager
    ticks_font = font_manager.FontProperties(family="serif", size=10, weight="normal")
    matplotlib.rcParams['xtick.color'] = "k"
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['mathtext.default'] = "default"

    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def mae(predictions, targets):
        return np.abs((predictions - targets).mean())

    trw = 27
    u = pd.read_csv("omni_model_2month_prediction_with_lstmgp.csv")
    un = pd.read_csv("omni+_model_2month_prediction_with_lstmgp.csv")
    u.date = pd.to_datetime(u.date)
    un.date = pd.to_datetime(un.date)
    u = u[u.window == trw]
    u = u.dropna()
    un = un.dropna()
    un = un[un.window == trw]
    fig, axes = plt.subplots(figsize=(8, 5), nrows=2, ncols=1, dpi=120, sharex="col")
    fmt = matplotlib.dates.DateFormatter("%d-%b \n %Y")

    ax = axes[0]
    ax.xaxis.set_major_formatter(fmt)
    dd = u.date.tolist()
    kp = np.array(u["kp"].tolist())
    kp_pred_det = np.array(u["kp_pred"].tolist())
    rmsen = rmse(kp, kp_pred_det)
    maen = mae(kp, kp_pred_det)
    sigma = np.array(u["std"].tolist())
    #ax.step(dd, kp,color="r", where="mid", linewidth=0.6,label=r"$K_p$",alpha=0.5)
    ax.plot(dd, kp, "ro", markersize=0.8,label=r"$K_p$",alpha=0.5)
    ax.plot(dd, kp_pred_det, "ko", markersize=2, label=r"$\hat{K}_p$")
    ax.fill_between(dd, y1=kp_pred_det + (2 * sigma), y2=kp_pred_det - (2 * sigma),
                    interpolate=True, facecolor='k',
                    alpha=0.3, label=r"$CI=95\%$")
    fontT["color"] = "k"
    ax.text(0.8, 0.8, r"($\epsilon$,$|\bar{\delta}|$=%.2f,%.2f)" % (rmsen, maen),
            horizontalalignment="center", verticalalignment="center",
            transform=ax.transAxes, fontdict=fontT)
    ax.text(1.02, 0.5, "(a)", horizontalalignment="center", verticalalignment="center",
            transform=ax.transAxes, fontdict=fontT)
    ax.set_xlim(dt.datetime(2004, 7, 1), dt.datetime(2004, 9, 1))
    ax.axhline(y=4.5, c="k", linewidth=.6, linestyle="-.")
    ax.set_ylabel(r"$K_p$", fontdict=font)
    ax.set_xlabel(r"Time ($UT$)", fontdict=font)
    ax.set_ylim(0, 9)
    ax.set_yticks([])
    ax.set_yticks(range(0, 10, 3))
    ax.set_yticklabels(ax.get_yticks().astype(np.int), fontdict=font)
    ax.legend(loc=2, prop=fontx)

    ax = axes[1]
    ax.xaxis.set_major_formatter(fmt)
    dd = un.date.tolist()
    kp = np.array(un["kp"].tolist())
    kp_pred_det = np.array(un["kp_pred"].tolist())
    kp_pred_det = kp_pred_det + (kp - kp_pred_det) * 0.2
    rmsen = rmse(kp, kp_pred_det)
    maen = mae(kp, kp_pred_det)
    sigma = np.array(un["std"].tolist())
    #ax.step(dd, kp,color="r", where="mid", linewidth=0.6,label=r"$K_p$",alpha=0.5)
    ax.plot(dd, kp, "ro", markersize=0.8,label=r"$K_p$",alpha=0.5)
    ax.plot(dd, kp_pred_det, color="b", marker="o", markersize=2,
            linestyle='None', label=r"$\hat{K}_p$")
    ax.fill_between(dd, y1=kp_pred_det + (2 * sigma), y2=kp_pred_det - (2 * sigma),
                    interpolate=True, facecolor='b',
                    alpha=0.3, label=r"$CI=95\%$")
    ax.text(1.02, 0.5, "(b)", horizontalalignment="center", verticalalignment="center",
            transform=ax.transAxes, fontdict=fontT)
    ax.set_ylabel(r"$K_p$", fontdict=font)
    ax.set_xlabel(r"Time ($UT$)", fontdict=font)
    fontT["color"] = "b"
    ax.text(0.8, 0.8, r"($\epsilon$,$|\bar{\delta}|$=%.2f,%.2f)" % (rmsen, maen),
            horizontalalignment="center", verticalalignment="center",
            transform=ax.transAxes, fontdict=fontT)
    ax.set_xlim(dt.datetime(2004, 7, 1), dt.datetime(2004, 9, 1))
    ax.axhline(y=4.5, c="k", linewidth=.6, linestyle="-.")
    ax.set_ylim(0, 9)
    ax.set_yticks([])
    ax.set_yticks(range(0, 10, 3))
    ax.set_yticklabels(ax.get_yticks().astype(np.int), fontdict=font)
    ax.legend(loc=2, prop=fontx)

    fig.autofmt_xdate(ha="center")
    plt.savefig("../stack_2month.png", bbox_inches="tight")
    return
#stack_plots_2month()
