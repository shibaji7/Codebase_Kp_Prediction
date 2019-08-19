import os
import numpy as np
import datetime as dt
import pandas as pd
np.random.seed(0)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, BayesianRidge
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic as RQ

from keras.layers.core import Dense, Activation, Dropout
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.utils import to_categorical

from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping
from kgp.utils.assemble import load_NN_configs, load_GP_configs, assemble
from kgp.utils.experiment import train
from kgp.losses import gen_gp_loss

import database as db
os.environ["GPML_PATH"] = "/home/shibaji/anaconda3/envs/deep/lib/python2.7/site-packages/kgp-0.3.2-py2.7.egg/kgp/backend/gpml/"

##########################################################################################
## Classifier Models
##########################################################################################
class Classifier(object):
    '''
    This is a classifier model, random forest classifier is trained here.
    '''

    def __init__(self, threshold=4.5,delay_hours=3, is_goes=False):
        rus = RandomUnderSampler(return_indices=True)
        if is_goes: _o, xparams, yparam = db.load_data_RB(case=0,threshold=threshold,delay_hours=delay_hours)
        else: _o, xparams, yparam = db.load_data(case=0,threshold=threshold,delay_hours=delay_hours)
        X_resampled, y_resampled, idx_resampled = rus.fit_sample(_o[xparams].values, _o[yparam].values.ravel())
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=1.0/3.0, random_state=42)
        self.clf = RandomForestClassifier(n_estimators=100)
        self.clf.fit(X_train,y_train)
        self.test_unit(X_test,y_test)
        return

    def test_unit(self,X_test,y_test):
        y_pred = self.clf.predict_proba(X_test)[:, 1]
        fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred)
        auc_rf = auc(fpr_rf, tpr_rf)
        print "=========================================> AUC:",np.round(auc_rf,3)
        return

    def predict_proba(self,X):
        y_pred = self.clf.predict_proba(X)[:, 1]
        return y_pred

class LSTMDataSource(object):
    '''
    This class is used to create common LSTM data source
    '''

    def __init__(self, _o, xparams, yparam_reg, yparam_clf, look_back=3):
        self._o = _o
        self.xparams = xparams
        self.yparam_reg = yparam_reg
        self.yparam_clf = yparam_clf
        self.look_back = look_back

        self.ini()

        self.X_clf, self.y_clf = self.form_look_back_array(self.X_tx,self._o[self.yparam_clf].values)
        self.X_reg, self.y_reg = self.form_look_back_array(self.X_tx,self.y_reg_tx)

        self.Date_WS = self._o["Date_WS"].tolist()
        return

    def ini(self):
        self.sclX = MinMaxScaler(feature_range=(0, 1))
        self.sclY = MinMaxScaler(feature_range=(0, 1))

        self.X_tx = self.sclX.fit_transform(self._o[self.xparams].values)
        self.y_reg_tx = self.sclY.fit_transform(self._o[self.yparam_reg].values)
        return

    def form_look_back_array(self,X,y):
        dataX, dataY = [], []
        for i in range(self.look_back+1,len(X)):
            a = X[i-self.look_back:i, :].T
            dataX.append(a)
            dataY.append(y[i].tolist())
            pass
        return np.array(dataX), np.array(dataY)

    def create_master_model_data(self):
        X, y = self._o[self.xparams].values, self._o[self.yparam_clf].values.ravel()
        X = self.sclX.fit_transform(X)
        self.rus = RandomUnderSampler(return_indices=True)
        X_resampled, y_resampled, idx_resampled = self.rus.fit_sample(X,y)
        y_bin = to_categorical(y_resampled)
        X_resampled, y_resampled = self.form_look_back_array(X_resampled, y_bin)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=1.0/3.0, random_state=42)
        return X_train, X_test, y_train, y_test

    def get_data_for_regression_window(self, dn, window):
        _tstart = dn - dt.timedelta(days=window)
        _tend = dn - dt.timedelta(hours=3)
        is_run = False
        X_train, y_train, X_test, y_test, y_obs = None, None, None, None, None

        if _tstart in self.Date_WS and _tend in self.Date_WS and dn in self.Date_WS:
            _istart = self.Date_WS.index(_tstart)
            _iend = self.Date_WS.index(_tend)
            _i = self.Date_WS.index(dn)
            _xy_train = (self.X_reg[_istart:_iend,:,:], self.y_reg[_istart:_iend,:])
            _xy_test = (np.reshape(self.X_reg[_i,:,:], (1, len(self.xparams), self.look_back)), self.y_reg[_i,:])
            if _i > 0:
                X_train = _xy_train[0]
                y_train = _xy_train[1]
                X_test = _xy_test[0]
                y_test = _xy_test[1]
                y_obs = self._o[(self._o["Date_WS"] == dn)]["K_P_LT_delay"].tolist()[0]
                is_run = True
                pass
            pass
        return is_run, X_train, y_train, X_test, y_test, y_obs

    def tx_y(self,y):
        txy = self.sclY.inverse_transform(y)
        return txy


class LSTMClassifier(object):
    '''
    This is a classifier model, LSTM classifier is trained here.
    '''

    def __init__(self, threshold=4.5,delay_hours=3, source=None):
        self.threshold = threshold
        self.delay_hours = delay_hours
        self.source = source
        self.ini()
        self.setup()
        return

    def test_unit(self,X_test,y_test):
        y_pred_keras = self.model.predict_proba(X_test)[:,1]
        y_test_x = np.array([np.argmax(ym, axis=None, out=None) for ym in y_test])
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test_x, y_pred_keras)
        auc_keras = auc(fpr_keras, tpr_keras)
        print "=========================================> AUC:",np.round(auc_keras,3)
        return

    def ini(self):
        self.look_back = 3
        self.hidden_node = 100
        self.batch_size = 100
        self.epochs = 30
        return

    def create_model(self):
        self.model = Sequential()
        self.model.add(LSTM(self.hidden_node, return_sequences=False, input_shape=self.input_shape))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=2))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        return

    def setup(self):
        if self.source is None:
            _o, xparams, yparam = db.load_data(case=0,threshold=self.threshold,delay_hours=self.delay_hours)
            self.source = LSTMDataSource(_o,xparams, yparam_reg=["K_P_LT_delay"], yparam_clf=yparam)
            pass
        X_train, X_test, y_train, y_test = self.source.create_master_model_data()
        self.input_shape = (len(self.source.xparams), self.look_back)
        self.create_model()
        self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, validation_data=(X_test,y_test), verbose=0)
        self.test_unit(X_test,y_test)
        return

    def predict_proba(self,X):
        '''
        X should be reformed and transformed
        '''
        y_pred = self.model.predict_proba(X)[:,1]
        return y_pred




##############################################################################################################
## Regressor Models
##############################################################################################################
class DetRegressor(object):
    '''
    This is a regressor model, all types of deterministic regressors trained here.
    '''

    def opt_regressor(self):
        REGs = {}
        # basic regressor            
        REGs["dummy"] = DummyRegressor(strategy="median")
        REGs["regression"] = LinearRegression()
        REGs["elasticnet"] = ElasticNet(alpha=.5,tol=1e-2)
        REGs["bayesianridge"] = BayesianRidge(n_iter=300, tol=1e-5, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, fit_intercept=True)
        
        # decission trees
        REGs["dtree"] = DecisionTreeRegressor(random_state=0,max_depth=5)
        REGs["etree"] = ExtraTreeRegressor(random_state=0,max_depth=5)
        
        # NN regressor
        REGs["knn"] = KNeighborsRegressor(n_neighbors=25,weights="distance")
        
        # ensamble models
        REGs["ada"] = AdaBoostRegressor()
        REGs["bagging"] = BaggingRegressor(n_estimators=50, max_features=3)
        REGs["etrees"] = ExtraTreesRegressor(n_estimators=50)
        REGs["gboost"] = GradientBoostingRegressor(max_depth=5,random_state=0)
        REGs["randomforest"] = RandomForestRegressor(n_estimators=100)
        return REGs[self.name]

    def __init__(self, name, dn, dataset, clf, window=27, alt_window=365, is_mix_model = False, source = None):
        self.name = name
        self.window = window
        self.alt_window = alt_window
        self.dn = dn
        self.source = source
        self.is_mix_model = is_mix_model
        self._o,self.xparams,self.yparam = dataset[0],dataset[1],dataset[2]
        
        self.reg = self.opt_regressor()
        self.clf = clf
        self.data_windowing(self.window)
        self.fname = "../out/storm/prediction.csv"
        return

    def data_windowing(self,trw):
        _tstart = self.dn - dt.timedelta(days=trw)
        _tend = self.dn - dt.timedelta(hours=3)
        _o_train = self._o[(self._o["Date_WS"] >= _tstart) & (self._o["Date_WS"] <= _tend)]
        _o_test = self._o[(self._o["Date_WS"] == self.dn)]
        self.is_run = False
        if len(_o_test) == 1:
            self.X_train = _o_train[self.xparams].values
            self.y_train = _o_train[self.yparam].values.ravel()
            self.X_test = _o_test[self.xparams].values
            self.y_test = _o_test[self.yparam].values.ravel()
            self.y_obs = self.y_test[0]
            self.is_run = True
        return

    def save_results(self):
        if os.path.exists(self.fname): self._sum.to_csv(self.fname,mode="a",index=False,header=False)
        else: self._sum.to_csv(self.fname,mode="w",index=False,header=True)
        return

    def pred_lstm_proba(self, trw):
        pr = None
        self.is_run, _,_, X_test, _,_ = self.source.get_data_for_regression_window(self.dn, trw)
        if self.is_run : pr = self.clf.predict_proba(X_test)
        return pr

    def run_model(self):
        pr_th = 0.5
        self._sum = pd.DataFrame()
        self._sum["date"] = [self.dn]
        self._sum["name"] = [self.name]
        if self.is_mix_model: self._sum["name"] = ["mix_"+self.name]
        self._sum["kp"] = [-1.]
        self._sum["kp_pred"] = [-1.]
        self._sum["pr"] = [-1.]
        self._sum["pr_th"] = [pr_th]
        self._sum["window"] = [self.window]
        self._sum["alt_window"] = [self.alt_window]
        if self.is_run:
            self._sum["kp"] = np.round(self.y_obs,2)
            X = self.X_test
            if self.is_mix_model: 
                pr = self.pred_lstm_proba(self.window)
                self._sum["name"] = ["mix_"+self.name]
            else: pr = self.clf.predict_proba(self.X_test)
            if self.is_run:
                self._sum["pr"] = np.round(pr,2)
                if pr[0] > pr_th: self.data_windowing(self.alt_window)
                self.reg.fit(self.X_train, self.y_train)
                y_pred = self.reg.predict(self.X_test)
                if self.is_mix_model and y_pred >= 9.: y_pred = 9.
                self._sum["kp_pred"] = np.round(y_pred,3)
                pass
            pass
        self.save_results()
        return


class LSTMRegressor(object):
    '''
    This is a regressor model, LSTM regressors trained here.
    '''

    def opt_regressor(self):
        self.model = Sequential()
        self.model.add(LSTM(self.hidden_node, input_shape=self.input_shape))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(10,activation='linear'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(1,activation='linear'))
        self.model.compile(loss='mse', optimizer='adam',metrics=['mse','mae'])
        return

    def __init__(self, dn, source, clf, window=27, alt_window=365*5):
        self.name = "LSTM"
        self.window = window
        self.alt_window = alt_window
        self.dn = dn
        self.clf = clf
        self.source = source
        
        self.ini()
        self.opt_regressor()
        self.data_windowing(self.window)
        self.fname = "../out/storm/prediction.csv"
        return

    def ini(self):
        self.hidden_node = 150
        self.batch_size = 1000
        self.epochs = 10
        self.input_shape = (len(self.source.xparams), self.source.look_back)
        return

    def data_windowing(self,trw):
        self.is_run, self.X_train, self.y_train, self.X_test, self.y_test, self.y_obs = self.source.get_data_for_regression_window(self.dn, trw)
        return

    def save_results(self):
        if os.path.exists(self.fname): self._sum.to_csv(self.fname,mode="a",index=False,header=False)
        else: self._sum.to_csv(self.fname,mode="w",index=False,header=True)
        return

    def run_model(self):
        pr_th = 0.5
        self._sum = pd.DataFrame()
        self._sum["date"] = [self.dn]
        self._sum["name"] = [self.name]
        self._sum["kp"] = [-1.]
        self._sum["kp_pred"] = [-1.]
        self._sum["pr"] = [-1.]
        self._sum["pr_th"] = [pr_th]
        self._sum["window"] = [self.window]
        self._sum["alt_window"] = [self.alt_window]
        if self.is_run:
            self._sum["kp"] = np.round(self.y_obs,2)
            pr = self.clf.predict_proba(self.X_test)
            self._sum["pr"] = np.round(pr,2)
            if pr[0] > pr_th: self.data_windowing(self.alt_window)
            self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs, validation_split=0.2,verbose=0)
            y_pred = self.source.tx_y(self.model.predict(self.X_test))
            self._sum["kp_pred"] = np.round(y_pred,3)
            pass
        self.save_results()
        return

class GPRegressor(object):
    '''
    This is a regressor model, GP regressors trained here.
    '''

    def opt_regressor(self):
        self.model = GaussianProcessRegressor(kernel = self.kernel, n_restarts_optimizer = self.nrst)
        return

    def __init__(self, dn, dataset, clf, k_type, window=27, alt_window=365*5, source = None, th = 4.5):
        self.name = "gpr_"+k_type.lower()
        self.window = window
        self.alt_window = alt_window
        self.dn = dn
        self.clf = clf
        self.source = source
        self.k_type = k_type
        self.th = th
        self._o,self.xparams,self.yparam = dataset[0],dataset[1],dataset[2]
    
        self.ini()
        self.opt_regressor()
        self.data_windowing(self.window)
        self.fname = "../out/storm/prediction_gpr.csv"
        return

    def ini(self):
        self.nrst = 10
        if self.k_type == "RBF": self.kernel = RBF(length_scale=1.0,length_scale_bounds=(1e-02, 1e2))
        if self.k_type == "RQ": self.kernel = RQ(length_scale=1.0,alpha=0.1,length_scale_bounds=(1e-02, 1e2),alpha_bounds=(1e-2, 1e2))
        if self.k_type == "Matern": self.kernel = Matern(length_scale=1.0,length_scale_bounds=(1e-02, 1e2), nu=1.4)
        return

    def data_windowing(self,trw,use_th=False):
        _tstart = self.dn - dt.timedelta(days=trw)
        _tend = self.dn - dt.timedelta(hours=3)
        if use_th:
            _o_train = self._o[(self._o["Date_WS"] >= _tstart) & (self._o["Date_WS"] <= _tend) & (self._o["K_P_LT_delay"] >= self.th)]
        else: _o_train = self._o[(self._o["Date_WS"] >= _tstart) & (self._o["Date_WS"] <= _tend)]
        _o_test = self._o[(self._o["Date_WS"] == self.dn)]
        self.is_run = False
        if len(_o_test) == 1:
            self.X_train = _o_train[self.xparams].values
            self.y_train = _o_train[self.yparam].values.ravel()
            self.X_test = _o_test[self.xparams].values
            self.y_test = _o_test[self.yparam].values.ravel()
            self.y_obs = self.y_test[0]
            self.is_run = True
        return

    def save_results(self):
        if os.path.exists(self.fname): self._sum.to_csv(self.fname,mode="a",index=False,header=False)
        else: self._sum.to_csv(self.fname,mode="w",index=False,header=True)
        return

    def pred_lstm_proba(self, trw):
        pr = None
        _, _,_, X_test, _,_ = self.source.get_data_for_regression_window(self.dn, trw)
        if X_test is not None: pr = self.clf.predict_proba(X_test)
        return pr

    def run_model(self):
        pr_th = 0.5
        self._sum = pd.DataFrame()
        self._sum["date"] = [self.dn]
        self._sum["name"] = [self.name]
        self._sum["kp"] = [-1.]
        self._sum["kp_pred"] = [-1.]
        self._sum["pr"] = [-1.]
        self._sum["pr_th"] = [pr_th]
        self._sum["std"] = [-1.]
        self._sum["window"] = [self.window]
        self._sum["alt_window"] = [self.alt_window]
        if self.is_run:
            self._sum["kp"] = np.round(self.y_obs,2)
            pr = self.pred_lstm_proba(self.window)
            if pr is not None:
                self._sum["pr"] = np.round(pr,2)
                if pr[0] > pr_th: self.data_windowing(self.alt_window,True)
                self.model.fit(self.X_train,self.y_train)
                y_pred,y_std = self.model.predict(self.X_test,return_std=True)
                self._sum["kp_pred"] = np.round(y_pred,3)
                self._sum["std"] = np.round(y_std,3)
                pass
            pass
        self.save_results()
        return


class DeepGPR(object):
    '''
    This is a regressor model, LSTM GP regressors trained here.
    '''
    def opt_regressor(self):
        REGs = {}
        # basic regressor            
        REGs["dummy"] = DummyRegressor(strategy="median")
        REGs["regression"] = LinearRegression()
        REGs["elasticnet"] = ElasticNet(alpha=.5,tol=1e-2)
        REGs["bayesianridge"] = BayesianRidge(n_iter=300, tol=1e-5, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, fit_intercept=True)
        
        # decission trees
        REGs["dtree"] = DecisionTreeRegressor(random_state=0,max_depth=5)
        REGs["etree"] = ExtraTreeRegressor(random_state=0,max_depth=5)
        
        # NN regressor
        REGs["knn"] = KNeighborsRegressor(n_neighbors=25,weights="distance")
        
        # ensamble models
        REGs["ada"] = AdaBoostRegressor()
        REGs["bagging"] = BaggingRegressor(n_estimators=50, max_features=3)
        REGs["etrees"] = ExtraTreesRegressor(n_estimators=50)
        REGs["gboost"] = GradientBoostingRegressor(max_depth=5,random_state=0)
        REGs["randomforest"] = RandomForestRegressor(n_estimators=100)
        return REGs[self.name]

    def __init__(self, name, dn, dataset, clf, window=27, alt_window=365, source = None, fname = "../out/storm/prediction_lstm_mixgp.csv"):
        self.name = name
        self.window = window
        self.alt_window = alt_window
        self.dn = dn
        self.source = source
        self._o,self.xparams,self.yparam = dataset[0],dataset[1],dataset[2]
        
        self.reg = self.opt_regressor()
        self.clf = clf
        self.data_windowing(self.window)
        self.fname = fname
        return

    def data_windowing(self,trw):
        _tstart = self.dn - dt.timedelta(days=trw)
        _tend = self.dn - dt.timedelta(hours=3)
        _o_train = self._o[(self._o["Date_WS"] >= _tstart) & (self._o["Date_WS"] <= _tend)]
        _o_test = self._o[(self._o["Date_WS"] == self.dn)]
        self.is_run = False
        if len(_o_test) == 1:
            self.X_train = _o_train[self.xparams].values
            self.y_train = _o_train[self.yparam].values.ravel()
            self.X_test = _o_test[self.xparams].values
            self.y_test = _o_test[self.yparam].values.ravel()
            self.y_obs = self.y_test[0]
            self.is_run = True
        return

    def save_results(self):
        if os.path.exists(self.fname): self._sum.to_csv(self.fname,mode="a",index=False,header=False)
        else: self._sum.to_csv(self.fname,mode="w",index=False,header=True)
        return

    def pred_lstm_proba(self, trw):
        _, _,_, X_test, _,_ = self.source.get_data_for_regression_window(self.dn, trw)
        if X_test is not None: pr = self.clf.predict_proba(X_test)
        return pr

    def opt_gp(self):
        self.nrst = 10
        self.kernel = RQ(length_scale=1.0,alpha=0.1,length_scale_bounds=(1e-02, 1e2),alpha_bounds=(1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel = self.kernel, n_restarts_optimizer = self.nrst)
        return

    def run_model(self):
        pr_th = 0.5
        self._sum = pd.DataFrame()
        self._sum["date"] = [self.dn]
        self._sum["name"] = ["dgp_" + self.name]
        self._sum["kp"] = [-1.]
        self._sum["kp_pred_det"] = [-1.]
        self._sum["kp_pred_prob"] = [-1.]
        self._sum["pr"] = [-1.]
        self._sum["pr_th"] = [pr_th]
        self._sum["window"] = [self.window]
        self._sum["alt_window"] = [self.alt_window]
        if self.is_run:
            self._sum["kp"] = np.round(self.y_obs,2)
            X = self.X_test
            pr = self.pred_lstm_proba(self.window)
            if pr is not None:
                self._sum["pr"] = np.round(pr,2)
                if pr[0] > pr_th: self.data_windowing(self.alt_window)
                self.reg.fit(self.X_train, self.y_train)
                y_pred = self.reg.predict(self.X_test)
                if y_pred >= 9.: y_pred = 9.
                self._sum["kp_pred_det"] = np.round(y_pred,3)

                self.opt_gp()
                self.y_train_opt = self.reg.predict(self.X_train)
                self.gp.fit(self.X_train,self.y_train_opt)
                y_pred,y_std = self.gp.predict(self.X_test,return_std=True)
                self._sum["kp_pred_prob"] = np.round(y_pred,3)
                self._sum["std"] = np.round(y_std,3)
                pass
            pass
        self.save_results()
        return

class DeepGPRegressor(object):
    '''
    This is a regressor model, LSTM GP regressors trained here.
    '''

    def opt_regressor(self):
        self.model = assemble('GP-LSTM', [self.nn_configs['2H'], self.gp_configs['GP']])
        loss = [gen_gp_loss(gp) for gp in self.model.output_gp_layers]
        self.model.compile(optimizer=Adam(1e-2), loss=loss)
        return

    def __init__(self, dn, source, clf, k_type, window=27, alt_window=27, th = 4.5):
        self.name = "deepgpr_"+k_type.lower()
        self.window = window
        self.alt_window = alt_window
        self.dn = dn
        self.clf = clf
        self.source = source
        self.k_type = k_type
        self.th = th
    
        self.data_windowing(self.window)
        self.fname = "../out/storm/prediction_deepgpr.csv"
        return

    def ini(self):
        self.batch_size = 128
        self.epochs = 20
        self.nb_train_samples = self.X_train.shape[0]
        self.input_shape = self.X_train.shape[1:]
        self.nb_outputs = len(self.y_train)
        self.gp_input_shape = (1,)
        self.nn_params = {
                'H_dim': 16,
                'H_activation': 'tanh',
                'dropout': 0.1,
                }
        if self.k_type == "RBF":
            self.gp_params = {
                    'cov': 'SEiso',
                    'hyp_lik': -1.0,
                    'hyp_cov': [[1.], [0.0]],
                    'opt': {},
                    }
        elif self.k_type == "RQ":
            self.gp_params = {
                    'cov': 'RQiso',
                    'hyp_lik': -1.0,
                    'hyp_cov': [[1.],[1.], [0.0]],
                    'opt': {},
                    }
            pass
        self.nn_configs = load_NN_configs(filename='lstm.yaml',
                input_shape=self.input_shape,
                output_shape=self.gp_input_shape,
                params=self.nn_params)
        self.gp_configs = load_GP_configs(filename='gp.yaml',
                nb_outputs=self.nb_outputs,
                batch_size=self.batch_size,
                nb_train_samples=self.nb_train_samples,
                params=self.gp_params)
        return

    def data_windowing(self,trw):
        self.is_run, self.X_train, self.y_train, self.X_test, self.y_test, self.y_obs = self.source.get_data_for_regression_window(self.dn, trw)
        if self.is_run:
            #print self.X_train.shape, self.y_train.shape, self.X_test.shape, self.y_test.shape, trw
            self.y_train = [np.reshape(self.y_train, (len(self.y_train),1))]
            self.y_test = [np.reshape(self.y_test, (1,1))]
            pass
        return

    def save_results(self):
        if os.path.exists(self.fname): self._sum.to_csv(self.fname,mode="a",index=False,header=False)
        else: self._sum.to_csv(self.fname,mode="w",index=False,header=True)
        return

    def run_model(self):
        pr_th = 0.5
        self._sum = pd.DataFrame()
        self._sum["date"] = [self.dn]
        self._sum["name"] = [self.name]
        self._sum["kp"] = [-1.]
        self._sum["kp_pred"] = [-1.]
        self._sum["pr"] = [-1.]
        self._sum["pr_th"] = [pr_th]
        self._sum["std"] = [-1.]
        self._sum["window"] = [self.window]
        self._sum["alt_window"] = [self.alt_window]
        success = False
        if self.is_run:
            self._sum["kp"] = np.round(self.y_obs,2)
            pr = self.clf.predict_proba(self.X_test)
            self._sum["pr"] = np.round(pr,2)
            if pr[0] > pr_th: self.data_windowing(self.alt_window)
            self.database = {
                    'train': [self.X_train, self.y_train],
                    'test': [self.X_test, self.y_test],
                    }
            self.ini()
            self.opt_regressor()
            callbacks = [EarlyStopping(monitor='mse', patience=10)]
            history = train(self.model, self.database, callbacks=callbacks, gp_n_iter=5,
                    checkpoint='lstm', checkpoint_monitor='mse',
                    epochs=self.epochs, batch_size=self.batch_size, verbose=0)
            y_pred,y_std = self.model.predict(self.X_test,return_var=True)
            y_pred,y_std = np.reshape(y_pred, (1,1)), np.reshape(y_std, (1,1))
            self._sum["kp_pred"] = np.round(self.source.tx_y(y_pred),3)
            self._sum["std"] = np.round(self.source.tx_y(y_std),3)
            success = True
            pass
        self.save_results()
        return success

