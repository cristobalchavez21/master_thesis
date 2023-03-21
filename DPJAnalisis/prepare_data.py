
import yaml
import awkward as aw
import uproot
from util import get_minit_from_procces_file, save_array_to_file, significance, get_cutted_files, ks_weighted
import hist
import math
import numpy as np
from scipy.stats import kstest
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE
from IPython.display import Audio
from matplotlib.offsetbox import AnchoredText
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import random
from sklearn.metrics import mean_squared_error
import time
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sort_jets_vbf import *
import tensorflow as tf



with open('config.yaml') as conf_file:
    config = yaml.load(conf_file, Loader=yaml.Loader) 


# #### Prepare the cuted files 


do_cut = 0
with_VBF_cut = False
if with_VBF_cut:
    VBF_cut = '_with_VBF_cut'
else:
    VBF_cut = '_without_VBF_cut'



# #### get cuted files 


# cache = uproot.LRUArrayCache("1 GB")
data_chain = uproot.lazy([f"output/{procees}_{do_cut}_cuts{VBF_cut}.root:miniT" for procees in config['ttree']['files'].keys()])#,array_cache=cache)
# consider only events with weight>0
data_s = aw.flatten(data_chain.mask[data_chain["isVBF"] == True].mask[data_chain["weight"]>=0], axis=0)
# use only ggf events with njet30>1 for training and fitting
data_b = aw.flatten(data_chain.mask[data_chain["isVBF"] == False].mask[data_chain["njet30"]>1].mask[data_chain["weight"]>=0], axis=0)
# w=data_chain["weight"]
# cache
# data_chain



# data_s2 = aw.to_numpy(aw.flatten(data_s["weight"], axis=0))
# cache



# Check the significance


s = aw.sum(data_s['weight'])
b = aw.sum(data_b['weight'])
s_e = len(data_s["weight"])
b_e = len(data_b)
s_b = s/b
# del data_s2
print(aw.sum(data_s['isVBF']))
print(f"signal: {s}, bkg: {b}, significance: {s/math.sqrt(s+b)}, asimov_significance: {significance(s,b)}")
print(f"No cuts s/b= {s_b}")
print(f"s_mc_events: {s_e}, b_mc_events: {b_e}")
# cache


# create the split train test


training_frac = 0.7
val_frac = 0.15
array_len = len(data_s["weight"])
mask = np.zeros(array_len, dtype=int)
mask1=mask.copy()
mask2=mask.copy()
mask3 = mask.copy()
mask[:int(array_len * training_frac)] = 1
mask[int(array_len * training_frac):int(array_len * (training_frac+val_frac))] = 2
np.random.shuffle(mask)
for i in range(len(mask)):
    if mask[i]==1:
        mask1[i] = 1
    elif mask[i]==2:
        mask2[i] = 1
    elif mask[i]==0:
        mask3[i] = 1
del mask
mask1 = aw.Array(mask1.astype(bool))
mask2 = aw.Array(mask2.astype(bool))
mask3 = aw.Array(mask3.astype(bool))
s_training = aw.flatten(data_s.mask[mask1], axis=0)
s_val = aw.flatten(data_s.mask[mask2],axis=0)
s_test = aw.flatten(data_s.mask[mask3],axis=0)
del mask1
del mask2
del mask3
# cache



array_len = len(data_b["weight"])
mask = np.zeros(array_len, dtype=int)
mask1=mask.copy()
mask2=mask.copy()
mask3 = mask.copy()
mask[:int(array_len * training_frac)] = 1
mask[int(array_len * training_frac):int(array_len * (training_frac+val_frac))] = 2
np.random.shuffle(mask)
for i in range(len(mask)):
    if mask[i]==1:
        mask1[i] = 1
    elif mask[i]==2:
        mask2[i] = 1
    elif mask[i]==0:
        mask3[i] = 1
del mask
mask1 = aw.Array(mask1.astype(bool))
mask2 = aw.Array(mask2.astype(bool))
mask3 = aw.Array(mask3.astype(bool))
b_training = aw.flatten(data_b.mask[mask1], axis=0)
b_val = aw.flatten(data_b.mask[mask2],axis=0)
b_test = aw.flatten(data_b.mask[mask3],axis=0)
del mask1
del mask2
del mask3
# cache


# #### Prepare Data for training


BDT_inputs = ["jets_pt_sorted", "jets_eta_sorted", "jets_phi_sorted",
                "jets_e_sorted"
             ]
# # cache


train = aw.concatenate([s_training, b_training])


w_train = train["weight"]


train[1][BDT_inputs].tolist()


# np.array(list(train[1][BDT_inputs].tolist().values()))[:,:max_jets].T


# # maximum number of jets to be used in rnn
# max_jets = 6
# train_f_dic = {}
# # crea x_train lleno de ceros (n_event, n_features)
# x_train = np.zeros((len(train),len(BDT_inputs)))
# for branch in BDT_inputs:
#     # para cada feature z es un array (n_events, max_jets)
#     Z = np.zeros((len(train[branch]), max_jets))
#     for enu, row in enumerate(train[branch]):
#         # print(enu)
#         # print(row)
#         # se llena z con cada feature hasta el maximo de jets, si son menos jets se rellena con 0
#         Z[enu, :len(row)] = row[:max_jets] 
#     train_f_dic[branch] = Z






# maximum number of jets to be used in rnn
max_jets = 6
# creates array with zeros (n_event, max_jets, n_features)
x_train = np.zeros((len(train), max_jets, len(BDT_inputs)))
n_ev = len(train)
print(n_ev)
for ev, row in enumerate(train):
    # fill the train array with shape 
    # [[[pt1, eta1, phi1, e1], [pt2, eta2, phi2, e2],     ...     , [0,0,0,0], [0,0,0,0]],
    #   [pt1, eta1, phi1, e1], [pt2, eta2, phi2, e2], ..., [pt5,eta5,phi5,e5], [0,0,0,0]],
    #    ...
    #   [pt1, eta1, phi1, e1], [pt2, eta2, phi2, e2],    ...    , [0,0,0,0], [0,0,0,0]]]]
    x_train[ev,:len(row["jets_pt_sorted"])] = np.array(list(row[BDT_inputs].tolist().values()))[:,:max_jets].T
    if ev%20000==0:
            print("Processed {} of {} entries".format(ev,n_ev))
# takes 30 hrs to run



# # maximum number of jets to be used in rnn
# max_jets = 4
# # crea x_train lleno de ceros (n_event, n_features)
# x_train_2 = np.zeros((len(train), max_jets, len(BDT_inputs)))
# n_events = len(train)
# for i in range(n_events):
#     for j in range(min(len(train[i]["jets_pt_sorted"]), max_jets)):
#         x_train_2[i,j,0] = train[i][BDT_inputs[0]][j]
#         x_train_2[i,j,1] = train[i][BDT_inputs[1]][j]
#         x_train_2[i,j,2] = train[i][BDT_inputs[2]][j]
#         x_train_2[i,j,3] = train[i][BDT_inputs[3]][j]
#     if i%1000==0:
#         print("Processed {} of {} entries".format(i,n_events))
# # horribly slow, aprox. 58 hrs to run.


train = aw.to_numpy(aw.flatten(aw.concatenate([s_training, b_training]), axis=0))
np.random.shuffle(train)
x_train = np.array(train[BDT_inputs].tolist())
y_train = train["isVBF"]
w_train = train["weight"]
del train
test = aw.to_numpy(aw.flatten(aw.concatenate([s_test, b_test]), axis=0))
np.random.shuffle(test)
x_test = np.array(test[BDT_inputs].tolist())
y_test = test["isVBF"]
w_test = test["weight"]
del test
val = aw.to_numpy(aw.flatten(aw.concatenate([s_val, b_val]), axis=0))
np.random.shuffle(val)
x_val = np.array(val[BDT_inputs].tolist())
y_val = val["isVBF"]
w_val = val["weight"]
del val


len_s_train = np.sum(y_train == 1)
len_b_train = np.sum(y_train == 0)
len_s_train_w = np.sum(w_train[y_train==1])
len_b_train_w = np.sum(w_train[y_train==0])
print(f"mc_signal_events: {len_s_train}")
print(f"mc_bkg_events: {len_b_train}")
print(f"scale_pos_weight: {len_b_train/len_s_train}")
print(f"weighted_signal_events: {len_s_train_w}")
print(f"weighted_bkg_events: {len_b_train_w}")
print(f"weighted_scale_pos_weight: {len_b_train_w/len_s_train_w}")
# scale_pos_weight=len_b_train/len_s_train
scale_pos_weight=len_b_train_w/len_s_train_w



y_val = y_val.astype("float")
y_train=y_train.astype("float")


# ## Hyperparameter Tunning
# Choosing parameters using Randomized Grid Search


from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_log_error,  make_scorer, roc_auc_score
rmsle=make_scorer(mean_squared_log_error, greater_is_better=False, squared=False)
auc = make_scorer(roc_auc_score)
scores = {"rmsle": rmsle,
            "auc": auc}
params_grid = { 
            # "max_depth": [10, 11, 12],
            "learning_rate": [0.05, 0.1],
            "gamma": [0.6, 2, 6],
            # "min_child_weight": [1, 5],
            "reg_lambda": [10, 50, 100],
            # "eval_metric": ["auc", "rmsle"],
            "scale_pos_weight":[scale_pos_weight/6, scale_pos_weight/4, scale_pos_weight/2],
            }
reg1 = xgb.XGBRegressor(n_estimators=100,
                        # scale_pos_weight=scale_pos_weight,
                        objective="binary:logistic",
                        early_stopping_rounds=10,
                        max_depth=12,
                         eval_metric=["auc", "rmsle"]
                         )
random_search = RandomizedSearchCV(estimator=reg1, 
                           param_distributions=params_grid, 
                           n_iter=25,
                           scoring=scores, 
                           refit='rmsle', 
                           n_jobs=-2, 
                           cv=5, 
                           verbose=3)
random_result = random_search.fit(x_train, y_train, sample_weight=w_train, eval_set=[(x_val, y_val)], sample_weight_eval_set=[w_val])



random_result.cv_results_


print(f'The best score is {random_result.best_score_:.4f}')
print(f'The best hyperparameters are {random_result.best_params_}')


# ## Training


# reg = cloudpickle.load(open(f'BDT_model_0.pkl', 'rb'))
# params = reg.get_params()
# reg = xgb.XGBRegressor(**params)


from tabnanny import verbose
import time

params = { "n_estimators": 800,
            "max_depth": 12,
            "learning_rate": 0.1,
            "gamma": 0.6,
            "min_child_weight": 1,
            "reg_lambda": 100,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": ["auc", "rmsle"],
            "early_stopping_rounds": 10,
            "objective":"binary:logistic",
            "verbosity": 1
            }

reg = xgb.XGBRegressor(**params)


start = time.time() # time at start of BDT fit
reg.fit(x_train, y_train, sample_weight=w_train, eval_set=[(x_val, y_val)], sample_weight_eval_set=[w_val])
elapsed = time.time() - start # time after fitting BDT
print("Time taken to fit BDT: "+str(round(elapsed,1))+"s") # print total time taken to fit BDT
print(reg)



y_pred_test = reg.predict(x_test)
y_pred_val = reg.predict(x_val)


y_pred_test


len(x_train)



#predict in all together
y_pred_train = reg.predict(x_train)



mean_squared_error(y_test, y_pred_test)


y_s = len(y_test[y_test==1])
y_b = len(y_test[y_test==0])
y_s_w = np.sum(w_test[y_test==1])
y_b_w = np.sum(w_test[y_test==0])


def puntaje_corte(s_as_s, s, b_as_b,b):
    # fraccion de señal bien identificada
    x = s_as_s/s
    #fraccion de background bien identificada
    y = b_as_b/b
    #distancia al punto maximo
    r = np.sqrt((1-x)**2 + (1-y)**2)
    # 1-r porque queremos que el puntaje aumente pare mejores cortes
    # mejor puntaje sera 1, el peor sera 0
    return 1-r, x, y
best_mc_score = [0]
best_w_score = [0]
best_mc_cut = 0
best_w_cut = 0
for cut in np.linspace(0,1,201):
    #all events predicted as vbf
    y_pred_s = y_test[y_pred_test>cut]
    # mc vbf events predicted as vbf
    s_as_s = len(y_pred_s[y_pred_s==1])
    #weights of events predicted as vbf
    w_pred_as_s = w_test[y_pred_test>cut]
    #weighted vbf events predicted as vbf
    s_as_s_w = np.sum(w_pred_as_s[y_pred_s==1])

    #all events predicted as ggf
    y_pred_b = y_test[y_pred_test<cut]
    # mc ggf events predicted as ggf
    b_as_b = len(y_pred_b[y_pred_b==0])
    #weights of events predicted as ggf
    w_pred_as_b = w_test[y_pred_test<cut]
    #weighted ggf events predicted as ggf
    b_as_b_w = np.sum(w_pred_as_b[y_pred_b==0])

    mc_score = puntaje_corte(s_as_s,y_s, b_as_b, y_b)
    w_score = puntaje_corte(s_as_s_w, y_s_w, b_as_b_w, y_b_w)
    if mc_score[0]> best_mc_score[0]:
        best_mc_cut = cut
        best_mc_score = mc_score
    if w_score[0]> best_w_score[0]:
        best_w_score = w_score
        best_w_cut = cut
print(f"best mc cut is {best_mc_cut} with score {best_mc_score[0]}, frac_vbf: {best_mc_score[1]}, \
frac_ggf: {best_mc_score[2]}")
print(f"best w_cut is {best_w_cut} with score {best_w_score[0]}, frac_vbf: {best_w_score[1]}, \
frac_ggf: {best_w_score[2]}")


# A confusion matrix is a performance measurement technique for Machine learning classification.
# True_Negative False_Negative
# False_Positive True_Positive

division_cut = best_w_cut
y_class = y_pred_test.copy()
y_class[y_pred_test>division_cut]=1.0
y_class[y_pred_test<=division_cut]=0.0
# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None, None),
                  ("Normalized confusion matrix", 'true', None),
                  ("Weighted confusion matrix, without normalization", None, w_test),
                  ("Weighted normalized confusion matrix", 'true', w_test)]
for title, normalize, w in titles_options:
    # disp = plot_confusion_matrix(reg, X_test, Y_test,
    #                              #display_labels=class_names,
    #                              cmap=plt.cm.Blues,
    #                              normalize=normalize)
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_class,
                                 normalize=normalize, 
                                 sample_weight=w,
                                 display_labels = ["ggF","VBF"],
                                 cmap=plt.cm.Blues)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()


from sklearn.metrics import roc_curve,auc
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = [20, 12]

fpr, tpr, _ = roc_curve(y_test, y_pred_test)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
# plt.savefig("auc_prueba.pdf")
print(f"AUC: {auc(fpr, tpr)}")


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

prec, recall, _ = precision_recall_curve(y_test, y_pred_test)
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()


from scipy import stats
sig_stats = stats.ks_2samp(y_pred_test[y_test == True],y_pred_train[y_train == True])
bkg_stats = stats.ks_2samp(y_pred_test[y_test == False],y_pred_train[y_train == False])
sig_p_v = sig_stats.pvalue
bkg_p_v = bkg_stats.pvalue


sig_stats.statistic


#define the hist
from scipy.stats import ks_2samp
bins = 20

h = hist.Hist.new.Reg(bins=bins, start=0, stop=1, name="MVA response")    \
    .StrCat(categories=["Train VBF","Train ggF", "Test VBF" ,"Test ggF"
    # , "val VBF", "val ggF"
    ], name="data") \
    .Double()

#fill the hist
h = h.fill(y_pred_test[y_test == True],  data="Test VBF") #weight=w_test[y_test == True],
h = h.fill(y_pred_test[y_test == False],  data="Test ggF") #weight=w_test[y_test == False],

h = h.fill(y_pred_train[y_train == True],  data="Train VBF") #weight=w_train[y_train == True],
h = h.fill(y_pred_train[y_train == False],  data="Train ggF") #weight=w_train[y_train == False],

# h = h.fill(y_pred_val[y_val == True],  data="val VBF") 
# h = h.fill(y_pred_val[y_val == False],  data="val ggF")

# print(f"The signal(background) p-value in Kolmogorov-Smirnov test is: {signal_test.pvalue:.5f}({bkg_test.pvalue:.5f}) ")
# print(f"The KS statistic is: {signal_test.statistic}({bkg_test.statistic})")


plot1 = h.stack("data")
plot1


fig, ax = plt.subplots()
at = AnchoredText(
    f"The signal(background) p-value in Kolmogorov-Smirnov test is: \n {sig_p_v:.5f}({bkg_p_v:.5f})",
    prop=dict(size=15),
    frameon=True,
    loc='upper center')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax.add_artist(at)
plot1.plot(density=True, ax=ax)
plt.legend()
fig.show()
# plt.savefig(f"plots/k-s_plot_xgb_scale_pos_weight.pdf")


h = hist.Hist.new.Reg(bins=bins, start=0, stop=1, name="ML response")    \
        .StrCat(categories=["Train signal","Train background", "Test signal" ,"Test background"], name="data") \
        .Double()

#fill the hist
h = h.fill(y_pred_test[y_test == True], weight=w_test[y_test == True], data="Test signal") #
h = h.fill(y_pred_test[y_test == False], weight=w_test[y_test == False], data="Test background") #

h = h.fill(y_pred_train[y_train == True], weight=w_train[y_train == True], data="Train signal") #
h = h.fill(y_pred_train[y_train == False], weight=w_train[y_train == False], data="Train background") #

plot1 = h.stack("data")
plot1

sig_stat_w, sig_p_v_w = ks_weighted(y_pred_test[y_test == True], y_pred_train[y_train == True], w_test[y_test == True], w_train[y_train == True] , alternative='two-sided')
bkg_stat_w, bkg_p_v_w = ks_weighted(y_pred_test[y_test == False], y_pred_train[y_train == False], w_test[y_test == False], w_train[y_train == False] , alternative='two-sided')

fig, ax = plt.subplots()
at = AnchoredText(
    f"The signal(background) p-value in Kolmogorov-Smirnov test is:{sig_p_v_w:.5f}({bkg_p_v_w:.5f})",
    prop=dict(size=15),
    frameon=True,
    loc='upper center')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax.add_artist(at)
plot1.plot(density=True, ax=ax)
plt.legend()
fig.show()
# plt.savefig(f"plots/k-s_plot_xgb_scale_pos_weight_weighted.pdf")


from numpy import save
path = "models/div1/"
# save to npy file
save(path+'x_train.npy', x_train)
save(path+'x_val.npy', x_val)
save(path+'x_test.npy', x_test)
save(path+'w_train.npy', w_train)
save(path+'w_val.npy', w_val)
save(path+'w_test.npy', w_test)
save(path+'y_train.npy', y_train)
save(path+'y_val.npy', y_val)
save(path+'y_test.npy', y_test)


import cloudpickle
with open(path+"BDT_model_scale_pos_weight_div1_v1.pkl", "wb") as o_file:
    cloudpickle.dump(reg, o_file)
# cloudpickle.dump(scaler, open('scaler.pkl', 'wb'))<



import matplotlib.pyplot as plt
import numpy as np
t = 17
plt.rcParams['figure.figsize'] = (t, t)
plt.rcParams.update({'font.size': 40})

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Create the mesh in polar coordinates and compute corresponding Z.
n=15
r = np.linspace(0, 2, n)
p = np.linspace(0, 1.5*np.pi, n)
p2 = np.linspace(0, 2*np.pi, n)
R, P = np.meshgrid(r, p)
R2, P2 = np.meshgrid(r, p2)
m=-4
l=1
Z = m*R**2+l*R**4
Z2 = m*R2**2+l*R2**4
# Express the mesh in the cartesian system.
X, Y = R*np.cos(P), R*np.sin(P)
X2, Y2 = R2*np.cos(P2), R2*np.sin(P2)

# Plot the surface.
bottom = -7
colormap = "coolwarm"
ax.plot_surface(X, Y, Z, cmap=colormap)
ax.contourf(X2, Y2, Z2,8, zdir='z', offset=bottom,cmap=colormap)

# Tweak the limits and add latex math labels.
ax.set_zlim(bottom, 0)
lim = 2.5
plt.rcParams['mathtext.fontset'] = 'cm'
# plt.rcParams['font.family'] = 'STIXGeneral'
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel(r'$|\Phi_\mathrm{1}|$')
ax.set_ylabel(r'$|\Phi_\mathrm{2}|$')
ax.set_zlabel(r'$V(\Phi^\dagger\Phi)$')
plt.setp(ax.get_xticklabels(), visible = False)
plt.setp(ax.get_yticklabels(), visible = False)
plt.setp(ax.get_zticklabels(), visible = False)

plt.show()


from sort_jets_vbf import *
with open('config.yaml') as conf_file:
    config = yaml.load(conf_file, Loader=yaml.Loader)
r, array = sort_jets("VBF_500757", config)


import numpy as np
a=np.zeros(59024)
array["jaja"] = a






