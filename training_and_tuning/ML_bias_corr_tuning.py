"""
the code to perform hyperparameter-tuning
"""
#%%
import tensorflow as tf
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle 
from lo_tools import Lfun, obs_functions
Ldir = Lfun.Lstart()
#%%
plt.rcParams.update({
    'axes.labelsize': 14,          # Labels font size
    'xtick.labelsize': 12,         # X-axis ticks font size
    'ytick.labelsize': 12,         # Y-axis ticks font size
    'axes.labelweight': 'bold',    # Bold axis labels
    'xtick.direction': 'out',       # X-axis tick direction
    'ytick.direction': 'out',       # Y-axis tick direction
    'font.size': 12,               # Default font size for other text
})
#%%
SEED = 4

os.environ["PYTHONHASHSEED"] = str(SEED)

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

tf.config.experimental.enable_op_determinism()

#%%
# where to put output figures

fig_dir0 = Ldir['LOo'] / 'LO_bias_corr' / 'plots' / 'keras'  
Lfun.make_dir(fig_dir0)

in_dir = Ldir['LOo'] / 'LO_bias_corr' / 'data'/ 'processed'
otype_list = ['bottle', 'ctd']
gtx = 'cas2k_v4_lo2b'

# turn on and off different strategies
add_hake_hpic_trawl = True
add_hake_hpic = False
cast_split = True
cast_balanced = True
z_fraction = True
benthic_aware = False
hypoxia_aware = False
hypoxia_shelf_aware = False
hypoxia_two_tier_shelf_aware = True
remove_NO3 = True

remove_dfo = True
full_z = True
filter_z = False

for otype in otype_list:
    if otype == 'bottle':
        in_fn = in_dir / ('combined_' + otype + '_all_' + '1993-2021' + '.p')
        bottle = pickle.load(open(in_fn, 'rb'))
        # Filter by z
        if filter_z:
            Z ='30-800 m'
            mask = (bottle['obs']['z'] >= -800) & (bottle['obs']['z'] <= -30)
            bottle['obs'] = bottle['obs'][mask]
            bottle[gtx] = bottle[gtx][mask]

        elif not filter_z and not full_z:
            Z ='below 30m'
            mask = (bottle['obs']['z'] <= -30)
            bottle['obs'] = bottle['obs'][mask]
            bottle[gtx] = bottle[gtx][mask]

        elif full_z:
            Z ='full_depth'
        # calculate DO bias
        bias_bottle = bottle[gtx]['DO (uM)'] - bottle['obs']['DO (uM)']

    if otype == 'ctd':
        if add_hake_hpic:
            runame = 'with_hake_hpic'
            in_fn = in_dir / ('combined_' + otype + '_all_' + '1998-2022' + '.p') # added Hake+IPHC
            if cast_split:
                name = 'cid_split'
                if cast_balanced and z_fraction:
                    name += '_cast_balanced_z_fraction'
                elif cast_balanced:
                    name += '_cast_balanced'
            else:
                name = ''
            fig_dir = fig_dir0/ runame / name
            Lfun.make_dir(fig_dir)
        elif add_hake_hpic_trawl:
            runame = 'with_hake_hpic_trawl'
            in_fn = in_dir / ('combined_' + otype + '_all_' + '1998-2022' + '.p') # added Hake+IPHC
            if cast_split:
                name = 'cid_split'
                if cast_balanced and z_fraction:
                    name += '_cast_balanced_z_fraction'
                elif cast_balanced:
                    name += '_cast_balanced'
                if benthic_aware:
                    name += '_benthic_aware'
                if hypoxia_aware:
                    name += '_hypoxia_aware'
                if hypoxia_shelf_aware:
                    name += '_hypoxia_shelf_aware'
                if hypoxia_two_tier_shelf_aware:
                    name += '_hypoxia_two_tier_shelf_aware'
                if remove_NO3:
                    name += '_no_NO3'
            else:
                name = ''
            fig_dir = fig_dir0/ runame / name
            Lfun.make_dir(fig_dir)
        else:
            in_fn = in_dir / ('combined_' + otype + '_' + '1998-2021' + '.p') # old
            runame = ''
            fig_dir = fig_dir0/ runame
            Lfun.make_dir(fig_dir)

        ctd = pickle.load(open(in_fn, 'rb'))
        if remove_dfo:
            # Drop rows where 'source' is equal to 'dfo1'
            ctd['obs'] = ctd['obs'][ctd['obs']['source'] != 'dfo1']
            ctd[gtx] = ctd[gtx][ctd[gtx]['source'] != 'dfo1']
        # Filter by z
        if filter_z:
            Z ='30-800 m'
            mask = (ctd['obs']['z'] >= -800) & (ctd['obs']['z'] <= -30)
            ctd['obs'] = ctd['obs'][mask]
            ctd[gtx] = ctd[gtx][mask]
        elif not filter_z and not full_z:
            Z ='below 30m'
            mask = (ctd['obs']['z'] <= -30)
            ctd['obs'] = ctd['obs'][mask]
            ctd[gtx] = ctd[gtx][mask]

        elif full_z:
            Z ='full_depth'
        # # calculate DO bias
        bias_ctd = ctd[gtx]['DO (uM)'] - ctd['obs']['DO (uM)']

# combine bottle and ctd
columns = list(set(bottle[gtx].columns).intersection(set(ctd[gtx].columns)))
data = pd.concat([bottle[gtx][columns], ctd[gtx][columns]],ignore_index=True)

DO_obs = pd.DataFrame({
    'DO_obs': pd.concat([bottle['obs']['DO (uM)'], ctd['obs']['DO (uM)']], ignore_index=True)
})

bias = pd.DataFrame({
    'bias': pd.concat([bias_bottle, bias_ctd], ignore_index=True)
})

data['DO_obs'] = DO_obs['DO_obs']
data['bias'] = bias['bias']
data['h'] = -data['h'].abs()
data['z_fraction_raw'] = data['z'] / data['h']
data['z_fraction'] = data['z_fraction_raw'].clip(0, 1)
data = data.loc[data['source'].ne('HABRISE')].copy()
# First perform point-level QC
data = data[data['DO_obs'] <= 400].copy()

data['cid_new'] = (
        data.groupby(
            ['source', 'lat', 'lon', 'time'],
            sort=False
        ).ngroup() + 1
)
print('Fig_fir:', fig_dir)
print('Data sources:', data['source'].unique())
print('Old unique cid:', data['cid'].nunique())
print('New unique cid:', data['cid_new'].nunique())
# Remove original cid
data = data.drop(columns='cid')
# Rename cid_new -> cid
data = data.rename(columns={'cid_new': 'cid'})

data['z_bin'] = (data['z'] / 6).round()

n_depth_bins = (
    data.groupby('cid')['z_bin']
        .transform('nunique')
)

keep = (
    data['source'].eq('GroundfishTrawl')
    | n_depth_bins.ge(5)
)

data = data.loc[keep].drop(columns='z_bin').copy()

print('After filtering, number of unique cids:', data['cid'].nunique())
print('Number of rows:', len(data))

#%%
# convert time to numerical if time needs to be an input variable
# Convert to day of year (1 to 365)
day_of_year = data['time'].dt.dayofyear
# Encode as sine and cosine
data['sin_t'] = np.sin(2 * np.pi * day_of_year / 365)
data['cos_t'] = np.cos(2 * np.pi * day_of_year / 365)

plot = False
if plot:
    # day 90 (spring) and day 270 (fall) would both have cosine values near 0, but they are very different seasons
    plt.figure(figsize=(8,4))
    # plt.plot(data['time'], data['sin_time'], marker = '.', linestyle = '-', color = 'black', label='sin_time')
    # plt.plot(data['time'], data['cos_time'], marker = '.', linestyle = '-', color = 'red',label='cos_time')
    plt.plot(data['time'], data['sin_t'], marker = '.', linestyle = 'None', color = 'black', label='sin_time')
    plt.plot(data['time'], data['cos_t'], marker = '.', linestyle = 'None', color = 'red',label='cos_time')
    plt.title('Seasonal Encoding with Sine and Cosine')
    plt.xlabel('Date')
    plt.ylabel('Encoded Value')
    plt.xlim(pd.Timestamp('2007-01-01'), pd.Timestamp('2009-01-31'))
    plt.legend(loc = 'lower left')
    plt.grid(True)
    # plt.tight_layout()
    plt.show()

#%%
# =========================================================
# 1. choose predictors
# =========================================================
# Variables used by the MLP
predictors = [
    'lon',
    'lat',
    'z',
    'cos_t',
    'sin_t',
    'CT',
    'SA',
    'DO (uM)',
    'NO3 (uM)',
    'h',
    'z_fraction',
]
if remove_NO3:
    predictors.remove('NO3 (uM)')
# Keep DO_obs and cid for evaluation / grouping
vn = predictors + ['DO_obs', 'cid']

X = data[vn]
# choose output
y = data['bias'].values
#%%
# =========================================================
# 2. Split 80% development / 20% testing BY CAST
# =========================================================

from sklearn.model_selection import GroupShuffleSplit

# Split by cast (cid)
gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=SEED
)

train_idx, test_idx = next(
    gss.split(X, y, groups=data['cid'])
)

X_train = X.iloc[train_idx]
X_test  = X.iloc[test_idx]

y_train = y[train_idx]
y_test  = y[test_idx]
#%%
# =========================================================
# 3. Split initial training into 80% train / 20% validation
#    BY CAST
# =========================================================

gss_val = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED
)

train2_idx, val_idx = next(
    gss_val.split(
        X_train,
        y_train,
        groups=X_train['cid']
    )
)

X_train2 = X_train.iloc[train2_idx].copy()
X_val    = X_train.iloc[val_idx].copy()

y_train2 = y_train[train2_idx]
y_val    = y_train[val_idx]


#%%
# =========================================================
# 4. Check that casts are completely independent
# =========================================================

cid_train = X_train2['cid'].unique()
cid_val   = X_val['cid'].unique()
cid_test  = X_test['cid'].unique()

print('Number of casts:')
print('Training:  ', len(cid_train))
print('Validation:', len(cid_val))
print('Testing:   ', len(cid_test))

print('\nCast overlap:')
print('Train-Val: ', len(np.intersect1d(cid_train, cid_val)))
print('Train-Test:', len(np.intersect1d(cid_train, cid_test)))
print('Val-Test:  ', len(np.intersect1d(cid_val, cid_test)))

print('\nNumber of observations:')
print('Training:  ', len(X_train2))
print('Validation:', len(X_val))
print('Testing:   ', len(X_test))

# =========================================================
# 4b. Construct cast-balanced sample weights
# Each cast has equal total weight
# =========================================================
def make_cast_two_tier_hypoxia_shelf_weights(
    df,
    cid_col='cid',
    h_col='h',
    obs_col='DO_obs',
    hypoxia_threshold=62.2,
    low_oxygen_threshold=125.0,
    hypoxia_factor=10.0,
    low_oxygen_factor=1.5,
    shelf_depth=200.0,
    shelf_factor=1.2,
):
    # Same cast-balanced base weight as your original scheme
    n_per_cast = (
        df.groupby(cid_col)[cid_col]
        .transform('size')
        .to_numpy()
    )
    cast_base = 1.0 / n_per_cast

    # Only change: 10 : 1.5 : 1 oxygen multipliers
    do_obs = df[obs_col].to_numpy()
    oxygen_multiplier = np.where(
        do_obs <= hypoxia_threshold,
        hypoxia_factor,                       # 10: hypoxic
        np.where(
            do_obs <= low_oxygen_threshold,
            low_oxygen_factor,                # 2: intermediate low oxygen
            1.0                               # 1: otherwise
        )
    )

    # Same fixed shelf weighting as before
    shelf_multiplier = np.where(
        np.abs(df[h_col].to_numpy()) <= shelf_depth,
        shelf_factor,
        1.0
    )

    w = cast_base * oxygen_multiplier * shelf_multiplier

    # Same overall rescaling as before
    w = w / w.mean()

    return w.astype(np.float32)


w_train = make_cast_two_tier_hypoxia_shelf_weights(
    X_train2,
    hypoxia_factor=15.0,
    low_oxygen_factor=1.5,
    low_oxygen_threshold=125.0,
    shelf_factor=1.2,
)

w_val = make_cast_two_tier_hypoxia_shelf_weights(
    X_val,
    hypoxia_factor=15.0,
    low_oxygen_factor=1.5,
    low_oxygen_threshold=125.0,
    shelf_factor=1.2,
)

# =========================================================
# 5. Standardize predictors
# Fit scaler ONLY using final training dataset
# =========================================================
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
scaler_y = StandardScaler()

standardized_X = scaler_x.fit_transform(
    X_train2[predictors]
)

standardized_X_val = scaler_x.transform(
    X_val[predictors]
)

standardized_X_test = scaler_x.transform(
    X_test[predictors]
)

# Standardize target (bias)
standardized_y = scaler_y.fit_transform(
    y_train2.reshape(-1, 1)
)

standardized_y_val = scaler_y.transform(
    y_val.reshape(-1, 1)
)

standardized_y_test = scaler_y.transform(
    y_test.reshape(-1, 1)
)

#%%
# save the scaler
import joblib
out_dir = Ldir['LOo'] / 'LO_bias_corr'/'keras'/runame/name
Lfun.make_dir(out_dir)
joblib.dump(scaler_y, f'{out_dir}/scaler_y.pkl')
joblib.dump(scaler_x, f'{out_dir}/scaler_x.pkl')
#%%
# import packages
from tensorflow import keras
from keras import Sequential
from tensorflow.keras import layers, regularizers
from keras.layers import Dense
from keras.optimizers import Adam
from keras.metrics import RootMeanSquaredError
import keras_tuner as kt

import numpy as np, random, tensorflow as tf

np.random.seed(4); random.seed(4); tf.random.set_seed(4)

"""
Hyperparameter Tuning
 
# * Number of hidden layers
# * Neurons per hidden layer:
# * Batch size:
# * Learning rate:
#     * what about activation?
#         * activation is like the action of firing the neurons
#     * add regularizers when the model might be over-fitting?
"""

# 1. Define an MLP HyperModel
class MLPHyperModel(kt.HyperModel):
    def __init__(self, input_dim):
        self.input_dim = input_dim

    def build(self, hp):
        model = keras.Sequential(name="MLP_tuned")
        activation = 'gelu'
        # activation = hp.Choice("activation", values=["gelu", "relu", "swish"])
        # Learning rate
        lr = hp.Float("learning_rate", min_value=1e-5, max_value=1e-2, sampling="log")

        # no need for drop_out
        dropout_rate = 0
        # dropout_rate = hp.Float("dropout", min_value=0.0, max_value=0.4, step=0.1)
        # Number of hidden layers
        n_layers = hp.Int("n_layers", min_value=1, max_value=4, step=1)

        # L2 regularization
        l2_value = hp.Choice(
            "l2",
            values=[0.0, 1e-5, 1e-4, 1e-3]
        )
        # NN
        model.add(layers.Input(shape=(self.input_dim,)))

        for i in range(n_layers):
            units = hp.Choice(f"units_{i}", values=[16, 32, 64, 128, 256])
            model.add(layers.Dense(units, activation=activation, kernel_regularizer=regularizers.l2(l2_value)))
            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate))

        # Output
        model.add(layers.Dense(1))
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="mse",
            metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
            weighted_metrics=[
                keras.metrics.RootMeanSquaredError(name="weighted_rmse")
            ],
        )
        return model

# 2. Create the tuner and run search
input_dim = standardized_X.shape[1]

print("Input dimension:", input_dim)

print("Training X:", standardized_X.shape)
print("Training y:", standardized_y.shape)

print("Validation X:", standardized_X_val.shape)
print("Validation y:", standardized_y_val.shape)

print("NaN in training X:", np.isnan(standardized_X).any())
print("NaN in training y:", np.isnan(standardized_y).any())


# delete this, and use 256
class MyTuner(kt.Hyperband):
    def run_trial(self, trial, *args, **kwargs):
        # hp = trial.hyperparameters
        kwargs["batch_size"] = 256
        # kwargs["batch_size"] = hp.Choice("batch_size", values=[128, 256, 512]) # not necessary for this
        return super().run_trial(trial, *args, **kwargs)


tuner = MyTuner(
    MLPHyperModel(input_dim),
    objective="val_loss",
    max_epochs=300,
    factor=3,  # how aggressively Hyperband reduces the number of candidate configurations as it allocates more training resources
    directory=str(out_dir),
    project_name="mlp_bias_corr_cast_balanced_no_NO3",
    overwrite=True,
    seed=SEED,
)
print("Hyperparameter tuning will be saved to:")
print(out_dir / "mlp_bias_corr_cast_balanced_no_NO3")

es = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True,
)

tuner.search(
    standardized_X, standardized_y,
    sample_weight=w_train,  # cast-balanced weight
    # validation_data=(standardized_X_val, standardized_y_val),
    validation_data = (standardized_X_val, standardized_y_val, w_val),
    epochs=300,
    callbacks=[es],
    shuffle=True,
)

# 3. Get the best model + best hyperparameters
best_hp = tuner.get_best_hyperparameters(1)[0]
print(best_hp.values)

# best_batch_size = best_hp.get("batch_size") # no need for this

#%%
"""
build best_model mannually just like how it was done in the original notebook
build two, one with regularization, the other without
Zen thought without reg maybe better
"""
# best_model = tuner.get_best_models(1)[0]
best_model = tuner.hypermodel.build(best_hp)
#%%
patience= 20
# Fresh early-stopping callback
es_final = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=patience,
    restore_best_weights=True,
)
# Retrain from scratch
history = best_model.fit(
    standardized_X, standardized_y,
    sample_weight=w_train,  # cast-balanced weight
    # validation_data=(standardized_X_val, standardized_y_val),
    validation_data = (standardized_X_val, standardized_y_val, w_val),
    epochs=300,
    batch_size=256,
    callbacks=[es_final],
    shuffle=True,
    verbose=1,
)
#%%
# display the model summary
best_model.summary()
#%%
best_epoch = np.argmin(history.history['val_loss']) + 1
best_val_loss = np.min(history.history['val_loss'])

print('Best epoch:', best_epoch)
print('Best validation loss:', best_val_loss)
print('Stopped at:', len(history.history['val_loss']))

#%%
# check the model performance
plt.rcParams['figure.figsize'] = (6,5)

epochs_range = np.arange(len(history.history['loss']), dtype=int)

loss = np.array(history.history['loss'])
val_loss = np.array(history.history['val_loss'])

plt.plot(epochs_range, loss, label='Training loss')
plt.plot(epochs_range, val_loss, label='Validation loss')

plt.title('Monitoring')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.ylim([0, 1])
plt.xlim([0, len(epochs_range)])

plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f'{fig_dir}/loss_function_z={Z}_epochs={best_epoch}_patience={patience}.png')
#%%
# 5. Make predictions using testing data
predictions_test = best_model.predict(
                            standardized_X_test,
                            verbose =0,
                            )

predictions_val = best_model.predict(
                            standardized_X_val,
                            verbose =0,
                        )

predictions_train = best_model.predict(
                            standardized_X,
                            verbose =0,
                        )

print(predictions_train.shape)
print(predictions_val.shape)
print(predictions_test.shape)
#%%
# Calculate MSE between scaled true and predicted bias
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(
    standardized_y_test, predictions_test
)  # testing

mse1 = mean_squared_error(
    standardized_y_val, predictions_val
)  # validation

mse2 = mean_squared_error(
    standardized_y, predictions_train
)  # training

print("Testing MSE (scaled):", mse)
print("Validation MSE (scaled):", mse1)
print("Training MSE (scaled):", mse2)
#%%
""" inverse the predicted results back to real values"""
# Inverse-transform predictions
y_pred = scaler_y.inverse_transform(predictions_test).ravel()
y_pred_val = scaler_y.inverse_transform(predictions_val).ravel()
y_pred_train = scaler_y.inverse_transform(predictions_train).ravel()

# True values are already in original units
y_true_test = y_test
y_true_val = y_val
y_true_train = y_train2


""" next, calculate the MSE, RMSE """
# MSE
mse_test = mean_squared_error(y_true_test, y_pred)
mse_val = mean_squared_error(y_true_val, y_pred_val)
mse_train = mean_squared_error(y_true_train, y_pred_train)

# RMSE
rmse_test = np.sqrt(mse_test)
rmse_val = np.sqrt(mse_val)
rmse_train = np.sqrt(mse_train)

print(f"Training RMSE:   {rmse_train:.2f} µmol/L")
print(f"Validation RMSE: {rmse_val:.2f} µmol/L")
print(f"Testing RMSE:    {rmse_test:.2f} µmol/L")
#%%
rmse = np.sqrt(mse_test)
# bias = np.nanmean(np.array(y_pred) - np.array(y_test))
""" get the 1:1 plot, and calculate the r, slope, mse etc"""
bias_lim = [-250, 250]
# plot y_pred, y-test
plt.figure(figsize=(6, 6))
plt.scatter(y_true_test,y_pred,  alpha=0.6)
plt.plot(bias_lim, bias_lim, '--g', label='1:1 Line')  # 1:1 reference line
plt.text(0.25,.95,'rmse=%0.1f' % (rmse),
         c='k', transform=plt.gca().transAxes, ha='right', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.5),
         fontsize=12*1.0,style='italic')
plt.xlabel('Testing bias')
plt.ylabel('Predicted bias')
plt.xlim(bias_lim)
plt.ylim(bias_lim)
# Set ticks every 50 units
plt.xticks(np.arange(bias_lim[0], bias_lim[1]+1, 50))
plt.yticks(np.arange(bias_lim[0], bias_lim[1]+1, 50))
plt.tight_layout()
plt.savefig(f'{fig_dir}/pred_test_{runame}_{name}.png')
plt.show()
#%%
plt.figure(figsize=(8, 5))

plt.hist(y_true_test.ravel(), bins=50, alpha=0.6, label='Testing', color='blue')
plt.hist(y_pred.ravel(), bins=50, alpha=0.6, label='Predictions', color='orange')

plt.xlabel('Bias')
plt.ylabel('Frequency')
plt.legend()
plt.title('Comparison of testing and predictions')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{fig_dir}/pred_test_histgram_{runame}_{name}.png')
plt.show()
#%%
"""
add the predicted bias back to modeled DO,
plot bias corrected DO with obs DO
"""

# Original modeled DO
y_model = X_test['DO (uM)'].to_numpy()
# Predicted bias from MLP, already inverse-transformed
y_pred = np.asarray(y_pred).ravel()
# Corrected DO
# bias = modeled DO - observed DO
# therefore corrected DO = modeled DO - predicted bias
y = y_model - y_pred

# Observed DO
x = X_test['DO_obs'].to_numpy()
#%%
# Save all four objects in a single pickle file

out_fn = out_dir / 'test_pred_data.p'

test_results = {
    'X_test': X_test,
    'y_pred': y_pred,          # predicted bias
    'y_test': y_test,          # true bias
    'corrected_DO': y,
    'obs_DO': x,
}

with open(out_fn, 'wb') as f:
    pickle.dump(test_results, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f'Test_results saved to {out_fn}')
#%%
"""
add the predicted bias back to modeled DO,
plot bias corrected DO with obs DO
"""


bias = np.nanmean(y-x)
rmse = np.sqrt(np.nanmean((y-x)**2))

""" get the 1:1 plot, and calculate the r, slope, mse etc"""
DO_lim = [0, 400]

plt.figure(figsize=(6.5, 6))
plt.scatter(x,y_model,  alpha=0.6, color = 'cyan',s = 0.05)
plt.scatter(x,y,  alpha=0.6, color = 'black', s = 0.05)
plt.plot(DO_lim, DO_lim, '--g', label='1:1 Line')  # 1:1 reference line
plt.text(0.5,.95,'bias=%0.1f, rmse=%0.1f' % (bias, rmse),
         c='k', transform=plt.gca().transAxes, ha='right', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.5),
         fontsize=12*1.0,style='italic')
# plt.text(0.25,.95,'rmse=%0.1f' % (rmse),
#          c='k', transform=plt.gca().transAxes, ha='right', fontweight='bold',
#          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.5),
#          fontsize=12*1.0,style='italic')
plt.xlabel('DO Obs')
plt.ylabel('DO bias_corrected')
plt.xlim(DO_lim)
plt.ylim(DO_lim)
# Set ticks every 50 units
plt.xticks(np.arange(DO_lim[0], DO_lim[1]+1, 50))
plt.yticks(np.arange(DO_lim[0], DO_lim[1]+1, 50))
plt.tight_layout()
plt.savefig(f'{fig_dir}/bias_corrected_DO_{runame}_{name}.png')
plt.show()
#%%
"""
(3) plot the histogram residuals (bias corrected DO - obs DO)
"""
# Residuals
residuals = y - x  # bias-corrected DO - obs DO
mean = np.nanmean(residuals)
print('mean Residuals is:', mean)
plt.figure(figsize=(6, 5))
# Define bins
bin_min = np.floor(np.nanmin(residuals) / 5) * 5
bin_max = np.ceil(np.nanmax(residuals) / 5) * 5
bins = np.arange(bin_min, bin_max + 5, 5)  # step = 5

counts, bins, patches = plt.hist(residuals, bins=bins, edgecolor='black', alpha=0.7)

plt.ylim(0, counts.max()*1.1)

plt.axvline(0, color='black', linestyle='--', linewidth=2)
plt.text(0.95,.95, f'mean = {mean:.1f}',
          transform=plt.gca().transAxes, ha='right', fontweight='bold',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.5),
          fontsize=12*1.0,style='italic')

plt.xlabel('DO Residuals ($\mu mol\ L^{-1}$)')
plt.ylabel('Frequency')
plt.xlim([-100, 100])  # Set x-axis limits
# plt.ylim([0, 10e4])  # Set y-axis limits
plt.grid(True, linestyle='--', alpha=0.5)
# plt.title('Histogram of DO Residuals')
plt.tight_layout()
plt.savefig(f'{fig_dir}/residuals_histogram_{runame}_{name}.png')
plt.show()
#%%
"""
y = X_test['DO (uM)'] - y_pred.ravel() # y is bias_corrected value
x = X_test['DO_obs'] # x is observation value
z = X_test['z']
lon = X_test['lon']
plot two depth profiles of y (red )and x  (black) against z at lon.unique()
"""

X_test = X_test.copy()
X_test['bias_corrected_DO'] = X_test['DO (uM)'] - y_pred.ravel()
# Get unique (lat, lon) groups
# Group by unique cid
grouped = list(X_test.groupby('cid'))

DO_lim = [0, 400]
n_per_fig = 6
n_rows, n_cols = 2, 3

for i in range(0, len(grouped), n_per_fig):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    batch = grouped[i:i + n_per_fig]

    for ax, (cid, group) in zip(axes.flat, batch):
        group = group.sort_values('z')  # optional: sort by depth

        ax.plot(group['DO (uM)'], group['z'], 'r.-', label='Model', markersize=8, linewidth=0)
        ax.plot(group['bias_corrected_DO'], group['z'], 'g.-', label='Bias-corrected', markersize=8, linewidth=0)
        ax.plot(group['DO_obs'], group['z'], 'k.-', label='Obs.', markersize=6, linewidth=0)

        bias = np.nanmean(group['bias_corrected_DO'] - group['DO_obs'])
        rmse = np.sqrt(np.nanmean((group['bias_corrected_DO'] - group['DO_obs'])**2))

        lat = group['lat'].iloc[0]
        lon = group['lon'].iloc[0]

        ax.text(0.95, 0.05, f'bias={bias:.1f}\nrmse={rmse:.1f}',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.6))

        ax.set_title(f'cid={cid}\nLat={lat:.1f}, Lon={lon:.2f}', fontsize=10)
        ax.set_xlim(DO_lim)
        ax.invert_yaxis()  # for depth
        ax.xaxis.set_label_position('top')  # move x-axis label to top
        ax.xaxis.tick_top()
        ax.tick_params(labelsize=8)

    # Remove unused axes
    for j in range(len(batch), n_rows * n_cols):
        fig.delaxes(axes.flat[j])

    fig.text(0.5, 0.04, 'DO ($\mu mol\ L^{-1}$)', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'Depth (m)', va='center', rotation='vertical', fontsize=12)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3)

    fig.tight_layout(rect=[0.05, 0.08, 1, 0.95])

    plt.savefig(f'{fig_dir}/profile_bias_corr_DO_vs_obsDO_04_z={Z}_epochs={best_epoch}_patience={patience}_fig{i // n_per_fig + 1}_{runame}_{name}.png')
    plt.close()


#%%
# save the model, this includes architecture, weights, and optimizer state:
best_model.save(f'{out_dir}/DO_bias_corr_best_model.keras')

import json

with open(f'{out_dir}/best_hyperparameters.json', 'w') as f:
    json.dump(best_hp.values, f, indent=4)
# This creates a .keras directory (or single file) that can be reloaded later with:
# model = keras.models.load_model(f'{out_dir}/DO_bias_corr.keras')