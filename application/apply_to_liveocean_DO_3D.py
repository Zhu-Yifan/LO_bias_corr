#%%
"""
code to apply bias-corrected DO to daily-3D field
filter salish sea
for shelf-only
for April-Sep only
"""
#%%
import numpy as np
import xarray as xr
import pandas as pd
import gsw
import gc
from pathlib import Path
from lo_tools import Lfun
import joblib
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
#%%
Ldir = Lfun.Lstart()
# input location
gtx = 'cas2k_v4_lo2b'
in_dir0 = Ldir['roms_out2'] / gtx
#%%
# load trained model
dir = Ldir['LOo'] / 'LO_bias_corr'/'keras'
runame = 'with_hake_hpic_trawl'
name = 'cid_split_cast_balanced_z_fraction_hypoxia_two_tier_shelf_aware_no_NO3'
in_dir = dir/ runame / name
# Load the saved model
model = load_model(f'{in_dir}/DO_bias_corr.keras')
# load scaler_x and y saved in training step
scaler_x = joblib.load(f'{in_dir}/scaler_x.pkl')
scaler_y = joblib.load(f'{in_dir}/scaler_y.pkl')
#%%
def process_daily_file(fn):

    fn = Path(fn)
    output_file = fn.parent / 'ocean_lp_bias_corrected.nc'

    print(f'Processing: {fn}')

    with xr.open_dataset(fn) as dt:

        # ---------------------------------------
        # 1. Filter out Salish Sea
        # ---------------------------------------
        lat = dt['lat_rho']
        lon = dt['lon_rho']

        exclude_mask = (
            ((lat > 45.0) & (lat < 50.0) &
             (lon >= -123.8) & (lon < -122.0))

            | ((lat > 48.9) & (lat < 51.0) &
               (lon > -125.5) & (lon < -123.7))

            | ((lat > 50.5) & (lat < 51.0) &
               (lon > -126.5) & (lon < -125.5))

            | ((lat > 51.75) & (lat < 52.1) &
               (lon > -128.0) & (lon < -127.6))

            | ((lat >= 48.38) & (lat < 48.7) &
               (lon >= -124.65) & (lon < -123.725))

            | ((lat >= 48.15) & (lat < 48.7) &
               (lon >= -124.65) & (lon < -123.725))
        )

        # ---------------------------------------
        # 2. Shelf mask: 15–200 m
        # ---------------------------------------
        shelf_mask = (
            (dt['mask_rho'] == 1) &
            (~exclude_mask) &
            (dt['h'] > 14.99) &
            (dt['h'] < 200.1)
        )

        variables = [
            'temp', 'salt', 'oxygen', 'NO3',
            'zeta', 'h', 'mask_rho', 'hc', 'Cs_r'
        ]

        ds = dt[variables].where(shelf_mask, drop=True)

        # ---------------------------------------
        # 3. ROMS vertical coordinates
        # ---------------------------------------
        ds['Vtransform'] = 2

        Zo_rho = (
            ds['hc'] * ds['s_rho'] +
            ds['Cs_r'] * ds['h']
        ) / (ds['hc'] + ds['h'])

        z_rho = ds['zeta'] + (
            ds['zeta'] + ds['h']
        ) * Zo_rho

        ds['z_rho'] = z_rho.transpose(
            'ocean_time', 's_rho', 'eta_rho', 'xi_rho'
        )

        ds['z_rho'].attrs = {
            'units': 'm',
            'long_name': 'Vertical position on s_rho grid, positive up'
        }

        # ---------------------------------------
        # 4. TEOS-10 calculations
        # ---------------------------------------
        varT = ds['temp'].values
        varS = ds['salt'].values

        lat = ds['lat_rho'].values
        lon = ds['lon_rho'].values
        z = ds['z_rho'].values

        varP = gsw.p_from_z(z, lat)

        SA = gsw.SA_from_SP(varS, varP, lon, lat)
        CT = gsw.CT_from_pt(SA, varT)

        CT = xr.DataArray(
            CT,
            dims=ds['temp'].dims,
            coords=ds['temp'].coords
        )

        SA = xr.DataArray(
            SA,
            dims=ds['salt'].dims,
            coords=ds['salt'].coords
        )

        # ---------------------------------------
        # 5. Other predictors
        # ---------------------------------------
        DO = ds['oxygen']
        h = -ds['h']

        z_fraction = (
            np.abs(ds['z_rho']) / np.abs(h)
        ).clip(min=0, max=1)

        time = ds['ocean_time']
        day_of_year = time.dt.dayofyear

        sin_t = np.sin(
            2 * np.pi * day_of_year / 365
        )

        cos_t = np.cos(
            2 * np.pi * day_of_year / 365
        )

        # ---------------------------------------
        # 6. Construct feature dataset
        # ---------------------------------------
        feat = xr.Dataset({
            'lon': ds['lon_rho'],
            'lat': ds['lat_rho'],
            'z': ds['z_rho'],
            'cos_t': cos_t,
            'sin_t': sin_t,
            'CT': CT,
            'SA': SA,
            'DO_uM': DO,
            'h': h,
            'z_fraction': z_fraction,
        })

        feat = feat.broadcast_like(CT)

        # ---------------------------------------
        # 7. Stack into 1D samples
        # ---------------------------------------
        feat_stacked = feat.stack(
            sample=(
                'ocean_time',
                's_rho',
                'eta_rho',
                'xi_rho'
            )
        )

        # Predictor order must match training
        vn = [
            'lon',
            'lat',
            'z',
            'cos_t',
            'sin_t',
            'CT',
            'SA',
            'DO_uM',
            'h',
            'z_fraction',
        ]

        X_pd = pd.DataFrame({
            k: feat_stacked[k].values for k in vn
        })

        # ---------------------------------------
        # 8. Remove invalid samples
        # ---------------------------------------
        valid = np.isfinite(X_pd).all(axis=1)
        X = X_pd.loc[valid, vn]

        # print(f'Valid samples: {len(X):,}')

        corrected_all = np.full(
            len(X_pd), np.nan, dtype=np.float32
        )

        # ---------------------------------------
        # 9. Neural-network prediction
        # ---------------------------------------
        if len(X) > 0:

            standardized_X = scaler_x.transform(
                X.values
            )

            print('Applying bias correction...')

            y_pred = model.predict(
                standardized_X,
                batch_size=4096,
                verbose=0
            )

            bias = scaler_y.inverse_transform(
                y_pred
            ).ravel()

            # Assumes bias = DO_model - DO_obs
            corrected_all[valid.to_numpy()] = (
                X['DO_uM'].to_numpy() - bias
            )

        # ---------------------------------------
        # 10. Restore original dimensions
        # ---------------------------------------
        corrected_da = xr.DataArray(
            corrected_all,
            dims=('sample',),
            coords={
                'sample': feat_stacked['sample']
            }
        )

        corrected_da = corrected_da.unstack('sample')

        ds['corrected_DO'] = corrected_da.transpose(
            *ds['oxygen'].dims
        )
        ds['corrected_DO'].attrs = ds['oxygen'].attrs.copy()
        ds['corrected_DO'].attrs['long_name'] = 'Bias-corrected dissolved oxygen'

        # print('Original DO shape:', ds['oxygen'].shape)
        # print('Corrected DO shape:', ds['corrected_DO'].shape)

        # ---------------------------------------
        # 11. Save NetCDF
        # ---------------------------------------
        ds.to_netcdf(
            output_file,
            mode='w',
            format='NETCDF4'
        )

        print(f'Saved: {output_file}')

        ds.close()

    # ---------------------------------------
    # 12. Memory cleanup
    # ---------------------------------------
    gc.collect()
#%%
testing = False
if testing:
    year_list = [1993]
else:
    year_list = np.arange(1993,2023)
#%%
for year in year_list:

    dates = pd.date_range(
        start=f'{year}-04-01',
        end=f'{year}-09-30',
        # end=f'{year}-04-01',
        freq='D'
    )

    for date in dates:

        fod = f"f{date:%Y.%m.%d}"

        fn = in_dir0 / fod / 'ocean_lp.nc'

        if not fn.exists():
            print(f'Missing file: {fn}')
            continue

        process_daily_file(fn)