"""
effects of volume and depth on dsi measurement
two way anova
"""

import os
import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp


df_path = r"G:\bulk_LGN_database\dataframe_190530171338.csv"
# df_path = r"G:\bulk_LGN_database\dataframe_190530171338_axon_AllStimuli_DistanceThr_1.30.csv")

depths = [50, 100, 150, 200, 250, 300, 350, 400,]
mouse_ids = ['M360495', 'M376019', 'M386444', 'M426525', 'M439939', 'M439943']

# depths = [50, 100, 150, 200, 250, 300, 350, 400,]
# mouse_ids = ['M439943']

# depths = [50, 100, 150, 200, 250, 300, 350]
# mouse_ids = ['M439939']

# depths = [50, 100, 150, 200, 250, 300]
# mouse_ids = ['M426525']

# depths = [100, 200, 300]
# mouse_ids = ['M386444']

# depths = [50, 100, 150, 200, 250]
# mouse_ids = ['M376019', 'M360495']

# dire_type = 'peak_dire' # 'vs_dire' or 'peak_dire'
response_dir = 'pos'
response_type = 'dff'
post_process_type = 'ele' # 'raw', 'ele' or 'rec'
skew_thr = 0.6
dgc_peak_z_thr = 3.
dgc_p_anova_thr = 0.01
dsi_type = 'gdsi'
# dsi_thr = 0.5

# nti_half_span = 45.
# nti_sum_thr = 10

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

print('loading csv file: {}'.format(df_path))
df = pd.read_csv(df_path)
print('csv file loaded.')


df = df[(df['mouse_id'].isin(mouse_ids)) &
        (df['depth'].isin(depths)) &
        (df['skew_fil'] >= skew_thr) &
        (df['dgc_{}_peak_z'.format(response_dir)] >= dgc_peak_z_thr) &
        (df['dgc_p_anova_{}'.format(response_type)] <= dgc_p_anova_thr)]

model = ols(f'dgc_{response_dir}_{dsi_type}_{post_process_type}_{response_type} ~ ' \
            f'C(depth)*C(vol_n)', df).fit()

# model = ols(f'dgc_{response_dir}_{dsi_type}_{post_process_type}_{response_type} ~ ' \
            # f'C(depth)*C(mouse_id)', df).fit()

# model = ols(f'dgc_{response_dir}_{dsi_type}_{post_process_type}_{response_type} ~ ' \
#             f'C(depth)*C(vol_n)*C(mouse_id)', df).fit()

print(f"Overall model F({model.df_model: .0f},{model.df_resid: .0f})" \
      f"= {model.fvalue: .3f}, p = {model.f_pvalue: .4f}")

print(model.summary())

res = sm.stats.anova_lm(model, typ=2)

def anova_table(aov):
    aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']
    
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])
    
    cols = ['sum_sq', 'mean_sq', 'df', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
    aov = aov[cols]
    return aov

print(anova_table(res))