import pandas as pd
import utils
df = pd.read_csv('./preprocessed_loan_baseline.pnml_noise_0.09999999999999999_iteration_1_seed_14329_sample.csv')
print(df.head)

key_pair = {'Case ID':'caseid', 'Activity':'activity', 'Complete Timestamp':'ts'}
df = df.rename(columns=key_pair)

if 'resource' in df.columns.values:
    df = df.loc[:,['caseid','activity','ts','resource','noise']]

else:
    df = df.loc[:,['caseid','activity','ts','noise']]


for prefix in range(1,16):
    prefix_df = utils.filter_by_prefix(df, prefix)
    prefix_df.to_csv('./data/Prefix %s dataset.csv'%(prefix),index=False)