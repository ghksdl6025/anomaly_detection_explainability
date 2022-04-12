import pandas as pd
import utils
df = pd.read_csv('./loan_baseline.pnml_noise_0.09999999999999999_iteration_1_seed_14329.csv')

print(df.head)

key_pair = {'Case ID':'caseid', 'Activity':'activity', 'Complete Timestamp':'ts'}
df = df.rename(columns=key_pair)

if 'resource' in df.columns.values:
    df = df.loc[:,['caseid','activity','ts','resource','noise']]

else:
    df = df.loc[:,['caseid','activity','ts','noise']]

groups = df.groupby('caseid')
outcome = []
concating = []
for _, group in groups:
    group = group.sort_values(by='ts')
    group = group.reset_index(drop=True)
    actlist = list(group['activity'])
    outcomelist = actlist[1:] + ['End']

    case_length = len(group)
    group['outcome'] = outcomelist
    concating.append(group)

dfn = pd.concat(concating)
print(dfn)

for prefix in range(1,16):
    prefix_df = utils.filter_by_prefix(dfn, prefix)
    print(prefix_df)
    prefix_df.to_csv('./data/Prefix %s dataset.csv'%(prefix),index=False)