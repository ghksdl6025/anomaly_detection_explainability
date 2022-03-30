import pandas as pd
df = pd.read_csv('./loan_baseline.pnml_noise_0.049999999999999996_iteration_1_seed_42477_sample.csv')

remove_act = []
print(df.head)
for pos,x in enumerate(list(df['noise'])):
    if x =='Start' or x =='End':
        remove_act.append(pos)

dft = df.drop(remove_act, axis=0)
dft = dft.loc[:,['Case ID','Activity','Complete Timestamp','noise']]
dft.to_csv('./preprocessed_loan_baseline.pnml_noise_0.049999999999999996_iteration_1_seed_42477_sample.csv',index=False)
print(dft.head)