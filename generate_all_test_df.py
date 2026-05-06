import pandas as pd
import os

dtest = pd.DataFrame()
mod = ['hmm','nn','dl','dt','rf','svm']
#prop_scale = ['hoop_woods','emini','parker','levitt']
prop_scale = ['Kyte-Doolittle', 'Hopp-Woods', 'Cornette', 'Eisenberg', 'Rose', 'Janin', 'Engelman GES']
dt_test = [0.1,0.2,0.3]
mod4x = mod*len(prop_scale)
prop_scale6x = prop_scale*len(mod)
dtest['algoritm'] = mod4x
dtest['prop_scale'] = prop_scale6x
dtest['accuracy'] = 0.0
dtest['auc'] = 0.0
dtest['n_amino_acids'] = 1
dtest['test_size'] = 0.0
dtest = pd.concat([dtest,dtest,dtest])
dtest = dtest.reset_index(drop=True)
#dtest = dtest.sample(frac=1).reset_index(drop=True)
for i in mod:
    dtest.loc[dtest['algoritm'] == i, 'test_size'] = int(len(dtest[dtest['algoritm'] == i])/len(dt_test))*dt_test

dtest['accuracy'] = 0.00
dtest['auc'] = 0.00

dtest.to_csv("all_result_of_classification.csv", index=False)

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

for model_ in mod:
    filepath = os.path.join(results_dir, f"{model_}_result_of_classification.csv")
    df_temp = dtest["algoritm" == model_]
    df_temp = df_temp.reset_index(drop=True)
    df_temp.to_csv(filepath, index=False)
