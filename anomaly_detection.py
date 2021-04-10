from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from numpy import quantile, where, random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

Scaler = StandardScaler

data_df = pd.read_csv('housesToRentProcessed_train_full.csv', index_col=0).reset_index()

data = data_df.values[:, 1:]
Y = data[:, -1]

N = len(data)

X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=0.15, random_state=42)


BIC = []
for n_component in range(1, 15):
    gmm = GaussianMixture(n_components=n_component, max_iter=1500)
    gausMix = gmm.fit(X_train)

    bic = gausMix.bic(X_test)
    BIC.append(bic)
    print('num_componentes: ' + str(n_component) + ' BIC: ' + str(bic))
#
n_component = np.array(BIC).argsort()[0] + 1

plt.plot(range(1, 15), BIC)
plt.ylabel('BIC')
plt.title('Bayesian Information Criterion')
plt.show()

print('n_component: ', n_component)

gmm = GaussianMixture(n_components=n_component, max_iter=1500)
gauss_mix = gmm.fit(X_train)

scores = gauss_mix.score_samples(X_train)

thresh = quantile(scores, .03)

index = where(scores <= thresh)
values = X_train[index]

anomaly_data = data_df.loc[index[0]].sort_values(['y con + alu'])
normal_data = data_df.loc[np.logical_not(data_df.index.isin(index[0]))].sort_values(['y con + alu'])

data_test_df = pd.read_csv('housesToRentProcessed_test_full.csv', index_col=0).reset_index()
data = data_test_df.values[:, 1:]
Y = data[:, -1]
score_test = gauss_mix.score_samples(data)

index_test = where(score_test <= thresh)[0]
values2 = data[index_test]
anomaly_detected_test = data_test_df.loc[index_test]

normal_data_test = data_test_df.loc[np.logical_not(data_test_df.index.isin(index_test))].sort_values(['y con + alu'])
















# n, bins, patches = plt.hist(x=a, bins=100, color='#0504aa',
#                             alpha=0.7, rwidth=0.85)
# plt.grid(axis='y', alpha=0.75)
# plt.xlabel('Log Probabilitiedrives')
# plt.ylabel('Frequency')
# plt.title('Histograma dos logs das probabilidades após retirada das anomalias')
# maxfreq = n.max()
# # Set a clean upper y-axis limit.
# plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
#


# plt.savefig('BIC.png')

