import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.linalg import pinv
from scipy.stats import multivariate_normal
from numpy.linalg import norm
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler

dataAll = pd.read_csv('housesToRentProcessed.csv')
valuesSaoPaulo = dataAll['São Paulo'].values
indexSaoPaulo = []
for i in range(dataAll.shape[0]):
    if valuesSaoPaulo[i] == 1:
        indexSaoPaulo.append(i)
data = dataAll.iloc[:, :-1]
y = dataAll.iloc[:, -1:]
data = data.values.astype(float)

# Normalizing
scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)

# Inicialize os parâmetros

# Mi
sums = data.sum(axis=0)
dividirPorN = lambda x: x/data.shape[0]
Mi = dividirPorN(sums)

# W       
D = data.shape[1]   
L = 3
N = data.shape[0]

np.random.seed(42)
W = np.random.normal(0,0.01, size=(D,L))
sigma2 = np.median(data.var(axis=0))

def passoE(W, sigma2):
    # Eq1
    M = W.T @ W + sigma2*np.identity(L)
    
    # Eq2
    Minv = pinv(M)
    MinvWT = Minv @ W.T
    Ezis = np.zeros((N, L))
    
    
    # Eq3
    EziziTs = np.zeros((N, L, L))
    s2Minv = sigma2*Minv
    
    for i, xi in enumerate(data):
        # Eq2
        Ezi = MinvWT @ (xi - Mi).reshape((D,1))
        Ezis[i] = Ezi.T

        # Eq3
        Ezi_x_EziT = Ezi @ Ezi.T  
        EziziT = s2Minv + Ezi_x_EziT
        EziziTs[i] = EziziT
    
    return M, Ezis, EziziTs

def passoM(M, Ezis, EziziTs):
    # Atualizando W
    
    # W segundo termo
    sumEziziT = EziziTs.sum(axis=0)
    B = pinv(sumEziziT)

    # W primeiro termo
    A = np.zeros((D,L))
    for i in range(N):
        xi = data[i]
        xiLessMi = np.array([xi - Mi]).T
        EziT = np.array([Ezis[i]])
        A += xiLessMi @ EziT
    W = A @ B
    
    # Atualizando sigma2
    sumSigma = 0
    for i in range(N):
        xi = data[i]
        xilessMi = xi - Mi
        termo1 = norm(xilessMi,2)**2
#         print(termo1)
        
        
        Ezi = np.array([Ezis[i]])
        termo2 = 2*(Ezi @ W.T @ xilessMi)[0]
#         print(termo2)

        EziziT = np.array([EziziTs[i]])
        aux = EziziT @ W.T @ W
        termo3 = np.trace(aux[0])
#         print(termo3)
        
        sumSigma += termo1 - termo2 + termo3
    sigma2 = sumSigma/(N*D)

    return W, sigma2

for i in tqdm(range(50)):
    M, Ezis, EziziTs = passoE(W, sigma2)
    W, sigma2 = passoM(M, Ezis, EziziTs)

ZisProjetadosSaoPaulo = []
ZisResto = []
for i in range(N):
    xi = np.array([data[i]])
    xiLessMi = xi - np.array([Mi])
    
    Minv = pinv(M)
    MinvWT = Minv @ W.T
    zi = MinvWT @ xiLessMi.T # Projeção
    
    if i in indexSaoPaulo:
        ZisProjetadosSaoPaulo.append([zi[0,0], zi[1,0], zi[2,0]])
    else:
        ZisResto.append([zi[0,0], zi[1,0], zi[2,0]])
    
ZisProjetadosSaoPaulo = np.array(ZisProjetadosSaoPaulo)
ZisResto = np.array(ZisResto)

pricesValue = y.values.T[0]
pricesSaoPaulo = []
pricesResto = []
for i in range(len(pricesValue)):
    if i in indexSaoPaulo:
        pricesSaoPaulo.append(pricesValue[i])
    else:
        pricesResto.append(pricesValue[i])

l1 = ZisProjetadosSaoPaulo[:,0]
l2 = ZisProjetadosSaoPaulo[:,1]
l3 = ZisProjetadosSaoPaulo[:,2]

l1r = ZisResto[:,0]
l2r = ZisResto[:,1]
l3r = ZisResto[:,2]

mpl.rcParams['figure.figsize'] = (20, 20)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

y1Log = np.log(pricesSaoPaulo)
y2Log = np.log(pricesResto)
ax.scatter(l1, l2, l3, marker='v', c=y1Log, cmap="inferno")
ax.scatter(l1r, l2r, l3r, marker='o', c=y2Log, cmap="inferno")

ax.set_xlabel('L1')
ax.set_ylabel('L2')
ax.set_zlabel('L3')
ax.set_xlim(-2.5,1.5)
ax.set_ylim(-4.5,2)
ax.set_zlim(-4,3)

plt.show()

# plt.scatter(l1, l2, c=pricesSaoPaulo, cmap="inferno", marker='v')
# plt.scatter(l1r, l2r, c=pricesResto, cmap="inferno", marker='o')

# plt.xlim(-2.5,1.5)
# plt.ylim(-4.5,2)
# plt.colorbar()
# plt.show()