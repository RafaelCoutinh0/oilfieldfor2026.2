from scipy.io import loadmat
import matplotlib.pyplot as plt
from initialization_oil_production_bcs_head_revised import *

T, X, Z, U = Simdynamics_agendada(dt=1.0, tfinal=4550.0)

data = loadmat('dados_completos.mat')

def limpar(n):
    return data[n].flatten()

t = limpar('t')

# estacionários do modelo
pman_ss   = x_ss[0]
pfbhp1_ss = x_ss[2]
pfbhp2_ss = x_ss[5]

# sinais do MATLAB em valor absoluto
pm_mat = (limpar('pm') + 1) * pman_ss
p1_mat = (limpar('p1') + 1) * pfbhp1_ss
p2_mat = (limpar('p2') + 1) * pfbhp2_ss

# sinais simulados
pman_sim   = X[0, :]
pfbhp1_sim = X[2, :]
pfbhp2_sim = X[5, :]

# Plot 1
plt.figure(figsize=(12, 5))
plt.plot(t, pm_mat, label='Pman MATLAB', linewidth=2)
plt.plot(T, pman_sim, '--', label='Pman simulado', linewidth=2)
plt.xlabel('Tempo (s)')
plt.ylabel('Pressão')
plt.title('Comparação de Pman')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot 2
plt.figure(figsize=(12, 5))
plt.plot(t, p1_mat, label='Pfbhp1 MATLAB', linewidth=2)
plt.plot(T, pfbhp1_sim, '--', label='Pfbhp1 simulado', linewidth=2)
plt.xlabel('Tempo (s)')
plt.ylabel('Pressão')
plt.title('Comparação de Pfbhp1')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot 3
plt.figure(figsize=(12, 5))
plt.plot(t, p2_mat, label='Pfbhp2 MATLAB', linewidth=2)
plt.plot(T, pfbhp2_sim, '--', label='Pfbhp2 simulado', linewidth=2)
plt.xlabel('Tempo (s)')
plt.ylabel('Pressão')
plt.title('Comparação de Pfbhp2')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()