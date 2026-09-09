# -*- coding: utf-8 -*-

"""
File to simulate a manifold with 4 wells

Adapted from:
Rasmus (2011) - Automatic Start-up and Control of Artificially Lifted Wells

@authors: Rodrigo Lima Meira e Daniel Diniz Santana
"""
#%% Package import
from numpy import array
import matplotlib.pyplot as plt
import matplotlib.pyplot
from matplotlib import rcParams
from casadi import MX, interpolant, Function, sqrt, vertcat, integrator, jacobian, transpose
from bcs_models import *
from manifold import *
from numpy import linspace, array, eye, zeros, repeat, concatenate, delete, diag
from numpy.linalg import inv
from matplotlib.pyplot import plot, figure, title
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter
import random
from scipy.optimize import fsolve
from control import ss, tf, sample_system, forced_response
from scipy.signal import ss2tf
import numpy as np
#%% Creating functions of BCS, Choke and Pipes

def bcs_functions(f, q):
    """
    BCS Head, Efficiency and Power as function of frequency (f) and flow (q)
    :param f: pump frequency (Hz)
    :param q: flow [m^3/s]
    :return:
    H: head [m]
    eff: efficiency
    pot: power
    """

    f0 = 50
    q0 = q*(f0/f)
    H0 = -29.5615*(q0/0.0353)**4+25.3722*(q0/0.0353)**3-8.7944*(q0/0.0353)**2-8.5706*(q0/0.0353)+21.4278
    H = H0 * (f / f0) ** 2.
    eff = 1
    pot = 1
    return H, eff, pot


def choke_fun(alpha):
    """
    Valve characteristic function
    :param alpha: valve opening  (0 to 1)
    :return: choke characteristic
    """
    # Valve opening
    zc = [0, 13, 15, 17, 19, 22, 25, 29, 30, 32, 34, 36, 39, 41, 44,
          46, 49, 52, 55, 58, 61, 64, 67, 71, 75, 78, 82, 86, 91, 96, 100.01]

    # valve characteristic
    G = [0, 0.011052632, 0.024210526, 0.043157895, 0.067894737, 0.097894737,
         0.133157895, 0.173684211, 0.195789474, 0.219473684, 0.244736842,
         0.271052632, 0.298947368, 0.328421053, 0.358947368, 0.390526316,
         0.423684211, 0.458421053, 0.494210526, 0.531578947, 0.570526316,
         0.610526316, 0.651578947, 0.694210526, 0.738421053, 0.784210526,
         0.830526316, 0.878947368, 0.928421053, 0.979473684, 1]

    fun_choke = interpolant('choke', 'bspline', [zc], G)
    return fun_choke(alpha * 100)


# Pipes creation

# well pipe before BCS
pipe_sec1 = Pipe(0.081985330499706 * 2, 3.078838005940556e3, 1029.2 - 920, 1.5e+9, 984, 0.3, 5.752218216772682e+06,
                 3.903249155428134e+07)
# well pipe after BCS
pipe_sec2 = Pipe(0.0595 * 2, 9.222097306189842e+02, 920 - 126.5400, 1.5e9, 984, 4, 7.455247950618545e+06,
                 6.264914244217266e+07)
# manifold pipe
pipe_mani = Pipe(0.0595 * 2, 500, 0, 8.3022e+6, 984, 4, 5.752218216772682e+06, 3.903249155428134e+07)

#%% Defining the CasADi function for pumps and valves

f_ca = MX.sym('f', 1)
q_ca = MX.sym('q', 1)
alpha_ca = MX.sym('alpha', 1)

H_fun, eff_fun, pot_fun = bcs_functions(f_ca, q_ca)

head_fun = Function('head', [f_ca, q_ca], [64 * bcs_functions(f_ca, q_ca)[0]])
efficiency_fun = Function('efficiency', [f_ca, q_ca], [eff_fun])
power_fun = Function('power', [f_ca, q_ca], [pot_fun])

# Booster pump Head [m]
booster_fun = Function('booster', [f_ca, q_ca], [1.0963e3 * (f_ca / 50) ** 2])

valve_fun = Function('choke', [alpha_ca], [choke_fun(alpha_ca)])

# Defining the BCS of the wells and booster pump

bcs1 = Pump(head_fun, efficiency_fun, power_fun)
bcs2 = Pump(head_fun, efficiency_fun, power_fun)
bcs3 = Pump(head_fun, efficiency_fun, power_fun)
bcs4 = Pump(head_fun, efficiency_fun, power_fun)

booster = Pump(booster_fun, efficiency_fun, power_fun)

# Defining the valves in the wells
# k_choke: valve constant [m^3/s/Pa^0.5]
choke1 = Choke(212.54 / (6e+4 * sqrt(1e+5)), valve_fun)
choke2 = Choke(212.54 / (6e+4 * sqrt(1e+5)), valve_fun)
choke3 = Choke(212.54 / (6e+4 * sqrt(1e+5)), valve_fun)
choke4 = Choke(212.54 / (6e+4 * sqrt(1e+5)), valve_fun)

# Defining the wells and the manifold
#choke1 = [m^3/s/Pa^0.5]

well1 = Well(pipe_sec1, pipe_sec2, bcs1, choke1, 6.9651e-9, 9800000)
well2 = Well(pipe_sec1, pipe_sec2, bcs2, choke2, 6.9651e-9, 9800000)
well3 = Well(pipe_sec1, pipe_sec2, bcs3, choke3, 6.9651e-9, 9800000)
well4 = Well(pipe_sec1, pipe_sec2, bcs4, choke4, 6.9651e-9, 9800000)

mani = Manifold(pipe_mani, booster, 0, 0, [well1, well2, well3, well4])

#%% Defining the simulation variables
#time
t = MX.sym('t')
# Inputs
f_BP = MX.sym('f_BP')  # [Hz] Boost Pump frequency
p_topside = MX.sym('p_topside')  # [Hz] Boost Pump frequency
u = [f_BP, p_topside]

f_ESP_1 = MX.sym('f_ESP_1')  # [Hz] ESP frequency
alpha_1 = MX.sym('alpha_1')  # [%] Choke opening
u += [f_ESP_1, alpha_1]

f_ESP_2 = MX.sym('f_ESP_2')  # [Hz] ESP frequency
alpha_2 = MX.sym('alpha_2')  # [%] Choke opening
u += [f_ESP_2, alpha_2]

f_ESP_3 = MX.sym('f_ESP_3')  # [Hz] ESP frequency
alpha_3 = MX.sym('alpha_3')  # [%] Choke opening
u += [f_ESP_3, alpha_3]

f_ESP_4 = MX.sym('f_ESP_4')  # [Hz] ESP frequency
alpha_4 = MX.sym('alpha_4')  # [%] Choke opening
u += [f_ESP_4, alpha_4]

# States and algebraic variables
p_man = MX.sym('p_man')  # [bar] manifold pressure
q_tr = MX.sym('q_tr')  # [m^3/h] Flow through the transportation line
x = [p_man, q_tr] # states
z = [] # algebraic variables

# Well 1
P_fbhp_1 = MX.sym('P_fbhp_1')  # [bar] Pressure fbhp
P_choke_1 = MX.sym('P_choke_1')  # [bar] Pressure in chokes
q_mean_1 = MX.sym('q_mean_1')  # [m^3/h] Average flow in the wells
P_intake_1 = MX.sym('P_ìntake_1')  # [bar] Pressure intake in ESP's
dP_bcs_1 = MX.sym('dP_bcs_1')  # [bar] Pressure discharge in ESP's
x += [P_fbhp_1, P_choke_1, q_mean_1]
z += [P_intake_1, dP_bcs_1]

# Well 2
P_fbhp_2 = MX.sym('P_fbhp_2')  # [bar] Pressure fbhp in ESP's
P_choke_2 = MX.sym('P_choke_2')  # [bar] Pressure in chokes
q_mean_2 = MX.sym('q_mean_2')  # [m^3/h] Average flow in the wells
P_intake_2 = MX.sym('P_ìntake_2')  # [bar] Pressure intake in ESP's
dP_bcs_2 = MX.sym('dP_bcs_2')  # [bar] Pressure discharge in ESP's
x += [P_fbhp_2, P_choke_2, q_mean_2]
z += [P_intake_2, dP_bcs_2]

# Well 3
P_fbhp_3 = MX.sym('P_fbhp_3')  # [bar] Pressure fbhp in ESP's
P_choke_3 = MX.sym('P_choke_3')  # [bar] Pressure in chokes
q_mean_3 = MX.sym('q_mean_3')  # [m^3/h] Average flow in the wells
P_intake_3 = MX.sym('P_ìntake_3')  # [bar] Pressure intake in ESP's
dP_bcs_3 = MX.sym('dP_bcs_3')  # [bar] Pressure discharge in ESP's
x += [P_fbhp_3, P_choke_3, q_mean_3]
z += [P_intake_3, dP_bcs_3]

# Well 4
P_fbhp_4 = MX.sym('P_fbhp_4')  # [bar] Pressure fbhp in ESP's
P_choke_4 = MX.sym('P_choke_4')  # [bar] Pressure in chokes
q_mean_4 = MX.sym('q_mean_4')  # [m^3/h] Average flow in the wells
P_intake_4 = MX.sym('P_ìntake_4')  # [bar] Pressure intake in ESP's
dP_bcs_4 = MX.sym('dP_bcs_4')  # [bar] Pressure discharge in ESP's
x += [P_fbhp_4, P_choke_4, q_mean_4]
z += [P_intake_4, dP_bcs_4]

# Defining the symbolic manifold model
mani_model = mani.model(0, x, z, u)

# %% Evaluation of steady-state
u0 = [56., 10 ** 4, 50., .5, 50., .5, 50., .5, 50., .5]

x0 = [76.52500, 4 * 85,
      64.11666, 120.91641, 85,
      64.11666, 120.91641, 85,
      64.11666, 120.91641, 85,
      64.11666, 120.91641, 85]

z0 = [30.03625, 239.95338-30.03625,
      30.03625, 239.95338-30.03625,
      30.03625, 239.95338-30.03625,
      30.03625, 239.95338-30.03625]

mani_solver = lambda y: array([float(i) for i in mani.model(0, y[0:-8], y[-8:], u0)])

y_ss = fsolve(mani_solver, x0+z0)

z_ss = y_ss[-8:]

x_ss = y_ss[0:-8]

#%% Dynamic Simulation
dae = {'x': vertcat(*x), 'z': vertcat(*z), 'p': vertcat(*u), 'ode': vertcat(*mani_model[0:-8]),
       'alg': vertcat(*mani_model[-8:])}

tfinal = 1000 # [s]
qtd_pts  = 25
grid = linspace(0, tfinal, qtd_pts)

F = integrator('F', 'idas', dae, 0, grid)

res = F(x0 = x_ss, z0 = z_ss, p = u0)

#%% Novidades

def Sim_dynamics(n_pert, qtd_pts = 25, u0=u0,x0=x_ss, z0=z_ss):
    global res
    u0 = u0
    x0 = x0
    z0 = z0
    tfinal = 1000  # [s]
    grid = linspace(0, tfinal, qtd_pts)
    F = integrator('F', 'idas', dae, 0, grid)
    res = F(x0=x0, z0=z0, p=u0)

    grid = linspace(0, (n_pert + 1) * 1000, qtd_pts * (n_pert + 1))

    # Inicialização
    Inputs = [[] for _ in range(10)]
    for i in range(len(u0)):
        for j in range(qtd_pts):
            Inputs[i].append(u0[i])
    Lista_xf = []
    Lista_zf = []
    Lista_xf.append(res["xf"])
    Lista_zf.append(res["zf"])
    x0 = Lista_xf[-1][:, -1]
    z0 = Lista_zf[-1][:, -1]
    Lista_zf = np.array(Lista_zf)
    Lista_xf = np.array(Lista_xf)
    Lista_zf_reshaped = Lista_zf.reshape(8, qtd_pts)
    Lista_xf_reshaped = Lista_xf.reshape(14, qtd_pts)

    # criando as pertubações de u0
    base_valve_open = 0.5
    base_bcs = 50.
    delta = 0.1
    delta_bcs = 5
    valve_open1 = np.clip(base_valve_open + np.random.uniform(-delta, delta, n_pert), 0, 1.0)
    valve_open2 = np.clip(base_valve_open + np.random.uniform(-delta, delta, n_pert), 0, 1.0)
    valve_open3 = np.clip(base_valve_open + np.random.uniform(-delta, delta, n_pert), 0, 1.0)
    valve_open4 = np.clip(base_valve_open + np.random.uniform(-delta, delta, n_pert), 0, 1.0)
    bcs_freq1 = np.clip(base_bcs + np.random.uniform(-delta_bcs, delta_bcs, n_pert), 35, 65)
    bcs_freq2 = np.clip(base_bcs + np.random.uniform(-delta_bcs, delta_bcs, n_pert), 35, 65)
    bcs_freq3 = np.clip(base_bcs + np.random.uniform(-delta_bcs, delta_bcs, n_pert), 35, 65)
    bcs_freq4 = np.clip(base_bcs + np.random.uniform(-delta_bcs, delta_bcs, n_pert), 35, 65)
    booster_freq = np.clip(base_bcs + np.random.uniform(-delta_bcs, delta_bcs, n_pert), 35, 65)
    p_topo = np.random.uniform(8, 12, n_pert)



    for i in range(n_pert):
        u0 = [booster_freq[i], p_topo[i] ** 4, bcs_freq1[i], valve_open1[i], bcs_freq2[i], valve_open2[i], bcs_freq3[i], valve_open3[i], bcs_freq4[i], valve_open4[i]]
        for j in range(qtd_pts):
            for k in range(10):
                Inputs[k].append(u0[k])
        res = F(x0=x0, z0=z0, p=u0)
        x0 = res["xf"][:, -1]
        z0 = res["zf"][:, -1]
        Lista_xf_reshaped = np.hstack((Lista_xf_reshaped, np.array(res["xf"])))
        Lista_zf_reshaped = np.hstack((Lista_zf_reshaped, np.array(res["zf"])))
    #Plotted Graphs
    rcParams['axes.formatter.useoffset'] = False
    def Auto_plot(i, t, xl, yl, c):
        plt.plot(grid, i.transpose(), c)
        matplotlib.pyplot.title(t)
        matplotlib.pyplot.xlabel(xl)
        matplotlib.pyplot.ylabel(yl)
        conc = np.concatenate(i)
        y_min, y_max = np.min(conc), np.max(conc)
        plt.ylim([y_min - 0.1 * abs(y_max), y_max + 0.1 * abs(y_max)])
        plt.grid()
        plt.show()

    Auto_plot(Lista_zf_reshaped[[1, 3, 5, 7], :], "Pressão de Saída BCS", 'Time/(s)', 'Pressure/(bar)', 'b')
    Auto_plot(Lista_xf_reshaped[[2, 5, 8, 11], :], "Pressure de Fundo de Poço", 'Time/(s)', 'Pressure/(bar)', 'r')
    Auto_plot(Lista_xf_reshaped[[3, 6, 9, 12], :], 'Pressão da Choke', 'Time/(s)', 'Pressure/(bar)', 'g')
    Auto_plot(Lista_xf_reshaped[[4, 7, 10, 13], :], 'Vazão dos Poços', 'Time/(s)', 'Flow Rate/(m^3/h)', 'k')
    Auto_plot(Lista_xf_reshaped[[1], :], 'Vazão Manifold', 'Time/ s', 'Vazão Manifold / $m^3 \cdot s^{-1}$','b')
    Auto_plot(Lista_xf_reshaped[[0], :], 'Pressão Manifold', 'Time/ s', 'Pressão Manifold/ bar', 'm')
    Auto_plot(Lista_zf_reshaped[[0, 2, 4, 6], :],"Pressão de Entrada BCS", 'Time/(s)', 'Pressure/(bar)', 'c')
    return Lista_xf_reshaped, Lista_zf_reshaped, Inputs, grid


um_ss = u0.copy()

u_ss = um_ss.copy()

u_ss.pop(1)

# EDO
J_x = [[jacobian(edo, k) for k in x] for edo in mani.model(t, x, z, u)[0:-8]]

J_x = [Function('jacobian_in_x', [t, *x, *z, *u], J) for J in J_x]

J_x_ss = array([array(J(0, *x_ss, *z_ss, *u0)).reshape(14) for J in J_x])

J_u = [[jacobian(edo, k) for k in u] for edo in mani.model(t, x, z, u)[0:-8]]

J_u = [Function('jacobian_in_u', [t, *x, *z, *u], J) for J in J_u]

J_u_ss = array([array(J(0, *x_ss, *z_ss, *u0)).reshape(10) for J in J_u])

# Algebraic
J_g_z = [[jacobian(g, k) for k in z] for g in mani.model(t, x, z, u)[-8:]]
J_g_z_fun = [Function('jacobian_g_in_z', [t, *x, *z, *u], J) for J in J_g_z]
J_g_z_ss = array([array(J(0, *x_ss, *z_ss, *u0)).reshape(8) for J in J_g_z_fun])

J_g_x = [[jacobian(g, k) for k in x] for g in mani.model(t, x, z, u)[-8:]]
J_g_x_fun = [Function('jacobian_g_in_x', [t, *x, *z, *u], J) for J in J_g_x]
J_g_x_ss = array([array(J(0, *x_ss, *z_ss, *u0)).reshape(14) for J in J_g_x_fun])

# ==============================================================================
# TRECHO CORRIGIDO: initialization_oil_production_basic.py
# ==============================================================================

# Criação do sistema no espaço de estados (sem redução ainda)
sys = ss(J_x_ss, J_u_ss, eye(14, 14), zeros((14, 10)))

# --- DEFINIÇÃO DAS SAÍDAS DESEJADAS ---
# O usuário pediu as 14 variáveis de estado:
# 0: p_man
# 1: q_tr
# 2, 3, 4:   P_fbhp_1, P_choke_1, q_mean_1
# 5, 6, 7:   P_fbhp_2, P_choke_2, q_mean_2
# 8, 9, 10:  P_fbhp_3, P_choke_3, q_mean_3
# 11, 12, 13: P_fbhp_4, P_choke_4, q_mean_4
indices_saida = list(range(14))  # Cria lista [0, 1, 2, ..., 13]

# Seleciona os valores de regime permanente correspondentes (para normalização)
# concatenate((x_ss, z_ss)) cria o vetor completo de 22 variáveis (14 estados + 8 algébricas)
y_ss = [concatenate((x_ss, z_ss))[i] for i in indices_saida]

# Matrizes de Transformação (Normalização)
Ty = inv(diag(y_ss))  # Transformação de y
Tu = diag(u_ss)  # Transformação de u

# Matriz B (Entradas)
# Retira p_topside (índice 1) pois não é manipulada na linearização padrão
B_temp = delete(J_u_ss, 1, 1)
B = inv(diag(x_ss)).dot(B_temp.dot(Tu))

# Matriz C (Saídas)
# C_all contém a relação linearizada de TODAS as variáveis (Estados + Algébricas)
C_all = concatenate((eye(14), -inv(J_g_z_ss).dot(J_g_x_ss)))

# SELEÇÃO: Mantemos apenas as linhas definidas em 'indices_saida' (0 a 13)
# Isso descarta as variáveis algébricas (14 a 21)
C = C_all[indices_saida, :]

# Normaliza a matriz C
C = Ty.dot(C.dot(diag(x_ss)))

# Matriz A (Dinâmica)
A = inv(diag(x_ss)).dot(J_x_ss).dot(diag(x_ss))

# Geração da Função de Transferência (TF)
n_outputs = len(indices_saida)  # Deve ser 14
n_inputs = 9  # 10 entradas menos p_topside

sys_TF_data_num = []
sys_TF_data_den = []

# Inicializa as listas
for i in range(n_outputs):
    sys_TF_data_num.append([])
    sys_TF_data_den.append([])

# Calcula TFs elemento a elemento
for j in range(n_inputs):
    # input=j calcula a resposta de todas as saídas para a entrada j
    data_TF_i = ss2tf(A, B, C, zeros((n_outputs, n_inputs)), input=j)

    for i in range(n_outputs):
        sys_TF_data_num[i].append(data_TF_i[0][i, :].tolist())
        sys_TF_data_den[i].append(data_TF_i[1].tolist())

# Cria o objeto TF final
sys_tf = tf(sys_TF_data_num, sys_TF_data_den)

print(f"Modelo Linear gerado com sucesso!")
print(f"Número de Saídas: {n_outputs} (Índices 0 a 13)")
print(f"Número de Entradas: {n_inputs}")

Sim_dynamics(n_pert=2)