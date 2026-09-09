# -*- coding: utf-8 -*-

"""
File to simulate a manifold with 4 wells

Adapted from:
Rasmus (2011) - Automatic Start-up and Control of Artificially Lifted Wells

@authors: Rodrigo Lima Meira e Daniel Diniz Santana
"""

#%% Package import

from casadi import MX, interpolant, Function, sqrt, vertcat, integrator, jacobian
from bcs_models import *
from manifold import *
from numpy import linspace, array, eye, zeros, repeat, concatenate, delete, diag
from numpy.linalg import inv
from matplotlib.pyplot import plot, figure
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter
import matplotlib as mpl

# Desativa notação científica globalmente nos eixos dos gráficos
mpl.rcParams['axes.formatter.useoffset'] = False
mpl.rcParams['axes.formatter.limits'] = (-99, 99)

from scipy.optimize import fsolve
from control import ss, tf, sample_system, forced_response
from scipy.signal import ss2tf


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

choke1 = Choke(212.54 / (6e+4 * sqrt(1e+5)), valve_fun)
choke2 = Choke(212.54 / (6e+4 * sqrt(1e+5)), valve_fun)
choke3 = Choke(212.54 / (6e+4 * sqrt(1e+5)), valve_fun)
choke4 = Choke(212.54 / (6e+4 * sqrt(1e+5)), valve_fun)

# Defining the wells and the manifold

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
p_man = MX.sym('p_man')  # [Pa] manifold pressure
q_tr = MX.sym('q_tr')  # [m^3/s] Flow through the transportation line
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

#%% Evaluation of steady-state
u0 = [45., 10 ** 5, 50., 0.8, 50., 0.8, 50., 0.8, 50., 0.8]

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

x0 = [76.52500, 4 * 85,
      64.11666, 120.91641, 85,
      30.03625, 120.91641, 85,
      64.11666, 120.91641, 85,
      64.11666, 120.91641, 85]

z0 = [30.03625, 239.95338-30.03625,
      30.03625, 239.95338-30.03625,
      30.03625, 239.95338-30.03625,
      30.03625, 239.95338-30.03625]

#%% Dynamic Simulation
dae = {'x': vertcat(*x), 'z': vertcat(*z), 'p': vertcat(*u), 'ode': vertcat(*mani_model[0:-8]),
       'alg': vertcat(*mani_model[-8:])}

tfinal = 5000 # [s]

grid = linspace(0, tfinal, 100)

F = integrator('F', 'idas', dae, 0, grid)

#%% Linearization with Casadi and State-Space and Transfer Function

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

J_f_z = [[jacobian(edo, k) for k in z]
         for edo in mani.model(t, x, z, u)[0:-8]]

J_f_z_fun = [
    Function('jacobian_f_in_z',
             [t, *x, *z, *u], J)
    for J in J_f_z
]

J_f_z_ss = array([
    array(J(0, *x_ss, *z_ss, *u0)).reshape(8)
    for J in J_f_z_fun
])

J_g_u = [[jacobian(g, k) for k in u]
         for g in mani.model(t, x, z, u)[-8:]]

J_g_u_fun = [
    Function('jacobian_g_in_u',
             [t, *x, *z, *u], J)
    for J in J_g_u
]

J_g_u_ss = array([
    array(J(0, *x_ss, *z_ss, *u0)).reshape(10)
    for J in J_g_u_fun
])

sys = ss(J_x_ss, J_u_ss, eye(14, 14), zeros((14, 10)))

sys_TF_data_num = []
sys_TF_data_den = []

for i in range(9):
    sys_TF_data_num.append([])
    sys_TF_data_den.append([])

y_ss = [concatenate((x_ss,z_ss))[i] for i in [0,3,6,9,12,14,15,16,17,18,19,20,21]] # pman, pfbhp_i, p_intake_i, dP_bcs_i

Ty = inv(diag(y_ss))   # tabela de transformação de y  

Tu = diag(u_ss)  # tabela de transformação de y

J_u_ss_red = delete(J_u_ss, 1, 1)
J_g_u_ss_red = delete(J_g_u_ss, 1, 1)


Bred = (
    J_u_ss_red
    -
    J_f_z_ss.dot(
        inv(J_g_z_ss)
    ).dot(J_g_u_ss_red)
)

B = inv(diag(x_ss)).dot(Bred).dot(Tu)

C_all = concatenate((
    eye(14),
    -inv(J_g_z_ss).dot(J_g_x_ss)
))

C = delete(C_all,[1,2,4,5,7,8,10,11,13],0)

C = Ty.dot(C.dot(diag(x_ss)))

Ared = (
    J_x_ss
    -
    J_f_z_ss.dot(
        inv(J_g_z_ss)
    ).dot(J_g_x_ss)
)

A = inv(diag(x_ss)).dot(Ared).dot(diag(x_ss))

for j in range(9):
    data_TF_i = ss2tf(A, B, C, zeros((13, 9)), input=j)
    for i in range(9):
        sys_TF_data_num[i].append(data_TF_i[0][i, :].tolist())
        sys_TF_data_den[i].append(data_TF_i[1].tolist())

sys_tf = tf(sys_TF_data_num, sys_TF_data_den)

sys_measured = ss(A, B, C, zeros((13, 9)))

#%% Subsistemas modelos lineares
# Subssitemas
# x = [P_fbhp_i, P_choke_i, q_average_i]
# u = [f_ESP, alpha]
# y = [P_choke_i, P_intake_i, dp_i]

sys_measured_well1 = ss(A[2:5,2:5], B[2:5,1:3], C[[1,5,6],2:5], zeros((3, 2)))
sys_measured_well2 = ss(A[5:8,5:8], B[5:8,3:5], C[[2,7,8],5:8], zeros((3, 2)))
sys_measured_well3 = ss(A[8:11,8:11], B[8:11,5:7], C[[3,9,10],8:11], zeros((3, 2)))
sys_measured_well4 = ss(A[11:14,11:14], B[11:14,7:9], C[[4,11,12],11:14], zeros((3, 2)))

sys_measured_mani = ss(A[0:2,0:2], B[0:2,0], C[0,0:2], zeros((1, 1)))

objective_sym =  (3000 * x[1]) - (((9653.04 * (x[1]/3600) * (1.0963e3 * (u[0]/ 50) ** 2) * 0.001) + \
        ((x[4]/3600) * z[1] * 1e2) + \
        ((x[7]/3600) * z[3] * 1e2) + \
        ((x[10]/3600) * z[5] * 1e2)  + \
        ((x[13]/3600) * z[7] * 1e2)) * 0.91)

import matplotlib.pyplot as plt
import matplotlib.pyplot
from matplotlib import rcParams
import numpy as np

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

    # Auto_plot(Lista_zf_reshaped[[1, 3, 5, 7], :], "Pressão de Saída BCS", 'Time/(s)', 'Pressure/(bar)', 'b')
    # Auto_plot(Lista_xf_reshaped[[2, 5, 8, 11], :], "Pressure de Fundo de Poço", 'Time/(s)', 'Pressure/(bar)', 'r')
    # Auto_plot(Lista_xf_reshaped[[3, 6, 9, 12], :], 'Pressão da Choke', 'Time/(s)', 'Pressure/(bar)', 'g')
    # Auto_plot(Lista_xf_reshaped[[4, 7, 10, 13], :], 'Vazão dos Poços', 'Time/(s)', 'Flow Rate/(m^3/h)', 'k')
    # Auto_plot(Lista_xf_reshaped[[1], :], 'Vazão Manifold', 'Time/ s', 'Vazão Manifold / $m^3 \cdot s^{-1}$','b')
    # Auto_plot(Lista_xf_reshaped[[0], :], 'Pressão Manifold', 'Time/ s', 'Pressão Manifold/ bar', 'm')
    # Auto_plot(Lista_zf_reshaped[[0, 2, 4, 6], :],"Pressão de Entrada BCS", 'Time/(s)', 'Pressure/(bar)', 'c')
    return Lista_xf_reshaped, Lista_zf_reshaped, Inputs, grid


def Compare_linear_vs_nonlinear(
    estado=0,
    entrada=0,
    du=0.05,
    tempo=1000,
    qtd_pts=200
):

    # -----------------------------------------
    # nomes
    # -----------------------------------------
    xsym = [
        'pman','qtr',
        'Pfbhp1','Pchoke1','qmean1',
        'Pfbhp2','Pchoke2','qmean2',
        'Pfbhp3','Pchoke3','qmean3',
        'Pfbhp4','Pchoke4','qmean4'
    ]

    usym = [
        'fBP','ptopside',
        'fESP1','alpha1',
        'fESP2','alpha2',
        'fESP3','alpha3',
        'fESP4','alpha4'
    ]

    # -----------------------------------------
    # tempo
    # -----------------------------------------
    T = np.linspace(0, tempo, qtd_pts)

    # -----------------------------------------
    # entrada do modelo linear
    # -----------------------------------------
    U = np.zeros((10, qtd_pts))

    # degrau
    U[entrada, :] = du

    # -----------------------------------------
    # resposta linear
    # -----------------------------------------
    tout_lin, y_lin = forced_response(sys, T, U)

    y_linear = np.asarray(y_lin[estado]).squeeze()

    # adiciona ponto de operação
    y_linear_abs = x_ss[estado] * (1 + y_linear)

    # -----------------------------------------
    # entrada do modelo não linear
    # -----------------------------------------
    u_nl = np.array(u0, dtype=float).copy()

    # degrau percentual
    u_nl[entrada] += du * u0[entrada]
    print(u_nl[entrada])
    # integrador
    F_nl = integrator(
        'F_nl',
        'idas',
        dae,
        0,
        T
    )

    # simulação não linear
    sol = F_nl(
        x0=x_ss,
        z0=z_ss,
        p=u_nl
    )

    X_nl = np.array(sol["xf"])

    y_nonlinear = X_nl[estado, :]

    # -----------------------------------------
    # plot
    # -----------------------------------------
    plt.figure(figsize=(10,6))

    plt.plot(
        T,
        y_linear_abs,
        label='Linear',
        linewidth=2
    )

    plt.plot(
        T,
        y_nonlinear,
        '--',
        label='Não linear',
        linewidth=2
    )

    plt.xlabel('Tempo [s]')
    plt.ylabel(xsym[estado])

    plt.title(
        f'Comparação: {usym[entrada]} → {xsym[estado]}'
    )

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.show()

import numpy as np
from casadi import integrator

def entrada_agendada(t, u_ss):
    u = np.array(u_ss, dtype=float).copy()

    # u = [fBP, ptopside, fESP1, alpha1, fESP2, alpha2, fESP3, alpha3, fESP4, alpha4]

    if 50 <= t < 800:
        u[0] = 0.95 * u_ss[0]   # f_booster -10%
    elif 800 <= t < 1550:
        u[0] = 1.05 * u_ss[0]   # f_booster +10%

    elif 1550 <= t < 2300:
        u[2] = 1.05 * u_ss[2]   # f_bcs1 +10%

    elif 2300 <= t < 3050:
        u[2] = 0.95 * u_ss[2]   # f_bcs1 -10%

    elif 3050 <= t < 3800:
        u[4] = 0.95 * u_ss[4]   # f_bcs2 -20%

    elif 3800 <= t <= 4550:
        u[4] = 1.05 * u_ss[4]   # f_bcs2 +20%

    return u


def Simdynamics_agendada(dt=1.0, tfinal=4550.0, u_ini=None, x_ini=None, z_ini=None):
    if u_ini is None:
        u_ini = np.array(u0, dtype=float).copy()
    if x_ini is None:
        x_ini = np.array(x_ss, dtype=float).copy()
    if z_ini is None:
        z_ini = np.array(z_ss, dtype=float).copy()

    tempos_troca = [0, 50, 800, 1550, 2300, 3050, 3800, 4550]

    T_hist = []
    U_hist = []
    X_hist = []
    Z_hist = []

    xk = x_ini.copy()
    zk = z_ini.copy()

    for i in range(len(tempos_troca) - 1):
        t0 = tempos_troca[i]
        t1 = tempos_troca[i + 1]
        dur = t1 - t0

        uk = entrada_agendada(t0, u_ini)

        npts = int(dur / dt) + 1
        grid_seg = np.linspace(0, dur, npts)

        Fseg = integrator(f'Fseg_{i}', 'idas', dae, 0, grid_seg)
        res = Fseg(x0=xk, z0=zk, p=uk)

        Xseg = np.array(res['xf'])
        Zseg = np.array(res['zf'])
        Tseg = np.linspace(t0, t1, npts)
        Useg = np.tile(uk.reshape(-1, 1), (1, npts))

        if i > 0:
            Xseg = Xseg[:, 1:]
            Zseg = Zseg[:, 1:]
            Tseg = Tseg[1:]
            Useg = Useg[:, 1:]

        T_hist.append(Tseg)
        X_hist.append(Xseg)
        Z_hist.append(Zseg)
        U_hist.append(Useg)

        xk = Xseg[:, -1]
        zk = Zseg[:, -1]

    T_hist = np.concatenate(T_hist, axis=0)
    X_hist = np.hstack(X_hist)
    Z_hist = np.hstack(Z_hist)
    U_hist = np.hstack(U_hist)

    return T_hist, X_hist, Z_hist, U_hist