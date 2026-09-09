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

# manifold pipe
# ============================================================
# TUBULAÇÃO DO MANIFOLD
# Mais restritiva para elevar bastante P_man
# ============================================================

pipe_mani = Pipe(
    0.030 * 2,        # antes: 0.0595 * 2
    500,
    0,
    8.3022e6,
    984,
    4,
    5.752218216772682e6,
    3.903249155428134e7
)
#%% Defining the CasADi function for pumps and valves

f_ca = MX.sym('f', 1)
q_ca = MX.sym('q', 1)
alpha_ca = MX.sym('alpha', 1)

H_fun, eff_fun, pot_fun = bcs_functions(f_ca, q_ca)

head_fun = Function('head', [f_ca, q_ca], [64 * bcs_functions(f_ca, q_ca)[0]])
efficiency_fun = Function('efficiency', [f_ca, q_ca], [eff_fun])
power_fun = Function('power', [f_ca, q_ca], [pot_fun])

# Booster pump Head [m]
booster_fun = Function('booster', [f_ca, q_ca], [550 * (f_ca / 50) ** 2])

valve_fun = Function('choke', [alpha_ca], [choke_fun(alpha_ca)])

# Defining the BCS of the wells and booster pump

bcs1 = Pump(head_fun, efficiency_fun, power_fun)
bcs2 = Pump(head_fun, efficiency_fun, power_fun)

booster = Pump(booster_fun, efficiency_fun, power_fun)

# Defining the valves in the wells

choke1 = Choke(212.54 / (6e+4 * sqrt(1e+5)), valve_fun)
choke2 = Choke(212.54 / (6e+4 * sqrt(1e+5)), valve_fun)

# ============================================================
# TUBULAÇÕES - POÇO 1
# Mais produtivo / menor perda de carga
# ============================================================

pipe_sec1_1 = Pipe(
    0.090 * 2,        # diâmetro maior -> menor perda de carga
    2.80e3,           # comprimento menor
    1000 - 920,       # menor desnível
    1.5e9,
    984,
    0.3,
    5.752218216772682e6,
    3.903249155428134e7
)

pipe_sec2_1 = Pipe(
    0.065 * 2,        # diâmetro maior
    8.00e2,           # comprimento menor
    920 - 150,
    1.5e9,
    984,
    4,
    7.455247950618545e6,
    6.264914244217266e7
)


# ============================================================
# TUBULAÇÕES - POÇO 2
# Também mais produtivo, porém diferente do poço 1
# ============================================================

pipe_sec1_2 = Pipe(
    0.085 * 2,        # ligeiramente menor que o poço 1
    3.10e3,
    1050 - 920,
    1.5e9,
    984,
    0.3,
    5.752218216772682e6,
    3.903249155428134e7
)

pipe_sec2_2 = Pipe(
    0.062 * 2,
    9.00e2,
    920 - 170,
    1.5e9,
    984,
    4,
    7.455247950618545e6,
    6.264914244217266e7
)


# ============================================================
# BCS
# MESMA CURVA PARA OS DOIS POÇOS
# ============================================================

bcs1 = Pump(head_fun, efficiency_fun, power_fun)
bcs2 = Pump(head_fun, efficiency_fun, power_fun)


# ============================================================
# CHOKES
# EXATAMENTE IGUAIS
# ============================================================

choke1 = Choke(
    212.54 / (6e4 * sqrt(1e5)),
    valve_fun
)

choke2 = Choke(
    212.54 / (6e4 * sqrt(1e5)),
    valve_fun
)


# ============================================================
# RESERVATÓRIOS
# PI e PRES elevados para aumentar bastante a produção
# ============================================================

# Poço 1
# Poço 1
PI_1   = 2.0e-9
PRES_1 = 35.0e6       # 350 bar

# Poço 2
PI_2   = 2.5e-9
PRES_2 = 30.0e6       # 300 bar


# ============================================================
# POÇOS
# ============================================================

well1 = Well(
    pipe_sec1_1,
    pipe_sec2_1,
    bcs1,
    choke1,
    PI_1,
    PRES_1
)

well2 = Well(
    pipe_sec1_2,
    pipe_sec2_2,
    bcs2,
    choke2,
    PI_2,
    PRES_2
)


# ============================================================
# MANIFOLD
# ============================================================

mani = Manifold(
    pipe_mani,
    booster,
    0,
    0,
    [well1, well2]
)
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

# Defining the symbolic manifold model
mani_model = mani.model(0, x, z, u)

#%% Evaluation of steady-state
u0 = [45., 10 ** 5, 50., 0.8, 50., 0.8]

x0 = [76.52500, 4 * 85,
      64.11666, 120.91641, 85,
      64.11666, 120.91641, 85]

z0 = [30.03625, 239.95338-30.03625,
      30.03625, 239.95338-30.03625]

mani_solver = lambda y: array([float(i) for i in mani.model(0, y[0:-4], y[-4:], u0)])

y_ss = fsolve(mani_solver, x0+z0)

z_ss = y_ss[-4:]

x_ss = y_ss[0:-4]

x0 = [76.52500, 4 * 85,
      64.11666, 120.91641, 85,
      30.03625, 120.91641, 85]

z0 = [30.03625, 239.95338-30.03625,
      30.03625, 239.95338-30.03625]

#%% Dynamic Simulation
dae = {'x': vertcat(*x), 'z': vertcat(*z), 'p': vertcat(*u), 'ode': vertcat(*mani_model[0:-4]),
       'alg': vertcat(*mani_model[-4:])}

tfinal = 5000 # [s]

grid = linspace(0, tfinal, 100)

F = integrator('F', 'idas', dae, 0, grid)

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
    Inputs = [[] for _ in range(6)]
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
    Lista_zf_reshaped = Lista_zf.reshape(4, qtd_pts)
    Lista_xf_reshaped = Lista_xf.reshape(8, qtd_pts)

    # criando as pertubações de u0
    base_valve_open = 0.5
    base_bcs = 50.
    delta = 0.1
    delta_bcs = 5
    valve_open1 = np.clip(base_valve_open + np.random.uniform(-delta, delta, n_pert), 0, 1.0)
    valve_open2= valve_open1
    # valve_open2 = np.clip(base_valve_open + np.random.uniform(-delta, delta, n_pert), 0, 1.0)
    bcs_freq1 = np.clip(base_bcs + np.random.uniform(-delta_bcs, delta_bcs, n_pert), 35, 65)
    bcs_freq2 = bcs_freq1
    # bcs_freq2 = np.clip(base_bcs + np.random.uniform(-delta_bcs, delta_bcs, n_pert), 35, 65)
    booster_freq = np.clip(base_bcs + np.random.uniform(-delta_bcs, delta_bcs, n_pert), 35, 65)
    p_topo = np.random.uniform(8, 12, n_pert)



    for i in range(n_pert):
        u0 = [booster_freq[i], p_topo[i] * 10 ** 4, bcs_freq1[i], valve_open1[i], bcs_freq2[i], valve_open2[i]]
        for j in range(qtd_pts):
            for k in range(6):
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

    Auto_plot(Lista_zf_reshaped[[1, 3], :], "Pressão de Saída BCS", 'Time/(s)', 'Pressure/(bar)', 'b')
    Auto_plot(Lista_xf_reshaped[[2, 5], :], "Pressure de Fundo de Poço", 'Time/(s)', 'Pressure/(bar)', 'r')
    Auto_plot(Lista_xf_reshaped[[3, 6], :], 'Pressão da Choke', 'Time/(s)', 'Pressure/(bar)', 'g')
    Auto_plot(Lista_xf_reshaped[[4, 7], :], 'Vazão dos Poços', 'Time/(s)', 'Flow Rate/(m^3/h)', 'k')
    Auto_plot(Lista_xf_reshaped[[1], :], 'Vazão Manifold', 'Time/ s', r'Vazão Manifold / $m^3 \cdot s^{-1}$','b')
    Auto_plot(Lista_xf_reshaped[[0], :], 'Pressão Manifold', 'Time/ s', r'Pressão Manifold/ bar', 'm')
    Auto_plot(Lista_zf_reshaped[[0, 2], :],"Pressão de Entrada BCS", 'Time/(s)', 'Pressure/(bar)', 'c')
    return Lista_xf_reshaped, Lista_zf_reshaped, Inputs, grid


Lista_xf_reshaped, Lista_zf_reshaped, Inputs, grid = Sim_dynamics(500)