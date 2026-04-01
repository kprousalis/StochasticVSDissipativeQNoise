import numpy as np
import matplotlib.pyplot as plt
from qutip import *

B_field, Temp = 3.0, 0.5
E_VS, J, k_B = 0.15, 0.05, 8.617e-5
A_DSF, A_Raman, A_Valley = 1e-4, 1e-3, 1.0

gamma_1 = (A_DSF * B_field**5) + (A_Raman * Temp**2) + \
          (A_Valley * np.exp(-(E_VS*1e-3)/(k_B*Temp)))
T1_val = 1.0 / gamma_1

print(f"T1 Time: {T1_val:.1f} ns")

sz1, sz2 = tensor(sigmaz(), qeye(2)), tensor(qeye(2), sigmaz())
sx1, sx2 = tensor(sigmax(), qeye(2)), tensor(qeye(2), sigmax())
sy1, sy2 = tensor(sigmay(), qeye(2)), tensor(qeye(2), sigmay())
sm1, sm2 = tensor(sigmam(), qeye(2)), tensor(qeye(2), sigmam())

state_11 = tensor(basis(2,0), basis(2,0)) # Η αρχική κατάσταση Up-Up
psi0 = state_11

P_excited = state_11 * state_11.dag()

E_Z = 28.0 * B_field
H0 = 0.5*E_Z*sz1 + 0.5*(E_Z+0.01)*sz2 + (J/4.0)*(sx1*sx2 + sy1*sy2 + sz1*sz2)
times = np.linspace(0, 4 * T1_val, 300)

print("Running Lindblad...")
c_ops = [np.sqrt(gamma_1)*sm1, np.sqrt(gamma_1)*sm2]
res_lind = mesolve(H0, psi0, times, c_ops, [P_excited])
pop_lind = res_lind.expect[0]

print("Running Bloch-Redfield...")
def spec_T1(w): return gamma_1 if w > 0 else 0

a_ops = [[sx1, spec_T1], [sx2, spec_T1]]
res_br = brmesolve(H0, psi0, times, a_ops, [P_excited])
pop_br = res_br.expect[0]

print("Running Quantum Jumps...")
res_mc = mcsolve(H0, psi0, times, c_ops, [P_excited], ntraj=250)
pop_mc = res_mc.expect[0]

print("Running Classical Stochastic...")
n_traj_stoch = 200
dt = times[1] - times[0]
amp_x = np.sqrt(gamma_1 / (2*dt))
pop_stoch_avg = np.zeros(len(times))

for traj in range(n_traj_stoch):
    psi = psi0
    traj_vals = []
    for t in times:

        nx1, nx2 = np.random.normal(0,1)*amp_x, np.random.normal(0,1)*amp_x
        psi = ((-1j * (nx1*sx1 + nx2*sx2) * dt).expm() * psi).unit()
        traj_vals.append(expect(P_excited, psi))
    pop_stoch_avg += np.array(traj_vals)
pop_stoch = pop_stoch_avg / n_traj_stoch

plt.figure(figsize=(10, 7))

plt.plot(times, pop_lind, 'r-', lw=4, alpha=0.6, label='Lindblad')
plt.plot(times, pop_br, 'k--', lw=2, label='Bloch-Redfield')
plt.plot(times, pop_mc, 'g:', lw=3, label='Quantum Jumps')
plt.plot(times, pop_stoch, 'b--', lw=2, label='Classical Stochastic (Saturates at 0.25)')

plt.axhline(0.25, color='gray', linestyle=':', label='Classical Thermal Limit (0.25)')
plt.title(f'Comparison of 4 Simulation Methods for $T_1$ Decay from State $|11\\rangle$', fontsize=14)
plt.xlabel('Time (ns)', fontsize=12)
plt.ylabel('Population of $|11\\rangle$', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.show()