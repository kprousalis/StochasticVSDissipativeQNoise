import numpy as np
import matplotlib.pyplot as plt
from qutip import *

B_field = 3.0
Temp = 0.5
E_VS, J, k_B = 0.15, 0.05, 8.617e-5
A_DSF, A_Raman, A_Valley = 1e-4, 1e-3, 1.0
A_Anh, A_g, A_HF = 2e-3, 0.08, 0.005

gamma_1 = (A_DSF * B_field**5) + (A_Raman * Temp**2) + \
          (A_Valley * np.exp(-(E_VS*1e-3)/(k_B*Temp)))
gamma_phi = (A_Anh * Temp**2) + (A_g * B_field) + A_HF

T1_val = 1.0 / gamma_1 if gamma_1 > 0 else 1e9
Tphi_val = 1.0 / gamma_phi if gamma_phi > 0 else 1e9

sz1, sz2 = tensor(sigmaz(), qeye(2)), tensor(qeye(2), sigmaz())
sx1, sx2 = tensor(sigmax(), qeye(2)), tensor(qeye(2), sigmax())
sy1, sy2 = tensor(sigmay(), qeye(2)), tensor(qeye(2), sigmay())
sm1, sm2 = tensor(sigmam(), qeye(2)), tensor(qeye(2), sigmam())

state_11 = tensor(basis(2,0), basis(2,0))
P_11 = state_11 * state_11.dag()

E_Z = 28.0 * B_field
H0 = 0.5*E_Z*sz1 + 0.5*(E_Z+0.01)*sz2 + (J/4.0)*(sx1*sx2 + sy1*sy2 + sz1*sz2)

psi0_T1 = state_11
times_long = np.linspace(0, 5.0 * T1_val, 250)
dt_long = times_long[1] - times_long[0]

c_ops_T1 = [np.sqrt(gamma_1)*sm1, np.sqrt(gamma_1)*sm2]
res_lind_T1 = mesolve(H0, psi0_T1, times_long, c_ops_T1, [])

pop_lind = [expect(P_11, st) for st in res_lind_T1.states]

n_traj = 200
rho_avg_long = [tensor(qzero(2), qzero(2)) for _ in range(len(times_long))]
amp_x = np.sqrt(gamma_1 / (2*dt_long))

for traj in range(n_traj):
    psi = psi0_T1
    for i, t in enumerate(times_long):
        rho_avg_long[i] += psi * psi.dag()
        nx1, nx2 = np.random.normal(0,1)*amp_x, np.random.normal(0,1)*amp_x
        psi = ((-1j * (nx1*sx1 + nx2*sx2) * dt_long).expm() * psi).unit()

pop_stoch = [expect(P_11, rho_avg_long[i]/n_traj) for i in range(len(times_long))]

psi0_T2 = (tensor(basis(2,0), basis(2,1)) - tensor(basis(2,1), basis(2,0))).unit()
times_short = np.linspace(0, 3.0 * Tphi_val, 200)
dt_short = times_short[1] - times_short[0]

c_ops_phi = [np.sqrt(gamma_phi/2)*sz1, np.sqrt(gamma_phi/2)*sz2]
res_lind_T2 = mesolve(H0, psi0_T2, times_short, c_ops_phi, [])
coh_lind = [abs(st[1,2])*2.0 for st in res_lind_T2.states]

rho_avg_short = [tensor(qzero(2), qzero(2)) for _ in range(len(times_short))]
amp_z = np.sqrt(gamma_phi / (2*dt_short))

for traj in range(ntraj if 'ntraj' in locals() else n_traj):
    psi = psi0_T2
    for i, t in enumerate(times_short):
        rho_avg_short[i] += psi * psi.dag()
        nz1, nz2 = np.random.normal(0,1)*amp_z, np.random.normal(0,1)*amp_z
        H_step = H0 + (nz1*sz1 + nz2*sz2)
        psi = ((-1j * H_step * dt_short).expm() * psi).unit()

coh_stoch = [abs(rho_avg_short[i][1,2]/n_traj)*2.0 for i in range(len(times_short))]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(times_long, pop_lind, 'r-', linewidth=3, label='Quantum (Lindblad)')
ax1.plot(times_long, pop_stoch, 'b--', linewidth=2, label='Classical (Stochastic)')
ax1.axhline(0.25, color='gray', linestyle=':', label='Classical Limit (0.25)') # Αλλαγή σε 0.25
ax1.set_title(r'Relaxation ($T_1$)', fontsize=14)
ax1.set_xlabel('Time (ns)')
ax1.set_ylabel('Population of $|11\\rangle$')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(times_short, coh_lind, 'g-', linewidth=3, label='Quantum (Lindblad)')
ax2.plot(times_short, coh_stoch, 'k--', linewidth=2, label='Classical (Stochastic)')
ax2.set_title(r'Dephasing ($T_2^*$) - Success of Classical Noise', fontsize=14)
ax2.set_xlabel('Time (ns)')
ax2.set_ylabel('Coherence')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()