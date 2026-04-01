import numpy as np
import matplotlib.pyplot as plt
from qutip import *

B_field = 3.0
Temp = 0.5
E_VS = 0.15
J = 0.05
k_B = 8.617e-5
A_DSF, A_Ram, A_VO = 1e-4, 1e-3, 1.0
A_g, A_HF, A_Anh = 0.08, 0.005, 2e-3

gamma_DSF = A_DSF * (B_field**5)
gamma_Ram = A_Ram * (Temp**2)
gamma_VO  = A_VO * np.exp(-(E_VS*1e-3)/(k_B*Temp))
gamma_1   = gamma_DSF + gamma_Ram + gamma_VO

gamma_g   = A_g * B_field
gamma_HF  = A_HF
gamma_Anh = A_Anh * (Temp**2)
gamma_phi = gamma_g + gamma_HF + gamma_Anh

sz1, sz2 = tensor(sigmaz(), qeye(2)), tensor(qeye(2), sigmaz())
sx1, sx2 = tensor(sigmax(), qeye(2)), tensor(qeye(2), sigmax())
sy1, sy2 = tensor(sigmay(), qeye(2)), tensor(qeye(2), sigmay())
sm1, sm2 = tensor(sigmam(), qeye(2)), tensor(qeye(2), sigmam())

E_Z = 28.0 * B_field
H = 0.5*E_Z*sz1 + 0.5*(E_Z+0.01)*sz2 + (J/4.0)*(sx1*sx2 + sy1*sy2 + sz1*sz2)

times_T1 = np.linspace(0, 80, 500)
psi0_T1 = tensor(basis(2,0), basis(2,0))

P_excited = psi0_T1 * psi0_T1.dag()

c_ops_T1 = [np.sqrt(gamma_1)*sm1, np.sqrt(gamma_1)*sm2]
res_L_T1 = mesolve(H, psi0_T1, times_T1, c_ops=c_ops_T1)

def spec_T1(w): return gamma_1 if w > 0 else 0
a_ops_T1 = [[sx1, spec_T1], [sx2, spec_T1]]
res_BR_T1 = brmesolve(H, psi0_T1, times_T1, a_ops=a_ops_T1)

L_T1_curve = [expect(P_excited, rho) for rho in res_L_T1.states]
BR_T1_curve = [expect(P_excited, rho) for rho in res_BR_T1.states]

times_T2 = np.linspace(0, 15, 500)

psi0_T2 = (tensor(basis(2,0), basis(2,1)) - tensor(basis(2,1), basis(2,0))).unit()

c_ops_T2 = [np.sqrt(gamma_phi/2)*sz1, np.sqrt(gamma_phi/2)*sz2]
res_L_T2 = mesolve(H, psi0_T2, times_T2, c_ops=c_ops_T2)

def spec_T2(w): return gamma_phi / 2.0
a_ops_T2 = [[sz1, spec_T2], [sz2, spec_T2]]
res_BR_T2 = brmesolve(H, psi0_T2, times_T2, a_ops=a_ops_T2)

L_T2_curve = [abs(rho.full()[1, 2]) * 2.0 for rho in res_L_T2.states]
BR_T2_curve = [abs(rho.full()[1, 2]) * 2.0 for rho in res_BR_T2.states]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(times_T1, L_T1_curve, 'r-', lw=3, label='Lindblad (Markovian)')
ax1.plot(times_T1, BR_T1_curve, 'k--', lw=2, label='Bloch-Redfield')
ax1.set_title('Energy Relaxation ($T_1$)', fontsize=12)
ax1.set_xlabel('Time (ns)', fontsize=11)
ax1.set_ylabel('Population of $|11\\rangle$', fontsize=11)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(times_T2, L_T2_curve, 'b-', lw=3, label='Lindblad')
ax2.plot(times_T2, BR_T2_curve, 'k--', lw=2, label='Bloch-Redfield')
ax2.set_title('Coherence Decay ($T_2^*$)', fontsize=12)
ax2.set_xlabel('Time (ns)', fontsize=11)
ax2.set_ylabel('Coherence Amplitude', fontsize=11)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()