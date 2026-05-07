import numpy as np
import matplotlib.pyplot as plt
from qutip import *

B_field, Temp = 3.0, 0.5
E_VS, J, k_B = 0.15, 0.05, 8.617e-5
A_Anh, A_g, A_HF = 2e-3, 0.08, 0.005

gamma_phi = (A_Anh * Temp ** 2) + (A_g * B_field) + A_HF
Tphi_val = 1.0 / gamma_phi

print(f"--- T2* Dephasing Simulation (Unified) ---")
print(f"B={B_field}T, T={Temp}K | Expected T2*: {Tphi_val:.2f} ns\n")

sz1, sz2 = tensor(sigmaz(), qeye(2)), tensor(qeye(2), sigmaz())
sx1, sx2 = tensor(sigmax(), qeye(2)), tensor(qeye(2), sigmax())
sy1, sy2 = tensor(sigmay(), qeye(2)), tensor(qeye(2), sigmay())

state_01 = tensor(basis(2, 0), basis(2, 1))
state_10 = tensor(basis(2, 1), basis(2, 0))
psi0 = (state_01 - state_10).unit()
coh_op = state_01 * state_10.dag()

E_Z = 28.0 * B_field
H0 = 0.5 * E_Z * sz1 + 0.5 * (E_Z + 0.01) * sz2 + (J / 4.0) * (sx1 * sx2 + sy1 * sy2 + sz1 * sz2)

times = np.linspace(0, 3.5 * Tphi_val, 300)
dt = times[1] - times[0]

def get_bootstrap_coherence(all_complex_traj, iterations=1000):
    n_traj = all_complex_traj.shape[0]
    bootstrap_means = []
    for _ in range(iterations):
        idx = np.random.randint(0, n_traj, size=n_traj)
        sample = all_complex_traj[idx]
        avg_complex = np.mean(sample, axis=0)
        bootstrap_means.append(np.abs(avg_complex) * 2.0)
    return np.abs(np.mean(all_complex_traj, axis=0)) * 2.0, np.std(bootstrap_means, axis=0)

print("Running Lindblad...")
c_ops = [np.sqrt(gamma_phi / 2) * sz1, np.sqrt(gamma_phi / 2) * sz2]
res_lind = mesolve(H0, psi0, times, c_ops, [coh_op])
pop_lind = np.abs(res_lind.expect[0]) * 2.0

print("Running Bloch-Redfield...")
def spec_phi(w): return gamma_phi / 2.0
res_br = brmesolve(H0, psi0, times, a_ops=[[sz1, spec_phi], [sz2, spec_phi]], e_ops=[coh_op])
pop_br = np.abs(res_br.expect[0]) * 2.0

print("Running Quantum Jumps...")
n_traj_mc = 1000
res_mc = mcsolve(H0, psi0, times, c_ops, [coh_op], ntraj=n_traj_mc)
pop_mc_mean = np.abs(res_mc.expect[0]) * 2.0
pop_mc_err = np.sqrt(pop_mc_mean * (2.0 - pop_mc_mean) / n_traj_mc)

print("Running Classical Stochastic (250 trajectories)...")
n_traj_stoch = 250
all_stoch_traj = np.zeros((n_traj_stoch, len(times)), dtype=complex)
amp_z = np.sqrt(gamma_phi / (2 * dt))

for traj in range(n_traj_stoch):
    psi_t = psi0
    for i, t in enumerate(times):
        all_stoch_traj[traj, i] = expect(coh_op, psi_t)
        nz1 = np.random.normal(0, 1) * amp_z
        nz2 = np.random.normal(0, 1) * amp_z
        H_noise = H0 + nz1 * sz1 + nz2 * sz2
        psi_t = ((-1j * H_noise * dt).expm() * psi_t).unit()

pop_stoch_mean, pop_stoch_err = get_bootstrap_coherence(all_stoch_traj)

plt.figure(figsize=(11, 8))

plt.plot(times, pop_lind, color='red', linestyle='-', lw=4, alpha=0.5, label='Lindblad')
plt.plot(times, pop_br, color='black', linestyle='--', lw=2, label='Bloch-Redfield')

plt.plot(times, pop_mc_mean, color='green', linestyle=':', lw=3, label='Quantum Jumps (n=1000)')
plt.fill_between(times, pop_mc_mean - 2*pop_mc_err, pop_mc_mean + 2*pop_mc_err,
                 color='green', alpha=0.2, label='Quantum Jumps 95% Conf. Int.')

plt.plot(times, pop_stoch_mean, color='blue', linestyle='--', lw=1.5, label='Classical Stochastic (n=250)')
plt.fill_between(times, pop_stoch_mean - 2*pop_stoch_err, pop_stoch_mean + 2*pop_stoch_err,
                 color='blue', alpha=0.15, label='Classical 95% Conf. Int.')

plt.title(r'Comparison of 4 Simulation Methods for $T_2^*$ Dephasing' + '\n' + r'(B=3.0T, T=0.5K)', fontsize=14)
plt.xlabel('Time (ns)', fontsize=12)
plt.ylabel(r'Coherence Amplitude', fontsize=12)
plt.ylim(-0.05, 1.05)
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, which='both', linestyle='-', alpha=0.2)

plt.tight_layout()
plt.show()