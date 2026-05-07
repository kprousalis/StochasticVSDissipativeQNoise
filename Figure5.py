import numpy as np
import matplotlib.pyplot as plt
from qutip import *

B_field, Temp = 3.0, 0.5
E_VS, J, k_B = 0.15, 0.05, 8.617e-5
A_DSF, A_Raman, A_Valley = 1e-4, 1e-3, 1.0

gamma_1 = (A_DSF * B_field**5) + (A_Raman * Temp**2) + \
          (A_Valley * np.exp(-(E_VS*1e-3)/(k_B*Temp)))
T1_val = 1.0 / gamma_1

N_qubits = 4
sz_list = [tensor([sigmaz() if j == i else qeye(2) for j in range(N_qubits)]) for i in range(N_qubits)]
sx_list = [tensor([sigmax() if j == i else qeye(2) for j in range(N_qubits)]) for i in range(N_qubits)]
sy_list = [tensor([sigmay() if j == i else qeye(2) for j in range(N_qubits)]) for i in range(N_qubits)]
sm_list = [tensor([sigmam() if j == i else qeye(2) for j in range(N_qubits)]) for i in range(N_qubits)]

state_all_up = tensor([basis(2,0)] * N_qubits)
P_excited = state_all_up * state_all_up.dag()

E_Z = 28.0 * B_field
H0 = 0
for i in range(N_qubits):
    H0 += 0.5 * (E_Z + i*0.01) * sz_list[i]

for i in range(N_qubits - 1):
    H0 += (J/4.0) * (sx_list[i]*sx_list[i+1] + sy_list[i]*sy_list[i+1] + sz_list[i]*sz_list[i+1])

times = np.linspace(0, 5 * T1_val, 200)
dt = times[1] - times[0]

def get_bootstrap_stats(all_trajectories, iterations=1000):
    n_traj = all_trajectories.shape[0]
    bootstrap_means = [np.mean(all_trajectories[np.random.randint(0, n_traj, n_traj)], axis=0) for _ in range(iterations)]
    return np.mean(all_trajectories, axis=0), np.std(bootstrap_means, axis=0)

print("Running 4-qubit Lindblad (Benchmark)...")
c_ops = [np.sqrt(gamma_1) * sm for sm in sm_list]
res_lind = mesolve(H0, state_all_up, times, c_ops, [P_excited])
pop_lind = res_lind.expect[0]

print("Running 4-qubit Classical Stochastic...")
n_traj_classic = 100
all_classic_traj = np.zeros((n_traj_classic, len(times)))
amp_x = np.sqrt(gamma_1 / (2*dt))

for traj in range(n_traj_classic):
    psi = state_all_up
    for i, t in enumerate(times):
        all_classic_traj[traj, i] = expect(P_excited, psi)
        noise_op = sum([np.random.normal(0,1)*amp_x * sx_list[k] for k in range(N_qubits)])
        psi = ((-1j * noise_op * dt).expm() * psi).unit()

pop_classic_mean, pop_classic_err = get_bootstrap_stats(all_classic_traj)

print("Running 4-qubit Quantum Jumps...")
n_traj_mc = 200
res_mc = mcsolve(H0, state_all_up, times, c_ops, [P_excited], ntraj=n_traj_mc)
pop_mc_mean = res_mc.expect[0]
pop_mc_err = np.sqrt(pop_mc_mean * (1 - pop_mc_mean) / n_traj_mc)

plt.figure(figsize=(10, 7))
plt.plot(times, pop_lind, 'r-', linewidth=4, alpha=0.6, label='Lindblad (4-qubit)')

plt.plot(times, pop_classic_mean, 'b--', linewidth=2, label='Classical Stochastic (n=100)')
plt.fill_between(times, pop_classic_mean - 2*pop_classic_err, pop_classic_mean + 2*pop_classic_err,
                 color='blue', alpha=0.15, label='Classical 95% Conf. Int.')

plt.plot(times, pop_mc_mean, 'g:', linewidth=3, label='Quantum Jumps (n=200)')
plt.fill_between(times, pop_mc_mean - 2*pop_mc_err, pop_mc_mean + 2*pop_mc_err,
                 color='green', alpha=0.2, label='Quantum Jumps 95% Conf. Int.')

plt.axhline(0.0625, color='gray', linestyle='--', label='Classical Thermal Limit (0.0625)')

plt.title(f'Scaling to 4-Qubit Register: $T_1$ Decay Validation\n(B={B_field}T, Hilbert Space Dim=16)', fontsize=14)
plt.xlabel('Time (ns)', fontsize=12)
plt.ylabel('Population of $|1111\\rangle$', fontsize=12)
plt.ylim(-0.05, 1.05)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()