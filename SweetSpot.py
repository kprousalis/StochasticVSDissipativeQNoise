import numpy as np
import matplotlib.pyplot as plt

Temp = 0.15
E_VS = 0.15
k_B = 8.617e-5
A_DSF, A_Raman, A_Valley = 1e-4, 1e-3, 1.0
A_Anh, A_g, A_HF = 2e-3, 0.08, 0.005

B_values = np.logspace(np.log10(0.05), np.log10(5.0), 100)
T1_list = []
T2star_list = []

for B in B_values:
    gamma_1 = (A_DSF * B ** 5) + (A_Raman * Temp ** 2) + \
              (A_Valley * np.exp(-(E_VS * 1e-3) / (k_B * Temp)))
    gamma_phi = (A_Anh * Temp ** 2) + (A_g * B) + A_HF

    t1 = 1.0 / gamma_1 if gamma_1 > 0 else 1e9
    rate_total = gamma_phi + (gamma_1 / 2.0)
    t2_star = 1.0 / rate_total if rate_total > 0 else 1e9

    T1_list.append(t1)
    T2star_list.append(t2_star)

plt.figure(figsize=(10, 7))

plt.loglog(B_values, T1_list, 'r-', linewidth=3, label=r'$T_1$ (Relaxation)')
plt.loglog(B_values, T2star_list, 'b--', linewidth=3, label=r'$T_2^*$ (Coherence)')
plt.axvspan(0.1, 0.5, color='yellow', alpha=0.2, label='Sweet Spot Range')

plt.ylim(1e0, 1e7)

plt.axvline(x=0.1, color='gray', linestyle=':', linewidth=1.5)
plt.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5)

plt.text(0.1, 1.5, 'Min: 0.1 T', rotation=90, verticalalignment='bottom', horizontalalignment='right', fontsize=10,
         color='dimgray')
plt.text(0.5, 1.5, 'Max: 0.5 T', rotation=90, verticalalignment='bottom', horizontalalignment='right', fontsize=10,
         color='dimgray')

plt.text(0.22, 4e6, 'Optimal\nRange', horizontalalignment='center', fontsize=11, fontweight='bold', color='olive')

idx_t1 = -15
plt.annotate(r'$T_1 \propto B^{-5}$',
             xy=(B_values[idx_t1], T1_list[idx_t1]),
             xytext=(B_values[idx_t1] * 1.5, T1_list[idx_t1] * 15),
             arrowprops=dict(facecolor='red', shrink=0.05),
             fontsize=12, color='red', fontweight='bold')

idx_t2 = 70
plt.annotate(r'$T_2^* \propto B^{-1}$',
             xy=(B_values[idx_t2], T2star_list[idx_t2]),
             xytext=(B_values[idx_t2], T2star_list[idx_t2] * 5),
             arrowprops=dict(facecolor='blue', shrink=0.05),
             fontsize=12, color='blue', fontweight='bold')

plt.legend(loc='upper left', fontsize=11, framealpha=0.8)

plt.title('Optimization Analysis: Operation "Sweet Spot" (Si/SiGe)', fontsize=14)
plt.xlabel('Magnetic Field $B$ (Tesla)', fontsize=12)
plt.ylabel('Lifetime (ns)', fontsize=12)
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()