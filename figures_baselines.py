"""
Красивые графики по baseline экспериментам. Только PNG.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Стиль
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['font.family'] = 'DejaVu Sans'

# Загружаем данные
df = pd.read_csv('baselines/all_experiments_comparison.csv')

# Цвета
C_DEFAULT = '#2E86AB'
C_SIM_EPS = '#E63946'

# ============================================================
# График 1: Task 1 - Unconditional (RMSD + Seq Identity)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

task1 = df[df['task'] == 'Task1_unconditional']

for mode, color, marker in [('default', C_DEFAULT, 'o'), ('sim_eps', C_SIM_EPS, 's')]:
    sub = task1[task1['mode'] == mode].sort_values('noise_scale')
    
    # RMSD
    ax1.plot(sub['noise_scale'], sub['full_rmsd_mean'], 
             marker=marker, linewidth=2.5, markersize=9, color=color, label=mode)
    ax1.fill_between(sub['noise_scale'], 
                     sub['full_rmsd_mean'] - sub['full_rmsd_std'],
                     sub['full_rmsd_mean'] + sub['full_rmsd_std'],
                     alpha=0.15, color=color)
    
    # Seq Identity
    ax2.plot(sub['noise_scale'], sub['seq_identity_mean'], 
             marker=marker, linewidth=2.5, markersize=9, color=color, label=mode)
    ax2.fill_between(sub['noise_scale'], 
                     sub['seq_identity_mean'] - sub['seq_identity_std'],
                     sub['seq_identity_mean'] + sub['seq_identity_std'],
                     alpha=0.15, color=color)

ax1.set_xlabel('Noise Scale')
ax1.set_ylabel('Full RMSD (Å)')
ax1.set_title('Task 1: Unconditional — RMSD')
ax1.legend(title='Mode', loc='best')
ax1.set_ylim(0, 15)

ax2.set_xlabel('Noise Scale')
ax2.set_ylabel('Sequence Identity')
ax2.set_title('Task 1: Unconditional — Sequence')
ax2.legend(title='Mode', loc='best')
ax2.set_ylim(0.95, 1.01)

plt.tight_layout()
plt.savefig('baselines/plot_task1_unconditional.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ plot_task1_unconditional.png")

# ============================================================
# График 2: Task 2+3 - Conditional (Scaffold RMSD + Seq Identity)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

task23 = df[df['task'].isin(['Task2_conditional_small', 'Task3_conditional_large'])]

for mode, color, marker in [('default', C_DEFAULT, 'o'), ('sim_eps', C_SIM_EPS, 's')]:
    sub = task23[task23['mode'] == mode].sort_values('noise_scale')
    
    # Scaffold RMSD (log scale)
    ax1.semilogy(sub['noise_scale'], np.maximum(sub['scaffold_rmsd_mean'], 1e-3), 
                 marker=marker, linewidth=2.5, markersize=9, color=color, label=mode)
    ax1.fill_between(sub['noise_scale'], 
                     np.maximum(sub['scaffold_rmsd_mean'] - sub['scaffold_rmsd_std'], 1e-4),
                     sub['scaffold_rmsd_mean'] + sub['scaffold_rmsd_std'],
                     alpha=0.15, color=color)
    
    # Scaffold Seq Identity
    ax2.plot(sub['noise_scale'], sub['scaffold_seq_identity_mean'], 
             marker=marker, linewidth=2.5, markersize=9, color=color, label=mode)
    ax2.fill_between(sub['noise_scale'], 
                     sub['scaffold_seq_identity_mean'] - sub['scaffold_seq_identity_std'],
                     sub['scaffold_seq_identity_mean'] + sub['scaffold_seq_identity_std'],
                     alpha=0.15, color=color)

ax1.set_xlabel('Noise Scale')
ax1.set_ylabel('Scaffold RMSD (Å, log scale)')
ax1.set_title('Task 2+3: Conditional — Scaffold RMSD')
ax1.legend(title='Mode', loc='best')
ax1.axvline(x=1.5, color='gray', linestyle='--', alpha=0.4, label='Task 2 / Task 3 boundary')

ax2.set_xlabel('Noise Scale')
ax2.set_ylabel('Scaffold Sequence Identity')
ax2.set_title('Task 2+3: Conditional — Scaffold Sequence')
ax2.legend(title='Mode', loc='best')
ax2.set_ylim(0.98, 1.01)
ax2.axvline(x=1.5, color='gray', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('baselines/plot_task23_conditional.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ plot_task23_conditional.png")

# ============================================================
# График 3: Главный сравнительный — все задачи вместе
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Task 1 (sim_eps)
t1 = task1[task1['mode'] == 'sim_eps'].sort_values('noise_scale')
ax.plot(t1['noise_scale'], t1['full_rmsd_mean'], 
        marker='o', linewidth=3, markersize=10, color='#2E86AB',
        label='Task 1: Unconditional (full RMSD)')

# Task 2+3 scaffold (sim_eps)
t23 = task23[task23['mode'] == 'sim_eps'].sort_values('noise_scale')
ax.plot(t23['noise_scale'], t23['scaffold_rmsd_mean'], 
        marker='s', linewidth=3, markersize=10, color='#E63946',
        label='Task 2+3: Conditional (scaffold RMSD)')

# Motif RMSD (должен быть ~0)
ax.plot(t23['noise_scale'], t23['motif_rmsd_mean'], 
        marker='^', linewidth=2, markersize=8, color='#F18F01',
        linestyle='--', label='Task 2+3: Motif RMSD (fixed)')

ax.set_xlabel('Noise Scale', fontsize=15)
ax.set_ylabel('RMSD (Å)', fontsize=15)
ax.set_title('Sensitivity to Initial Conditions: All Tasks', fontsize=17, fontweight='bold')
ax.legend(loc='upper left', fontsize=11)
ax.set_ylim(0, 10)

# Аннотации
ax.annotate('High sensitivity\n(unconditional)', 
            xy=(1.0, 5.98), xytext=(0.5, 8.0),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=11, ha='center', fontweight='bold')

ax.annotate('Extreme robustness\n(conditional)', 
            xy=(1.0, 0.007), xytext=(0.3, 2.0),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=11, ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('baselines/plot_comparison_all.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ plot_comparison_all.png")

# ============================================================
# График 4: Структура vs Последовательность (главный парадокс)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

t23_sim = task23[task23['mode'] == 'sim_eps'].sort_values('noise_scale')

ax.plot(t23_sim['noise_scale'], t23_sim['scaffold_rmsd_mean'], 
        marker='o', linewidth=3, markersize=10, color='#E63946',
        label='Scaffold RMSD (structure)')

ax2 = ax.twinx()
ax2.plot(t23_sim['noise_scale'], t23_sim['scaffold_seq_identity_mean'], 
         marker='s', linewidth=3, markersize=10, color='#2E86AB',
         label='Scaffold Seq Identity (sequence)')

ax.set_xlabel('Noise Scale', fontsize=15)
ax.set_ylabel('Scaffold RMSD (Å)', fontsize=15, color='#E63946')
ax2.set_ylabel('Scaffold Sequence Identity', fontsize=15, color='#2E86AB')
ax.tick_params(axis='y', labelcolor='#E63946')
ax2.tick_params(axis='y', labelcolor='#2E86AB')
ax2.set_ylim(0.98, 1.01)

ax.set_title('The Paradox: Structures Diverge, Sequences Stay Identical', 
             fontsize=16, fontweight='bold')

# Легенды
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=11)

plt.tight_layout()
plt.savefig('baselines/plot_paradox_structure_vs_sequence.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ plot_paradox_structure_vs_sequence.png")

# ============================================================
# График 5: Default vs sim_eps — почему sim_eps обязателен
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Task 1
for mode, color, label in [('default', C_DEFAULT, 'default (SDE noise dominates)'), 
                            ('sim_eps', C_SIM_EPS, 'sim_eps (clean signal)')]:
    sub = task1[task1['mode'] == mode].sort_values('noise_scale')
    ax1.plot(sub['noise_scale'], sub['full_rmsd_mean'], 
             marker='o', linewidth=2.5, markersize=9, color=color, label=label)

ax1.set_xlabel('Noise Scale')
ax1.set_ylabel('Full RMSD (Å)')
ax1.set_title('Task 1: Why sim_eps is Critical')
ax1.legend(loc='best')
ax1.set_ylim(0, 15)

# Task 2+3
for mode, color, label in [('default', C_DEFAULT, 'default'), ('sim_eps', C_SIM_EPS, 'sim_eps')]:
    sub = task23[task23['mode'] == mode].sort_values('noise_scale')
    ax2.semilogy(sub['noise_scale'], np.maximum(sub['scaffold_rmsd_mean'], 1e-3), 
                 marker='o', linewidth=2.5, markersize=9, color=color, label=label)

ax2.set_xlabel('Noise Scale')
ax2.set_ylabel('Scaffold RMSD (Å, log scale)')
ax2.set_title('Task 2+3: Conditional — Default vs sim_eps')
ax2.legend(loc='best')

plt.tight_layout()
plt.savefig('baselines/plot_default_vs_sim_eps.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ plot_default_vs_sim_eps.png")

print("\nГотово! Все графики в baselines/*.png")