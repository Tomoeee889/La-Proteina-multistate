import pandas as pd

csv_file = 'df_pdb_f0.005_minl200_maxl300_mtprotein_etdiffraction_minoNone_maxoNone_minr0.0_maxr5.0_hl_rl_rnsrTrue_rpuTrue_l_rcuFalse.csv'

df = pd.read_csv(csv_file)

print(f'До фильтрации: {len(df)}')
df = df[df["length"] < 220]
print(f'После фильтрации: {len(df)}')

df.to_csv(csv_file, index=False)
print(f'Файл сохранен в {csv_file}')