#!/usr/bin/env python3
"""Filtered BioEmu evaluation. Run from project root; see --help.

State membership: CA RMSD <= radius; samples in BOTH neighborhoods are
ambiguous, never assigned arbitrarily. Two-state support is undefined for
targets separated by less than --distinct. It is NOT a bimodality test.
All metrics use one-sequence ensembles; best2 selects between seq_A/seq_B.
"""
import argparse
import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CORE = re.compile(r'noise_[0-9.]+_job_\d+_n_\d+_id_\d+')
REGIME = re.compile(r'task\d+_(?:default|sim_eps)')
SOURCE = re.compile(r'path([AB])(?:[_-]|$)', re.I)
COLORS = {'default': '#2878B5', 'sim_eps': '#E87524'}


def pdb_ca(path):
    xyz, keys = [], []
    seen = set()
    for line in path.read_text().splitlines():
        if line.startswith('ENDMDL'):
            break
        if not line.startswith('ATOM') or line[12:16].strip() != 'CA':
            continue
        if line[16:17] not in (' ', 'A'):
            continue
        key = (line[21:22], line[22:27])
        if key in seen:
            continue
        seen.add(key)
        xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        keys.append(key)
    x = np.asarray(xyz, dtype=float)
    if len(x) < 3 or not np.isfinite(x).all():
        raise ValueError('PDB has fewer than three CA atoms or nonfinite coordinates')
    if len({k[0] for k in keys}) != 1:
        raise ValueError('Only single-chain baselines are supported')
    return x, keys


def fit(x, y, indices=None):
    """Row-vector Kabsch rotation, fitted on optional indices; no atom rejection."""
    p, q = (x, y) if indices is None else (x[indices], y[indices])
    cx, cy = p.mean(0), q.mean(0)
    u, _, vt = np.linalg.svd((p-cx).T @ (q-cy))
    d = np.eye(3)
    d[-1, -1] = np.linalg.det(u @ vt)
    return (x-cx) @ (u @ d @ vt) + cy


def rmsd(x, y):
    return float(np.sqrt(np.mean(np.sum((fit(x, y)-y)**2, axis=1))))


def discover_pairs(base, scaffold, diagnostics):
    pairs = {}
    for directory in sorted(base.glob('task*/*')):
        if not directory.is_dir() or not REGIME.fullmatch(directory.parent.name) or not CORE.fullmatch(directory.name):
            continue
        regime, core = directory.parent.name, directory.name
        try:
            paths = {}
            for source in 'AB':
                path = directory / f'{core}_path{source}.pdb'
                if not path.exists():
                    path = directory / f'path{source}.pdb'
                paths[source] = path
            a, ka = pdb_ca(paths['A'])
            b, kb = pdb_ca(paths['B'])
            if ka != kb or a.shape != b.shape:
                raise ValueError('A/B residue numbering or CA count mismatch')
            pair = dict(regime=regime, task=regime.split('_')[0], mode=regime.split('_', 1)[1],
                        core=core, noise=float(core.split('_')[1]), length=len(a),
                        backbone_rmsd_AB=rmsd(a, b), A=a, B=b,
                        pdb_A=str(paths['A']), pdb_B=str(paths['B']))
            if scaffold and pair['task'] in ('task2', 'task3'):
                start, stop = scaffold
                if not 1 <= start <= stop <= len(a):
                    raise ValueError('Scaffold range exceeds chain length')
                free = np.arange(start-1, stop)
                motif = np.setdiff1d(np.arange(len(a)), free)
                if len(motif) < 3 or len(free) < 3:
                    raise ValueError('At least three CA atoms required in scaffold and motif')
                aligned = fit(b, a, motif)
                pair['scaffold_shape_rmsd_AB'] = rmsd(b[free], a[free])
                pair['scaffold_motion_after_motif_fit_AB'] = float(np.sqrt(np.mean(np.sum((aligned[free]-a[free])**2, axis=1))))
                pair['motif_fit_rmsd_AB'] = float(np.sqrt(np.mean(np.sum((aligned[motif]-a[motif])**2, axis=1))))
            pairs[(regime, core)] = pair
            diagnostics.append(dict(regime=regime, core=core, status='OK'))
        except Exception as exc:
            diagnostics.append(dict(regime=regime, core=core, status='INVALID_PAIR', detail=str(exc)))
    return pairs


def raw_count(folder):
    count = 0
    files = sorted(folder.glob('batch_*.npz'))
    try:
        for path in files:
            with np.load(path, allow_pickle=False) as archive:
                if 'pos' not in archive:
                    raise ValueError(f'{path.name}: no pos array')
                count += len(archive['pos'])
        return (count if files else np.nan), ''
    except Exception as exc:
        return np.nan, str(exc)


def trajectory_reader():
    if importlib.util.find_spec('mdtraj') is not None:
        import mdtraj as md
        def read(folder):
            trajectory = md.load(str(folder/'samples.xtc'), top=str(folder/'topology.pdb'))
            indices = trajectory.topology.select('name CA')
            if len(indices) != trajectory.topology.n_residues:
                raise ValueError('Topology must contain exactly one CA per residue')
            # MDTraj always exposes coordinates in nm, not guessed from size.
            return trajectory.xyz[:, indices, :].astype(float)*10.0
        return read, 'mdtraj'
    if importlib.util.find_spec('MDAnalysis') is not None:
        import MDAnalysis as mda
        def read(folder):
            universe = mda.Universe(str(folder/'topology.pdb'), str(folder/'samples.xtc'))
            ca = universe.select_atoms('name CA')
            if len(ca) != len(universe.residues):
                raise ValueError('Topology must contain exactly one CA per residue')
            # MDAnalysis converts XTC coordinates to Angstrom by default.
            return np.asarray([ca.positions.copy() for _ in universe.trajectory], dtype=float)
        return read, 'MDAnalysis'
    raise SystemExit('Install an XTC reader in this environment: python -m pip install mdtraj')


def memberships(ra, rb, radius, distinct, minimum):
    near_a, near_b = ra <= radius, rb <= radius
    only_a, only_b = near_a & ~near_b, near_b & ~near_a
    pa, pb = float(only_a.mean()), float(only_b.mean())
    return dict(p_near_A=float(near_a.mean()), p_near_B=float(near_b.mean()),
                p_exclusive_A=pa, p_exclusive_B=pb,
                p_overlap=float((near_a & near_b).mean()),
                p_neither=float((~near_a & ~near_b).mean()),
                support_score=min(pa, pb) if distinct else np.nan,
                two_state_success=int(distinct and pa >= minimum and pb >= minimum))


def interval(values, rng, binary=True):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan, np.nan
    if binary:
        # Wilson interval remains nonzero when no successes were observed.
        n, p, z = len(x), float(x.mean()), 1.95996398454
        denom = 1+z*z/n
        center = (p+z*z/(2*n))/denom
        half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
        return p, max(0.,center-half), min(1.,center+half)
    # Resample pairs, never individual BioEmu frames.
    means = x[rng.integers(0, len(x), (1000, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def aggregate(pairs, sequences, radii, distinct_cutoff):
    records = []
    for pair in pairs.values():
        for radius in radii:
            row = {k:v for k,v in pair.items() if k not in ('A', 'B')}
            row.update(radius=radius, distinct=pair['backbone_rmsd_AB'] >= distinct_cutoff,
                       distinct_threshold=distinct_cutoff)
            candidates = []
            for source in 'AB':
                key = (pair['regime'], pair['core'], source, radius)
                candidate = sequences.get(key)
                if candidate is not None:
                    candidates.append(candidate)
                row[f'seq{source}_available'] = candidate is not None
                for field in ('two_state_success', 'support_score', 'p_self', 'p_cross'):
                    row[f'seq{source}_{field}'] = candidate[field] if candidate is not None else np.nan
            row['complete'] = len(candidates) == 2
            row['n_candidates_scored'] = len(candidates)
            # Report one-attempt average and best2 only with BOTH candidates.
            row['one_attempt_success'] = np.mean([c['two_state_success'] for c in candidates]) if row['complete'] else np.nan
            row['best2_success'] = max(c['two_state_success'] for c in candidates) if row['complete'] else np.nan
            row['best2_support'] = max(c['support_score'] for c in candidates) if row['complete'] and row['distinct'] else np.nan
            row['observed_success_any'] = max([c['two_state_success'] for c in candidates], default=0)
            records.append(row)
    return pd.DataFrame(records)


def summaries(pair_df):
    rows, rng = [], np.random.default_rng(47)
    for keys, group in pair_df.groupby(['task', 'mode', 'noise', 'radius']):
        row = dict(zip(['task', 'mode', 'noise', 'radius'], keys))
        complete = group[group['complete']]
        distinct = complete[complete['distinct']]
        row.update(n_generated_pairs=len(group), n_complete=len(complete),
                   n_generated_distinct=int(group['distinct'].sum()), n_complete_distinct=len(distinct),
                   evaluation_coverage=len(complete)/len(group),
                   distinct_fraction=float(group['distinct'].mean()),
                   observed_success_yield=float(group['observed_success_any'].mean()),
                   median_AB_rmsd=float(group['backbone_rmsd_AB'].median()),
                   median_best2_support=distinct['best2_support'].median())
        for strategy in ('seqA', 'seqB', 'one_attempt', 'best2'):
            column = strategy+'_two_state_success' if strategy in ('seqA','seqB') else strategy+'_success'
            for label, data in (('all_complete', complete), ('distinct_complete', distinct)):
                value, low, high = interval(data[column], rng, binary=strategy!='one_attempt')
                row[f'{strategy}_{label}'] = value
                row[f'{strategy}_{label}_low'] = low
                row[f'{strategy}_{label}_high'] = high
        rows.append(row)
    return pd.DataFrame(rows)


def line_plot(df, column, ylabel, title, out, name, scale=100, ylim=True):
    tasks = sorted(df['task'].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2*len(tasks), 4.4), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        subset = df[df['task'] == task]
        plotted = False
        for (mode, radius), g in subset.groupby(['mode', 'radius']):
            g = g.sort_values('noise')
            if not g[column].notna().any():
                continue
            plotted = True
            color = COLORS.get(mode, 'gray')
            ax.plot(g['noise'], scale*g[column], marker='o', ms=4,
                    ls='-' if radius == 2 else '--', color=color, label=f'{mode}, r={radius:g} Å')
            if column+'_low' in g:
                ax.fill_between(g['noise'].to_numpy(), scale*g[column+'_low'].to_numpy(),
                                scale*g[column+'_high'].to_numpy(), color=color, alpha=.10)
        if not plotted:
            ax.text(.5,.5,'N/A: no eligible evaluated pairs', ha='center', transform=ax.transAxes, fontsize=9)
        positive = subset.loc[subset['noise'] > 0, 'noise']
        if len(positive):
            ax.set_xscale('symlog', linthresh=float(positive.min()))
        ax.set(title=task, xlabel='Initialization noise scale', ylabel=ylabel)
        if ylim:
            ax.set_ylim(-2,102)
        ax.grid(alpha=.2)
        if plotted:
            ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(out/f'{name}.{extension}', dpi=180)
    plt.close(fig)


def plots(pair_df, seq_df, summary, out):
    plt.rcParams.update({'axes.spines.top':False, 'axes.spines.right':False})
    criterion=f'Target A–B ≥ {pair_df["distinct_threshold"].iloc[0]:g} Å; each exclusive fraction ≥ {seq_df["minimum_fraction"].iloc[0]:.0%}; filtered XTC'
    line_plot(summary,'best2_all_complete','Success (%)','Best of two: success among all COMPLETE pairs\n'+criterion,out,'v2_01_success_all')
    line_plot(summary,'best2_distinct_complete','Success (%)','Best of two: success among DISTINCT COMPLETE pairs\n'+criterion,out,'v2_02_success_distinct')
    line_plot(summary,'one_attempt_all_complete','Success (%)','One-attempt baseline: mean of seq_A and seq_B success\n'+criterion,out,'v2_03_one_attempt')
    line_plot(summary,'median_best2_support','Median exclusive min(pA,pB) (%)','Two-state support only for DISTINCT COMPLETE pairs',out,'v2_04_support')
    line_plot(summary,'distinct_fraction','Distinct pairs (%)','Fraction of generated pairs meeting the separation threshold',out,'v2_05_distinct_fraction')
    line_plot(summary,'median_AB_rmsd','Median target A–B RMSD (Å)','Final structural separation',out,'v2_06_AB_rmsd',scale=1,ylim=False)
    for source in 'AB':
        g = seq_df[seq_df['sequence_source']==source]
        for field in ('p_self','p_cross'):
            data = g.groupby(['task','mode','noise','radius'])[field].mean().reset_index()
            line_plot(data,field,'Mean neighborhood coverage (%)',f'seq_{source}: {field} (overlap INCLUDED)',out,f'v2_07_seq{source}_{field}')
    rows = []
    for keys,g in seq_df.drop_duplicates('bioemu_folder').groupby(['task','mode','noise']):
        row = dict(zip(['task','mode','noise'],keys))
        row.update(radius=2, retained=g['retained_fraction'].mean())
        rows.append(row)
    line_plot(pd.DataFrame(rows),'retained','Mean retained samples (%)','BioEmu filtering: remaining / raw samples',out,'v2_08_retained')
    overlap=seq_df.groupby(['task','mode','noise','radius'])['p_overlap'].mean().reset_index()
    line_plot(overlap,'p_overlap','Mean ambiguous samples (%)','Samples within BOTH target neighborhoods; excluded from state support',out,'v2_11_overlap')
    tasks = sorted(seq_df['task'].unique())
    fig, axes = plt.subplots(2,len(tasks),figsize=(5.2*len(tasks),8),squeeze=False)
    for i,radius in enumerate(sorted(seq_df['radius'].unique())):
        for j,task in enumerate(tasks):
            ax = axes[i,j]
            g = seq_df[(seq_df['task']==task)&(seq_df['radius']==radius)&seq_df['distinct']]
            for source,marker,color in (('A','o','#2878B5'),('B','s','#E87524')):
                s = g[g['sequence_source']==source]
                ax.scatter(100*s['p_exclusive_A'],100*s['p_exclusive_B'],s=12,marker=marker,color=color,alpha=.45,label=f'seq_{source}')
            if g.empty:
                ax.text(.5,.5,'N/A: no distinct evaluated pairs',ha='center',transform=ax.transAxes,fontsize=8)
            ax.set(title=f'{task}; r={radius:g} Å',xlabel='Exclusive A (%)',ylabel='Exclusive B (%)',xlim=(-2,102),ylim=(-2,102))
            ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Separate sequences, DISTINCT targets only; overlapping neighborhoods excluded')
    fig.tight_layout(); fig.savefig(out/'v2_09_distinct_sequence_support.png',dpi=180); plt.close(fig)
    diagnostic = pair_df.drop_duplicates(['regime','core'])
    if 'scaffold_motion_after_motif_fit_AB' in diagnostic:
        data = diagnostic.groupby(['task','mode','noise'])['scaffold_motion_after_motif_fit_AB'].median().reset_index()
        data['radius']=2
        line_plot(data,'scaffold_motion_after_motif_fit_AB','Scaffold RMSD after motif fit (Å)',
                  'Conditional tasks: scaffold displacement relative to motif',out,'v2_10_scaffold_motion',scale=1,ylim=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--out',type=Path,default=None)
    parser.add_argument('--distinct',type=float,default=4.)
    parser.add_argument('--min-fraction',type=float,default=.05)
    parser.add_argument('--scaffold-range',help='Optional 1-based inclusive range, e.g. 10:40; ONLY task2/task3, only if verified from run configs')
    parser.add_argument('--plots-only',action='store_true')
    args=parser.parse_args()
    if args.distinct <= 0 or not 0 < args.min_fraction <= .5:
        parser.error('Require distinct > 0 and 0 < min-fraction <= 0.5')
    base=args.base.resolve(); out=args.out or base/'bioemu_plots'/'bioemu_baselines_statistics'
    out.mkdir(parents=True,exist_ok=True)
    if args.plots_only:
        plots(pd.read_csv(out/'v2_pairs.csv'),pd.read_csv(out/'v2_sequences.csv'),pd.read_csv(out/'v2_summary.csv'),out)
        print('Plots refreshed:',out); return
    scaffold=tuple(map(int,args.scaffold_range.split(':'))) if args.scaffold_range else None
    diagnostics=[]
    pairs=discover_pairs(base/'baselines',scaffold,diagnostics)
    pd.DataFrame(diagnostics).to_csv(out/'v2_pair_inventory.csv',index=False)
    if not pairs:
        raise SystemExit('No valid baseline pairs; inspect v2_pair_inventory.csv')
    read,engine=trajectory_reader()
    mapping=defaultdict(list); statuses=[]
    for folder in sorted((base/'bioemu_baselines').iterdir()):
        if not folder.is_dir() or folder.name.startswith('.') or 'checkpoint' in folder.name.lower():
            continue
        mr,mc,ms=REGIME.search(folder.name),CORE.search(folder.name),SOURCE.search(folder.name)
        if not (mr and mc and ms):
            statuses.append(dict(bioemu_folder=folder.name,status='UNPARSED_NAME')); continue
        mapping[(mr.group(),mc.group(),ms.group(1).upper())].append(folder)
    sequence_records=[]; frame_records=[]
    for i,(key,folders) in enumerate(mapping.items(),1):
        regime,core,source=key
        folder=folders[0]
        try:
            if len(folders)!=1:
                raise ValueError('Duplicate source folders; refuse best-of-many: '+','.join(f.name for f in folders))
            pair=pairs[(regime,core)]
            xyz=read(folder)
            if xyz.shape!=(len(xyz),pair['length'],3) or not np.isfinite(xyz).all():
                raise ValueError(f'CA count/coordinates mismatch: {xyz.shape}, target length {pair["length"]}')
            raw,raw_error=raw_count(folder)
            common={k:pair[k] for k in ('regime','task','mode','noise','core','length','backbone_rmsd_AB')}
            common.update(bioemu_folder=folder.name,sequence_source=source,n_raw=raw,n_filtered=len(xyz),
                          retained_fraction=len(xyz)/raw if raw>0 else np.nan,raw_count_error=raw_error,
                          distinct=pair['backbone_rmsd_AB']>=args.distinct,
                          minimum_fraction=args.min_fraction)
            if raw > 0 and len(xyz) > raw:
                common['retained_fraction'] = np.nan
                common['raw_count_error'] = 'Filtered frames exceed raw count: inspect sampling/restarts'
            if not len(xyz):
                raise ValueError('EMPTY_FILTERED_ENSEMBLE: no valid samples; coverage is undefined')
            ra=np.asarray([rmsd(x,pair['A']) for x in xyz]); rb=np.asarray([rmsd(x,pair['B']) for x in xyz])
            for frame,(a,b) in enumerate(zip(ra,rb)):
                frame_records.append(dict(bioemu_folder=folder.name,frame=frame,rmsd_A=a,rmsd_B=b))
            for radius in (2.,3.):
                row=dict(common,radius=radius,**memberships(ra,rb,radius,common['distinct'],args.min_fraction))
                row['p_self']=row['p_near_'+source]; row['p_cross']=row['p_near_'+('B' if source=='A' else 'A')]
                row['self_rmsd_median']=float(np.median(ra if source=='A' else rb))
                sequence_records.append(row)
            statuses.append(dict(bioemu_folder=folder.name,status='OK',n_filtered=len(xyz),n_raw=raw))
        except Exception as exc:
            statuses.append(dict(bioemu_folder=folder.name,status='ERROR',detail=str(exc)))
        if i%100==0:
            print(f'Processed {i}/{len(mapping)} ensembles',flush=True)
    pd.DataFrame(statuses).to_csv(out/'v2_mapping_status.csv',index=False)
    if not sequence_records:
        raise SystemExit('No ensembles evaluated; see v2_mapping_status.csv')
    seq_df=pd.DataFrame(sequence_records); seq_df.to_csv(out/'v2_sequences.csv',index=False)
    pd.DataFrame(frame_records).to_csv(out/'v2_frame_distances.csv',index=False)
    lookup={(r['regime'],r['core'],r['sequence_source'],r['radius']):r for r in sequence_records}
    pair_df=aggregate(pairs,lookup,(2.,3.),args.distinct); pair_df.to_csv(out/'v2_pairs.csv',index=False)
    summary=summaries(pair_df); summary.to_csv(out/'v2_summary.csv',index=False)
    examples=pair_df[(pair_df['radius']==2)&pair_df['complete']].copy()
    examples['category']=np.where(~examples['distinct'],'targets_not_distinct',
        np.where(examples['best2_success']>0,'two_state_supported',
            np.where((examples['seqA_p_self']>=args.min_fraction)|(examples['seqB_p_self']>=args.min_fraction),
                     'distinct_but_only_one_state_supported','distinct_and_poor_own_target_coverage')))
    examples=examples.sort_values(['noise','backbone_rmsd_AB']).groupby(['task','mode','category']).head(2)
    examples.to_csv(out/'v2_representative_pairs.csv',index=False)
    with (out/'v2_settings.json').open('w') as f:
        json.dump(dict(trajectory_engine=engine,units='Angstrom',radii=[2,3],distinct=args.distinct,
                       min_fraction=args.min_fraction,scaffold_range=scaffold,
                       overlap='excluded from two-state support; included in own-target coverage',
                       denominators='success rates: complete pairs; observed_success_yield: all valid generated pairs',
                       uncertainty='Wilson for binary pair rates; pair bootstrap for one-attempt mean; one-attempt all-zero bootstrap can be degenerate',
                       caveat='Support is not evidence of bimodality or experimental folding'),f,indent=2)
    plots(pair_df,seq_df,summary,out)
    print('Saved CSV, PNG/PDF and settings:',out)
    print('Review inventory, mapping errors and evaluation_coverage before interpreting success rates.')


if __name__=='__main__':
    main()
