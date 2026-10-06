#!/usr/bin/env python3
"""Joint positive design of existing La-Proteina A/B pairs using tied ProteinMPNN."""
import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

AA = dict(zip('ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR'.split(), 'ACDEFGHIKLMNPQRSTVWY'))
ATOMS = ('N', 'CA', 'C', 'O')


def backbone(path):
    residues = {}
    for line in path.read_text().splitlines():
        if line.startswith('ENDMDL'):
            break
        if not line.startswith('ATOM  ') or line[16] not in (' ', 'A'):
            continue
        atom = line[12:16].strip()
        if atom not in ATOMS:
            continue
        key = (line[21], line[22:26].strip(), line[26])
        res = residues.setdefault(key, {'name': line[17:20].strip(), 'atoms': {}})
        if atom in res['atoms']:
            raise ValueError(f'{path}: duplicate {atom} at {key}')
        xyz = [float(line[start:start+8]) for start in (30, 38, 46)]
        if not all(math.isfinite(v) for v in xyz):
            raise ValueError(f'{path}: nonfinite coordinates')
        res['atoms'][atom] = xyz
    if not residues or len({k[0] for k in residues}) != 1:
        raise ValueError(f'{path}: expected one nonempty chain')
    for key, res in residues.items():
        if res['name'] not in AA or set(res['atoms']) != set(ATOMS):
            raise ValueError(f'{path}: incomplete/unsupported backbone at {key}')
    return list(residues), list(residues.values())


def prepare(a, b, edges):
    keys_a, ra = backbone(a)
    keys_b, rb = backbone(b)
    if [k[1:] for k in keys_a] != [k[1:] for k in keys_b]:
        raise ValueError('A/B residue numbers and insertion codes do not correspond')
    length = len(ra)
    if length < edges:
        raise ValueError(f'Length {length} < model neighbors {edges}: cannot isolate state graphs')
    record = {'name': 'joint', 'num_of_chains': 2}
    for chain, residues in [('A', ra), ('B', rb)]:
        seq = ''.join(AA[r['name']] for r in residues)
        # Center both independently; translation preserves each state geometry.
        center = [sum(r['atoms']['CA'][d] for r in residues)/length for d in range(3)]
        record[f'seq_chain_{chain}'] = seq
        record[f'coords_chain_{chain}'] = {
            f'{atom}_chain_{chain}': [[r['atoms'][atom][d]-center[d] + (10000 if chain == 'B' and d == 0 else 0)
                                      for d in range(3)] for r in residues] for atom in ATOMS}
    record['seq'] = record['seq_chain_A'] + record['seq_chain_B']
    ca_a = record['coords_chain_A']['CA_chain_A']
    ca_b = record['coords_chain_B']['CA_chain_B']
    # Stronger than checking a distance cutoff: every within-state CA distance
    # must be smaller than every cross-state CA distance (kNN includes self).
    def d2(x, y):
        return sum((u-v)**2 for u, v in zip(x, y))
    max_within = max(d2(x, y) for coords in (ca_a, ca_b) for x in coords for y in coords)
    min_cross = min(d2(x, y) for x in ca_a for y in ca_b)
    if max_within >= min_cross:
        raise ValueError('Artificial cross-state neighbors possible')
    tied = {'joint': [{'A': [[i], [1.0]], 'B': [[i], [1.0]]} for i in range(1, length+1)]}
    return record, tied, length


def candidates(path, length, expected):
    records = []
    header, lines = None, []
    for line in path.read_text().splitlines() + ['>END']:
        if line.startswith('>'):
            if header and 'sample=' in header:
                parts = ''.join(lines).split('/')
                if len(parts) != 2 or parts[0] != parts[1] or len(parts[0]) != length:
                    raise ValueError('ProteinMPNN did not return two identical full-length chains')
                if set(parts[0]) - set(AA.values()):
                    raise ValueError('Unsupported amino acids in output')
                score = float(re.search(r'global_score=([^,\s]+)', header).group(1))
                if not math.isfinite(score):
                    raise ValueError('Nonfinite ProteinMPNN score')
                records.append({'sample': int(re.search(r'sample=(\d+)', header).group(1)),
                                'joint_global_score': score, 'sequence': parts[0]})
            header, lines = line, []
        elif line.strip():
            lines.append(line.strip())
    if len(records) != expected:
        raise ValueError(f'Expected {expected} candidates, received {len(records)}')
    return records


def write_csv(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--mpnn', type=Path, default=Path.home()/'ProteinMPNN')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--model', default='v_48_020')
    parser.add_argument('--num-seq', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=37)
    parser.add_argument('--task', help='For example task1_default')
    parser.add_argument('--limit', type=int, help='First N pairs, for a smoke run')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.num_seq < 1 or args.temperature <= 0 or (args.limit is not None and args.limit < 1):
        parser.error('num-seq, temperature and limit must be positive')
    args.base, args.mpnn = args.base.resolve(), args.mpnn.expanduser().resolve()
    out = (args.out or args.base/'mpnn_joint_results').resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Read actual checkpoint neighbor count; never assume filename implies it.
    import torch
    weight = args.mpnn/'vanilla_model_weights'/f'{args.model}.pt'
    checkpoint = torch.load(weight, map_location='cpu', weights_only=False)
    edges = int(checkpoint['num_edges'])
    del checkpoint
    mpnn_digest = hashlib.sha256(b''.join((args.mpnn/f).read_bytes() for f in
                                         ['protein_mpnn_run.py', 'protein_mpnn_utils.py']) + weight.read_bytes()).hexdigest()
    paths = sorted(p for p in (args.base/'baselines').rglob('*_pathA.pdb')
                   if not any(part.startswith('.') or 'checkpoint' in part for part in p.parts)
                   and re.fullmatch(r'task\d+_(default|sim_eps)', p.relative_to(args.base/'baselines').parts[0])
                   and (not args.task or p.relative_to(args.base/'baselines').parts[0] == args.task))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit('No matching baseline pairs')
    status, selected = [], []
    for number, a in enumerate(paths, 1):
        rel = a.relative_to(args.base/'baselines')
        b = a.with_name(a.name.replace('_pathA.pdb', '_pathB.pdb'))
        pair_id = '__'.join(rel.parent.parts) + '__' + a.stem.removesuffix('_pathA')
        folder = out/'pairs'/pair_id
        folder.mkdir(parents=True, exist_ok=True)
        print(f'[{number}/{len(paths)}] {pair_id}', flush=True)
        row = {'pair_id': pair_id, 'pdb_A': str(a), 'pdb_B': str(b), 'status': '', 'error': ''}
        try:
            record, tied, length = prepare(a, b, edges)
            settings = {'protocol': 1, 'model': args.model, 'num_seq': args.num_seq,
                        'temperature': args.temperature, 'seed': args.seed, 'mpnn_sha256': mpnn_digest,
                        'A_sha256': hashlib.sha256(a.read_bytes()).hexdigest(),
                        'B_sha256': hashlib.sha256(b.read_bytes()).hexdigest()}
            signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
            done = folder/'complete.json'
            if done.exists():
                cached = json.loads(done.read_text())
                if cached['signature'] != signature:
                    raise ValueError('Existing results have different settings/inputs; use another --out')
                samples = candidates(folder/'design'/'seqs'/'joint.fa', length, args.num_seq)
            else:
                for name, data in [('parsed.jsonl', record), ('tied.jsonl', tied),
                                   ('chains.jsonl', {'joint': [['A', 'B'], []]})]:
                    (folder/name).write_text(json.dumps(data)+'\n')
                (folder/'settings.json').write_text(json.dumps(settings, indent=2))
                if args.prepare_only:
                    row['status'] = 'PREPARED'
                    status.append(row)
                    continue
                command = [sys.executable, str(args.mpnn/'protein_mpnn_run.py'),
                           '--jsonl_path', str(folder/'parsed.jsonl'), '--chain_id_jsonl', str(folder/'chains.jsonl'),
                           '--tied_positions_jsonl', str(folder/'tied.jsonl'), '--out_folder', str(folder/'design'),
                           '--model_name', args.model, '--num_seq_per_target', str(args.num_seq),
                           '--sampling_temp', str(args.temperature), '--seed', str(args.seed), '--batch_size', '1']
                with (folder/'proteinmpnn.log').open('w') as log:
                    subprocess.run(command, cwd=args.mpnn, stdout=log, stderr=subprocess.STDOUT, check=True)
                samples = candidates(folder/'design'/'seqs'/'joint.fa', length, args.num_seq)
            best = min(samples, key=lambda r: (r['joint_global_score'], r['sample']))
            write_csv(folder/'candidates.csv', samples, ['sample', 'joint_global_score', 'sequence'])
            (folder/'sequence.fasta').write_text(f'>{pair_id}_joint_best\n{best["sequence"]}\n')
            done.write_text(json.dumps({'signature': signature, 'best': best}, indent=2))
            selected.append({'pair_id': pair_id, 'task_mode': rel.parts[0], 'pdb_A': str(a), 'pdb_B': str(b),
                             'sequence_fasta': str(folder/'sequence.fasta'), 'length': length, **best})
            row['status'] = 'COMPLETE'
        except Exception as error:
            row.update(status='ERROR', error=str(error))
            print(f'  ERROR: {error}', flush=True)
        status.append(row)
    # Separate reports for subsets, so a smoke run cannot replace the full manifest.
    label = ('_'+args.task if args.task else '') + (f'_first{args.limit}' if args.limit else '')
    write_csv(out/f'status{label}.csv', status, ['pair_id', 'pdb_A', 'pdb_B', 'status', 'error'])
    write_csv(out/f'manifest{label}.csv', selected, ['pair_id', 'task_mode', 'pdb_A', 'pdb_B',
              'sequence_fasta', 'length', 'sample', 'joint_global_score', 'sequence'])
    with (out/f'best_joint_sequences{label}.fasta').open('w') as handle:
        for row in selected:
            handle.write(f'>{row["pair_id"]}_joint_best\n{row["sequence"]}\n')
    print(f'Complete: {len(selected)}/{len(paths)}; reports: {out}')
    if any(row['status'] == 'ERROR' for row in status):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
