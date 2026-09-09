#!/usr/bin/env python3
"""
Обработка всех результатов ProteinMPNN из разных задач и режимов.
Выбирает лучшую последовательность для каждого бэкбона и сохраняет в единый FASTA + CSV.
"""

import os
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ==================== НАСТРОЙКИ ====================
MPNN_RESULTS_DIR = Path('/home/domain/aristowi/la-proteina-main/mpnn_results')
OUTPUT_FASTA = Path('/home/domain/aristowi/la-proteina-main/best_sequences_all.fasta')
OUTPUT_CSV = Path('/home/domain/aristowi/la-proteina-main/mpnn_metrics_all.csv')

# Регулярное выражение для парсинга заголовков сэмплов
sample_header_pattern = re.compile(
    r'>T=[\d.]+,\s*sample=(\d+),\s*score=[\d.]+,\s*global_score=([\d.]+),\s*seq_recovery=([\d.]+)'
)

def parse_mpnn_fasta(fasta_file):
    """Парсит .fa файл ProteinMPNN и извлекает информацию о всех последовательностях"""
    results = []
    
    with open(fasta_file, 'r') as f:
        lines = f.readlines()
    
    backbone_name = None
    backbone_score = None
    backbone_global_score = None
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if not line:
            i += 1
            continue
        
        # Заголовок бэкбона (первая запись в файле)
        if line.startswith('>') and 'sample=' not in line:
            # Извлекаем имя бэкбона (до запятой)
            backbone_name = line[1:].split(',')[0].strip()
            # Пытаемся извлечь score и global_score
            score_match = re.search(r'score=([\d.]+)', line)
            global_score_match = re.search(r'global_score=([\d.]+)', line)
            
            if score_match:
                backbone_score = float(score_match.group(1))
            if global_score_match:
                backbone_global_score = float(global_score_match.group(1))
            
            i += 1
            # Пропускаем последовательность бэкбона
            if i < len(lines) and not lines[i].strip().startswith('>'):
                i += 1
            continue
        
        # Заголовок сгенерированной последовательности
        if line.startswith('>') and 'sample=' in line:
            match_sample = sample_header_pattern.match(line)
            
            if match_sample:
                sample_id = int(match_sample.group(1))
                global_score = float(match_sample.group(2))
                seq_recovery = float(match_sample.group(3))
                
                # Следующая строка - сама последовательность
                i += 1
                if i < len(lines):
                    sequence = lines[i].strip()
                    
                    results.append({
                        'sample_id': sample_id,
                        'global_score': global_score,
                        'seq_recovery': seq_recovery,
                        'sequence': sequence
                    })
        
        i += 1
    
    return backbone_name, backbone_score, backbone_global_score, results

def main():
    print("=" * 70)
    print("ОБРАБОТКА ВСЕХ РЕЗУЛЬТАТОВ PROTEINMPNN")
    print("=" * 70)
    
    # Находим все .fa файлы рекурсивно
    fa_files = list(MPNN_RESULTS_DIR.rglob('*.fa'))
    print(f"\nНайдено {len(fa_files)} файлов .fa")
    
    if not fa_files:
        print("❌ Ошибка: Файлы не найдены! Проверь путь MPNN_RESULTS_DIR.")
        return
    
    all_results = []
    best_sequences = []
    
    # Обрабатываем каждый файл
    for fa_file in tqdm(fa_files, desc="Обработка файлов"):
        # Извлекаем информацию о пути для сохранения структуры
        rel_path = fa_file.relative_to(MPNN_RESULTS_DIR)
        task_name = rel_path.parts[0]  # task1_default, task2_sim_eps, etc.
        backbone_folder = rel_path.parts[1] if len(rel_path.parts) > 1 else ""
        fa_filename = fa_file.stem  # имя файла без расширения
        
        backbone_name, backbone_score, backbone_global_score, samples = parse_mpnn_fasta(fa_file)
        
        if samples:
            # Находим лучшую последовательность (минимальный global_score)
            best_sample = min(samples, key=lambda x: x['global_score'])
            
            # Создаем уникальное имя для бэкбона
            unique_backbone_name = f"{task_name}/{backbone_folder}/{fa_filename}"
            
            # Сохраняем все сэмплы для полной статистики
            for sample in samples:
                all_results.append({
                    'task': task_name,
                    'backbone_folder': backbone_folder,
                    'backbone_name': backbone_name or fa_filename,
                    'unique_backbone_name': unique_backbone_name,
                    'sample_id': sample['sample_id'],
                    'global_score': sample['global_score'],
                    'seq_recovery': sample['seq_recovery'],
                    'seq_length': len(sample['sequence']),
                    'sequence': sample['sequence'],
                    'is_best': False
                })
            
            # Помечаем лучший сэмпл
            for result in all_results[-len(samples):]:
                if result['sample_id'] == best_sample['sample_id']:
                    result['is_best'] = True
                    break
            
            # Добавляем лучшую последовательность в отдельный список
            best_sequences.append({
                'task': task_name,
                'backbone_folder': backbone_folder,
                'backbone_name': backbone_name or fa_filename,
                'unique_backbone_name': unique_backbone_name,
                'sample_id': best_sample['sample_id'],
                'global_score': best_sample['global_score'],
                'seq_recovery': best_sample['seq_recovery'],
                'seq_length': len(best_sample['sequence']),
                'sequence': best_sample['sequence']
            })
    
    # Создаем DataFrame
    df_all = pd.DataFrame(all_results)
    df_best = pd.DataFrame(best_sequences)
    
    # ==================== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ====================
    
    # 1. Сохраняем FASTA файл с лучшими последовательностями
    with open(OUTPUT_FASTA, 'w') as f:
        for seq_info in best_sequences:
            # Формат заголовка: >task/backbone_folder/backbone_name_best
            header = f">{seq_info['unique_backbone_name']}_best"
            f.write(f"{header}\n")
            f.write(f"{seq_info['sequence']}\n")
    
    print(f"\n✅ FASTA файл с лучшими последовательностями сохранен:")
    print(f"   📄 {OUTPUT_FASTA}")
    print(f"   📊 Количество последовательностей: {len(best_sequences)}")
    
    # 2. Сохраняем CSV файл со ВСЕМИ последовательностями
    df_all.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ CSV файл со всеми последовательностями сохранен:")
    print(f"   📄 {OUTPUT_CSV}")
    print(f"   📊 Количество строк: {len(df_all)}")
    
    # ==================== СТАТИСТИКА ====================
    print("\n" + "=" * 70)
    print("СТАТИСТИКА ПО ЗАДАЧАМ")
    print("=" * 70)
    
    # Статистика по каждой задаче
    for task in sorted(df_best['task'].unique()):
        task_df = df_best[df_best['task'] == task]
        print(f"\n📋 {task}:")
        print(f"   Количество бэкбонов: {len(task_df)}")
        print(f"   Средний global_score: {task_df['global_score'].mean():.3f}")
        print(f"   Медианный global_score: {task_df['global_score'].median():.3f}")
        print(f"   Средний seq_recovery: {task_df['seq_recovery'].mean():.3f}")
        print(f"   Средняя длина: {task_df['seq_length'].mean():.1f}")
    
    # Общая статистика
    print("\n" + "=" * 70)
    print("ОБЩАЯ СТАТИСТИКА")
    print("=" * 70)
    print(f"\n📈 Всего обработано бэкбонов: {len(df_best)}")
    print(f"📈 Всего сгенерировано последовательностей: {len(df_all)}")
    print(f"📈 Средний global_score (лучшие): {df_best['global_score'].mean():.3f}")
    print(f"📈 Медианный global_score (лучшие): {df_best['global_score'].median():.3f}")
    
    # Распределение по качеству
    print(f"\n📊 Распределение по качеству (global_score лучших):")
    excellent = len(df_best[df_best['global_score'] < 1.0])
    good = len(df_best[(df_best['global_score'] >= 1.0) & (df_best['global_score'] < 1.5)])
    ok = len(df_best[(df_best['global_score'] >= 1.5) & (df_best['global_score'] < 2.0)])
    poor = len(df_best[df_best['global_score'] >= 2.0])
    
    print(f"   • Отличные (< 1.0): {excellent} ({100*excellent/len(df_best):.1f}%)")
    print(f"   • Хорошие (1.0 - 1.5): {good} ({100*good/len(df_best):.1f}%)")
    print(f"   • Приемлемые (1.5 - 2.0): {ok} ({100*ok/len(df_best):.1f}%)")
    print(f"   • Плохие (≥ 2.0): {poor} ({100*poor/len(df_best):.1f}%)")
    
    # Топ-10 лучших дизайнов
    print("\n🏆 Топ-10 лучших дизайнов (по global_score):")
    top_10 = df_best.nsmallest(10, 'global_score')[['unique_backbone_name', 'global_score', 'seq_recovery', 'seq_length']]
    print(top_10.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("ГОТОВО!")
    print("=" * 70)
    print(f"\nСледующие шаги:")
    print(f"1. FASTA файл готов для BioEmu: {OUTPUT_FASTA}")
    print(f"2. CSV файл содержит полную информацию: {OUTPUT_CSV}")
    print(f"3. Можешь открыть CSV в Excel или загрузить в Python для анализа")

if __name__ == "__main__":
    main()