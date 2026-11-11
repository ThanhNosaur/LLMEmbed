"""Export observed metrics and diagnostic plots; never fill expected results."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

from .data import EXPECTED, write_json


def write_csv(path, rows):
    if not rows:
        Path(path).unlink(missing_ok=True)
        return
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_report(output):
    output = Path(output)
    records = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((output / 'results').glob('*/result.json'))]
    rows, groups = [], defaultdict(list)
    for r in records:
        row = {k: r[k] for k in ('dataset', 'experiment', 'seed', 'benchmark_status', 'sample_limit', 'protocol',
            'precision', 'best_epoch', 'parameters', 'extraction_seconds', 'training_seconds', 'inference_seconds',
            'token_retention', 'estimated_gpu_kwh', 'energy_method', 'data_fingerprint', 'comparison_key', 'study_key', 'run_group')}
        row.update(r['metrics'])
        rows.append(row)
        groups[r['run_group'], r['dataset'], r['experiment']].append(r)
    write_csv(output / 'metrics.csv', rows)
    summaries = []
    for (run_group, dataset, name), group in groups.items():
        group.sort(key=lambda r: r['seed'])
        key = group[0]['comparison_key']
        values = [r['metrics']['accuracy'] for r in group]
        baseline = [r for r in records if r['comparison_key'] == key and r['dataset'] == dataset and r['experiment'] == 'exp0']
        seeds = {r['seed']: r['metrics']['accuracy'] for r in baseline}
        paired = [r['metrics']['accuracy'] - seeds[r['seed']] for r in group if r['seed'] in seeds]
        summaries.append(dict(dataset=dataset, experiment=name, comparison_key=key, run_group=run_group, study_key=group[0]['study_key'],
            benchmark_status=group[0]['benchmark_status'], sample_limit=group[0]['sample_limit'],
            precision=group[0]['precision'], seeds=';'.join(str(r['seed']) for r in group), n_seeds=len(values),
            accuracy_mean=statistics.mean(values), accuracy_std=statistics.stdev(values) if len(values) > 1 else None,
            macro_f1_mean=statistics.mean(r['metrics']['macro_f1'] for r in group),
            delta_exp0_pp=100 * statistics.mean(paired) if len(paired) == len(group) and len(seeds) == len(baseline) else None,
            extraction_seconds=group[0]['extraction_seconds'],
            training_seconds_mean=statistics.mean(r['training_seconds'] for r in group)))
    write_csv(output / 'ablation.csv', summaries)
    averages = []
    by_study = defaultdict(list)
    for row in summaries:
        by_study[row['study_key'], row['experiment']].append(row)
    for (study, experiment), group in by_study.items():
        if len(group) == 5 and {r['dataset'] for r in group} == set(EXPECTED) and len({r['seeds'] for r in group}) == 1:
            averages.append(dict(study_key=study, experiment=experiment,
                avg5_accuracy=statistics.mean(r['accuracy_mean'] for r in group),
                dataset_protocols='; '.join(f'{r["dataset"]}:{r["benchmark_status"]}' for r in group),
                sample_limit=group[0]['sample_limit'], precision=group[0]['precision']))
    write_csv(output / 'average_five_datasets.csv', averages)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for r in records:
        directory = output / 'results' / Path(r['result_dir']).name
        if r['gate_mean']:
            fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
            ax.bar(range(7), r['gate_mean'])
            ax.set_xticks(range(7), [name.rsplit('/', 1)[-1] for name in r['source_names']], rotation=25, ha='right')
            ax.set(ylabel='Mean routing weight on test', title=f'{r["dataset"]} / {r["experiment"]} / seed {r["seed"]}')
            fig.savefig(directory / 'source_weights.png', dpi=180)
            plt.close(fig)
    paths = set((output / 'features').glob('*/state.json'))
    for r in records:
        for directory in r['feature_dirs']['llama'].values():
            local = output / 'features' / Path(directory).name / 'state.json'
            path = local if local.exists() else Path(directory) / 'state.json'
            if path.exists():
                paths.add(path)
    for path in sorted(paths):
        state = json.loads(path.read_text(encoding='utf-8'))
        examples = {}
        for example in state.get('explanations', []):
            # shortcut: plot the first chunk per document; all chunk weights remain in state.json.
            examples.setdefault(example['id'], example)
        for i, example in enumerate(list(examples.values())[:4]):
            tokens, weights = example['tokens'], example['weights']
            lines = (len(tokens) + 19) // 20
            fig, axes = plt.subplots(lines, 1, figsize=(14, max(2, lines * 1.1)), squeeze=False, layout='constrained')
            for line, ax in enumerate(axes[:, 0]):
                start = line * 20
                segment = tokens[start:start + 20]
                ax.imshow([weights[start:start + 20]], cmap='YlOrRd', aspect='auto', vmin=0, vmax=max(weights))
                ax.set_xticks(range(len(segment)), segment, rotation=30, ha='right', fontsize=8)
                ax.set_yticks([])
            fig.suptitle(f'Cosine-guided token weights: {example["id"]}, chunk {example["chunk"]}, layer {example["depth"]}')
            destination = output / 'explanations' / path.parent.name
            destination.mkdir(parents=True, exist_ok=True)
            fig.savefig(destination / f'tokens_{i}.png', dpi=150)
            plt.close(fig)
    write_json(output / 'report_notes.json', dict(
        accuracy_std='Sample standard deviation across seeds; null with one seed.',
        delta='Percentage points vs Exp 0, paired seeds and identical data/training settings.',
        extraction_seconds='Original extraction compute for all three sources; excludes model download/load. Reused caches retain their original time.',
        training_seconds='Training plus validation, excludes checkpoint serialization.',
        energy='No energy measurement. Optional kWh uses user-supplied total GPU watts.',
        explainability='Routing/cosine weights are diagnostics, not causal attributions or evidence of generated reasoning.',
        average='Produced only with all five datasets and matching seeds. MR proxy remains explicitly identified.'))
    print(f'Reports: {output / "metrics.csv"} and {output / "ablation.csv"}', flush=True)
    return summaries


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('output', type=Path)
    make_report(p.parse_args().output)
