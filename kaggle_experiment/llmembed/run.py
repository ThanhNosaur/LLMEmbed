"""python -m llmembed.run --help"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import torch

from .data import EXPECTED, prepare_data, write_json
from .features import extract_source
from .train import train_run


def parser():
    p = argparse.ArgumentParser(description='LLMEmbed++ Exp 0-5, with resumable extraction and classifier training.')
    p.add_argument('--datasets', nargs='+', choices=EXPECTED, default=['sst2'])
    p.add_argument('--experiments', nargs='+', choices=['exp0', 'exp1', 'exp2', 'exp3', 'exp4', 'exp5b', 'exp5c'],
                   default=['exp0', 'exp1', 'exp2', 'exp3', 'exp4'])
    p.add_argument('--output', type=Path, default=Path('runs'))
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--reuse-cache', type=Path, help='Read completed feature directories from a previous Kaggle output.')
    p.add_argument('--mr-source', choices=['local', 'imdb_resplit'], default='local')
    p.add_argument('--stage', choices=['data', 'extract', 'train', 'all'], default='all')
    p.add_argument('--plan', action='store_true', help='Print experiment matrix without downloads or training.')
    p.add_argument('--protocol', choices=['controlled', 'author'], default='controlled')
    p.add_argument('--precision', choices=['nf4', 'fp16', 'fp32'], default='nf4')
    p.add_argument('--llama-model', default='meta-llama/Llama-2-7b-hf')
    p.add_argument('--bert-model', default='google-bert/bert-large-uncased')
    p.add_argument('--roberta-model', default='FacebookAI/roberta-large')
    p.add_argument('--deberta-model', default='microsoft/deberta-v3-large')
    p.add_argument('--next-model', default='Qwen/Qwen2.5-7B')
    p.add_argument('--revision', default='main', help='Resolved model commit is recorded in each feature manifest.')
    p.add_argument('--limit', type=int, default=0, help='Stratified sample cap per original split; 0 means full dataset.')
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--val-fraction', type=float, default=0.1)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=1024, help='Effective classifier batch size.')
    p.add_argument('--micro-batch', type=int, default=64, help='Physical batch; gradients accumulate to --batch-size.')
    p.add_argument('--extract-batch', type=int, default=2, help='LLM batch; halves automatically after CUDA OOM.')
    p.add_argument('--encoder-batch', type=int, default=16)
    p.add_argument('--hidden', type=int, default=1024)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--sigma', type=float, default=0.1)
    p.add_argument('--gamma', type=float, default=0.5)
    p.add_argument('--tau', type=float, default=0.1)
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--fusion', type=int, choices=range(1, 16), default=15)
    p.add_argument('--fusion-sweep', action='store_true', help='Add all Table 1 fusions using Exp 0 features.')
    p.add_argument('--stride-sweep', action='store_true', help='Add Exp 1 with strides 1, 2, 4, 6.')
    p.add_argument('--chunk-size', type=int, default=256, help='Content tokens, excluding prompt/special tokens.')
    p.add_argument('--overlap', type=int, default=32)
    p.add_argument('--max-chunks', type=int, default=0, help='0 retains all chunks; nonzero cap is reported as truncation.')
    p.add_argument('--power-watts', type=float, help='Optional assumed TOTAL GPU power for estimated kWh, never measured kWh.')
    return p


def experiment_matrix(args):
    experiments = []
    for name in args.experiments:
        experiments.append(dict(name=name, stride=1 if name == 'exp0' else args.stride,
            cot=name not in ('exp0', 'exp1'), chunked=name in ('exp3', 'exp4', 'exp5b', 'exp5c'),
            adaptive=name in ('exp4', 'exp5b', 'exp5c'), fusion=args.fusion,
            llama=args.next_model if name == 'exp5c' else args.llama_model,
            bert=args.deberta_model if name in ('exp5b', 'exp5c') else args.bert_model,
            roberta=args.roberta_model))
    if args.fusion_sweep:
        baseline = dict(name='exp0', stride=1, cot=False, chunked=False, adaptive=False,
                        fusion=15, llama=args.llama_model, bert=args.bert_model, roberta=args.roberta_model)
        experiments.extend(dict(baseline, name=f'fusion_{i:02}', fusion=i) for i in range(1, 16))
    if args.stride_sweep:
        baseline = dict(name='exp1', cot=False, chunked=False, adaptive=False,
                        fusion=15, llama=args.llama_model, bert=args.bert_model, roberta=args.roberta_model)
        experiments.extend(dict(baseline, name=f'stride_{i}', stride=i) for i in (1, 2, 4, 6))
    return experiments


def extraction_spec(args, dataset, experiment, source):
    llama_lengths = {'sst2': 128, 'mr': 3000, 'agnews': 256, 'r8': 1024, 'r52': 1024}
    is_llama = source == 'llama'
    if is_llama:
        pooling = 'cot' if experiment['cot'] else 'legacy_mean' if args.protocol == 'author' else 'mean'
    else:
        pooling = 'pooler' if source == 'bert' and experiment['bert'] == args.bert_model else 'cls'
    return dict(model=experiment[source], revision=args.revision, task=dataset,
        stride=experiment['stride'] if is_llama else 1, pooling=pooling,
        precision=args.precision if is_llama or args.precision == 'fp32' else 'fp16',
        max_length=512 if experiment['chunked'] or not is_llama else llama_lengths[dataset],
        chunked=experiment['chunked'], chunk_size=args.chunk_size if experiment['chunked'] else 0,
        overlap=args.overlap if experiment['chunked'] else 0, max_chunks=args.max_chunks if experiment['chunked'] else 0,
        gamma=args.gamma if pooling == 'cot' else 0.5, tau=args.tau if pooling == 'cot' else 0.1)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    for name in ('epochs', 'batch_size', 'micro_batch', 'extract_batch', 'encoder_batch', 'hidden', 'heads', 'stride', 'chunk_size'):
        if getattr(args, name) < 1:
            p.error(f'{name} must be positive.')
    if args.limit < 0 or args.max_chunks < 0 or not 0 <= args.overlap < args.chunk_size:
        p.error('Require limit/max_chunks >= 0 and 0 <= overlap < chunk_size.')
    if args.tau <= 0 or not 0 <= args.gamma <= 1 or args.lr <= 0 or args.sigma <= 0:
        p.error('Require tau/lr/sigma > 0 and gamma in [0, 1].')
    if args.hidden < 4 or args.hidden % args.heads or args.batch_size % args.micro_batch:
        p.error('hidden must be >=4 and divisible by heads; batch-size must be divisible by micro-batch.')
    if args.protocol == 'author':
        if args.experiments != ['exp0'] or args.fusion_sweep or args.stride_sweep or args.precision != 'fp16':
            p.error('Author compatibility requires --experiments exp0 --precision fp16, without sweeps.')
        args.llama_model = 'meta-llama/Llama-2-7b-chat-hf'
        args.val_fraction = 0
        if args.hidden != 1024 or args.fusion != 15:
            p.error('Author compatibility requires hidden=1024 and fusion=15.')
    elif not 0 < args.val_fraction < 1:
        p.error('Controlled experiments require 0 < val-fraction < 1.')
    if args.power_watts is not None and args.power_watts <= 0:
        p.error('power-watts must be positive.')
    experiments = experiment_matrix(args)
    if args.plan:
        print(json.dumps(experiments, indent=2))
        return []
    if args.stage in ('all', 'extract') and args.precision != 'fp32' and not torch.cuda.is_available():
        p.error('CUDA unavailable: select --precision fp32 for CPU, or enable the Kaggle GPU.')
    environment = dict(python=platform.python_version(), torch=torch.__version__,
        cuda=torch.version.cuda, gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        packages={name: importlib.metadata.version(name) for name in ('numpy', 'transformers', 'datasets', 'accelerate', 'scikit-learn')},
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    write_json(args.output / 'environment.json', environment)
    results = []
    for dataset in args.datasets:
        data = prepare_data(dataset, args.data_dir, args.output, limit=args.limit, seed=args.split_seed,
                            val_fraction=args.val_fraction, mr_source=args.mr_source)
        if args.stage == 'data':
            continue
        for experiment in experiments:
            print(f'\n{dataset}: {experiment["name"]} ({args.precision}, {args.protocol})', flush=True)
            started = time.perf_counter()
            caches = {}
            for source in ('bert', 'roberta', 'llama'):
                spec = extraction_spec(args, dataset, experiment, source)
                caches[source] = extract_source(data, source, spec, args.output / 'features', args.reuse_cache,
                    args.extract_batch if source == 'llama' else args.encoder_batch, read_only=args.stage == 'train')
            print(f'Feature stage wall time (including cache reuse): {time.perf_counter() - started:.1f}s', flush=True)
            if args.stage != 'extract':
                for seed in args.seeds:
                    config = {key: getattr(args, key) for key in ('protocol', 'epochs', 'batch_size', 'micro_batch',
                        'hidden', 'heads', 'lr', 'weight_decay', 'sigma', 'power_watts')}
                    config['seed'] = seed
                    config['study'] = {key: getattr(args, key) for key in ('llama_model', 'bert_model', 'roberta_model',
                        'deberta_model', 'next_model', 'precision', 'revision', 'stride', 'gamma', 'tau',
                        'chunk_size', 'overlap', 'max_chunks')}
                    results.append(train_run(data, caches, experiment, config, args.output))
            for source in caches.values():
                for store in source.values():
                    store.close()
    if results:
        from .report import make_report
        make_report(args.output)
    return results


if __name__ == '__main__':
    main()
