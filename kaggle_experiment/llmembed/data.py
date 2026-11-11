"""Dataset provenance and deterministic train-only validation splits."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
import warnings

import numpy as np

EXPECTED = {'sst2': (67349, 872, 2), 'mr': (40000, 10000, 2),
            'agnews': (120000, 7600, 4), 'r8': (5485, 2189, 8), 'r52': (6532, 2568, 52)}
TEXTGCN_COMMIT = '962223652e9bb164ac2d83cd09fc7b8845ce860b'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def normalize_records(data):
    if isinstance(data, dict):
        texts, labels = data.get('text', data.get('sentence')), data.get('label')
        if texts is None or labels is None or len(texts) != len(labels):
            raise ValueError('Expected equally sized text and label arrays.')
        data = [dict(text=t, label=y) for t, y in zip(texts, labels)]
    if not isinstance(data, list) or not data:
        raise ValueError('Dataset must be a nonempty list of {text, label} records.')
    result = []
    for i, row in enumerate(data):
        text, label = row.get('text', row.get('sentence')), row.get('label')
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'Empty/non-string text at row {i}.')
        if not isinstance(label, (str, int)) or isinstance(label, bool) or str(label) in ('', '-1'):
            raise ValueError(f'Missing/invalid label at row {i}: {label!r}')
        result.append(dict(text=text, label=str(label)))
    return result


def split_indices(labels, fraction, seed):
    """Keep singleton classes in training; every other class contributes validation."""
    if not 0 <= fraction < 1:
        raise ValueError('Validation fraction must be in [0, 1).')
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train, val = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        n = min(len(indices) - 1, max(1, round(len(indices) * fraction))) if fraction else 0
        val.extend(indices[:n].tolist())
        train.extend(indices[n:].tolist())
    return sorted(train), sorted(val)


def subset(rows, limit, seed):
    if not limit or limit >= len(rows):
        return rows
    labels = np.array([r['label'] for r in rows])
    rng = np.random.default_rng(seed)
    # Round-robin classes makes a small smoke run useful even on imbalanced R52.
    groups = [rng.permutation(np.flatnonzero(labels == y)).tolist() for y in np.unique(labels)]
    chosen = []
    while len(chosen) < limit:
        for group in groups:
            if group and len(chosen) < limit:
                chosen.append(group.pop())
    return [rows[i] for i in sorted(chosen)]


def load_raw(dataset, data_dir, download_dir, mr_source):
    local = Path(data_dir) / dataset if data_dir else None
    if local and local.exists():
        def read(split):
            path = local / f'{split}.json'
            if path.exists():
                return normalize_records(json.loads(path.read_text(encoding='utf-8')))
            path = local / f'{split}.jsonl'
            return normalize_records([json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()])
        return read('train'), read('test'), f'local:{local.resolve()}', 'local_split_unverified'
    if dataset in ('r8', 'r52'):
        base = f'https://raw.githubusercontent.com/yao8839836/text_gcn/{TEXTGCN_COMMIT}/data/'
        parts = []
        for filename in (f'{dataset.upper()}.txt', f'corpus/{dataset.upper()}.txt'):
            target = Path(download_dir) / filename
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with urlopen(base + filename, timeout=120) as response:
                    content = response.read()
                temp = target.with_suffix('.tmp')
                temp.write_bytes(content)
                temp.replace(target)
            parts.append(target.read_text(encoding='utf-8').splitlines())
        metadata, texts = parts
        if len(metadata) != len(texts):
            raise ValueError('TextGCN text/metadata row counts differ.')
        splits = {'train': [], 'test': []}
        for meta, text in zip(metadata, texts):
            _, split, label = meta.split('\t')
            splits[split].append(dict(text=text, label=label))
        return (normalize_records(splits['train']), normalize_records(splits['test']),
                base, 'textgcn_split_preprocessing_not_verified_against_llmembed')
    from datasets import load_dataset
    if dataset == 'mr':
        if mr_source != 'imdb_resplit':
            raise ValueError('Exact MR split is unavailable. Supply data/mr/train.json and test.json, '
                             'or explicitly select --mr-source imdb_resplit (different benchmark split).')
        warnings.warn('MR proxy: IMDb re-split 40k/10k, seed 42; NOT the verified LLMEmbed MR split.')
        source = load_dataset('stanfordnlp/imdb')
        rows = normalize_records([dict(text=r['text'], label=r['label'])
                                  for split in ('train', 'test') for r in source[split]])
        train, test = split_indices([r['label'] for r in rows], 0.2, 42)
        return [rows[i] for i in train], [rows[i] for i in test], 'stanfordnlp/imdb:80-20:seed42', 'imdb_resplit_proxy'
    name, field, test_split = ('stanfordnlp/sst2', 'sentence', 'validation') if dataset == 'sst2' else ('fancyzhx/ag_news', 'text', 'test')
    source = load_dataset(name)
    rows = [normalize_records([dict(text=r[field], label=r['label']) for r in source[s]])
            for s in ('train', test_split)]
    return *rows, name, f'huggingface_train_{test_split}'


def prepare_data(dataset, data_dir, output_dir, limit=0, seed=42, val_fraction=0.1, mr_source='local'):
    if dataset not in EXPECTED or limit < 0:
        raise ValueError('Unknown dataset or negative sample limit.')
    train, test, source, status = load_raw(dataset, data_dir, Path(output_dir) / 'downloads', mr_source)
    names = sorted({r['label'] for r in train})
    mapping = {name: i for i, name in enumerate(names)}
    raw_counts = (len(train), len(test), len(names))
    raw_hash = digest([train, test])
    for split, rows in [('train', train), ('test', test)]:
        for i, row in enumerate(rows):
            if row['label'] not in mapping:
                raise ValueError(f'Test label absent from train: {row["label"]}')
            row.update(id=f'{split}:{i}', label=mapping[row['label']])
    train, test = subset(train, limit, seed), subset(test, limit, seed)
    tr, val = split_indices([r['label'] for r in train], val_fraction, seed)
    if val_fraction and not val:
        raise ValueError('Not enough training examples for validation; increase --limit.')
    splits = dict(train=[train[i] for i in tr], val=[train[i] for i in val], test=test)
    train_texts = {digest(r['text']) for r in train}
    overlap = sum(digest(r['text']) in train_texts for r in test)
    metadata = dict(dataset=dataset, source=source, benchmark_status=status, raw_sha256=raw_hash,
                    label_names=names, original_counts=raw_counts, paper_counts=EXPECTED[dataset],
                    counts_match_paper=raw_counts == EXPECTED[dataset], split_seed=seed,
                    val_fraction=val_fraction, limit=limit, exact_text_overlap_train_test=overlap)
    if overlap:
        warnings.warn(f'{dataset}: {overlap} test texts also occur in train; preserving the source split, recorded in manifest.')
    metadata['fingerprint'] = digest([metadata, splits])
    target = Path(output_dir) / 'data' / f'{dataset}-{metadata["fingerprint"][:16]}'
    write_json(target / 'manifest.json', metadata)
    for split, rows in splits.items():
        write_json(target / f'{split}.json', rows)
    return dict(metadata, splits=splits)
