"""Train only the classifier; select checkpoints using train-derived validation."""
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score

from .data import digest, write_json
from .models import ExperimentModel


class CachedDataset(Dataset):
    def __init__(self, stores, rows):
        self.stores, self.rows = stores, rows
        if any(store.done != len(rows) for store in stores):
            raise ValueError('Feature/label document counts differ.')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return [store[index] for store in self.stores], self.rows[index]['label']


def collate(batch):
    features, labels = zip(*batch)
    tensors, masks = [], []
    for source in range(3):
        arrays = [item[source] for item in features]
        longest = max(len(a) for a in arrays)
        tensor = torch.zeros(len(batch), longest, *arrays[0].shape[1:], dtype=torch.float32)
        mask = torch.zeros(len(batch), longest, dtype=torch.bool)
        for i, array in enumerate(arrays):
            tensor[i, :len(array)] = torch.from_numpy(np.array(array, dtype=np.float32, copy=True))
            mask[i, :len(array)] = True
        tensors.append(tensor)
        masks.append(mask)
    return tensors, masks, torch.tensor(labels, dtype=torch.long)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_device(batch, device):
    features, masks, labels = batch
    return [x.to(device) for x in features], [x.to(device) for x in masks], labels.to(device)


@torch.inference_mode()
def evaluate(model, loader, device, classes):
    model.eval()
    predictions, labels, gates, chunk_examples = [], [], [], []
    total_loss = 0.0
    for batch in loader:
        features, masks, target = to_device(batch, device)
        logits, gate, chunks = model(features, masks)
        total_loss += nn.functional.cross_entropy(logits, target, reduction='sum').item()
        predictions.extend(logits.argmax(-1).cpu().tolist())
        labels.extend(target.cpu().tolist())
        if gate is not None:
            gates.append(gate.cpu())
        if chunks is not None and len(chunk_examples) < 4:
            for i in range(min(len(target), 4 - len(chunk_examples))):
                chunk_examples.append([weights[i, masks[s][i]].cpu().tolist() for s, weights in enumerate(chunks)])
    metrics = dict(loss=total_loss / len(labels), accuracy=accuracy_score(labels, predictions),
                   macro_f1=f1_score(labels, predictions, labels=list(range(classes)), average='macro', zero_division=0),
                   micro_f1=f1_score(labels, predictions, labels=list(range(classes)), average='micro', zero_division=0))
    return metrics, predictions, torch.cat(gates).mean(0).tolist() if gates else None, chunk_examples


def save_checkpoint(path, content):
    temporary = path.with_suffix('.tmp')
    torch.save(content, temporary)
    temporary.replace(path)


def train_run(data, caches, experiment, config, output_dir):
    implementation = digest([Path(__file__).read_text(encoding='utf-8'),
                             Path(__file__).with_name('models.py').read_text(encoding='utf-8')])
    signature = dict(data=data['fingerprint'], experiment=experiment, training=config,
                     features=[caches[s]['train'].state['signature'] for s in ('llama', 'bert', 'roberta')],
                     training_format=1, implementation=implementation)
    target = Path(output_dir) / 'results' / f'{data["dataset"]}-{experiment["name"]}-{digest(signature)[:16]}'
    target.mkdir(parents=True, exist_ok=True)
    result_path = target / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text(encoding='utf-8'))
    seed_everything(config['seed'])
    classes = len(data['label_names'])
    dims = [caches[s]['train'].shape[-1] for s in ('llama', 'bert', 'roberta')]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    author = config['protocol'] == 'author'
    model = ExperimentModel(classes, dims, chunked=experiment['chunked'], adaptive=experiment['adaptive'],
        strategy=experiment['fusion'], hidden=config['hidden'], heads=config['heads'], sigma=config['sigma'], author=author).to(device)
    optimizer = (torch.optim.Adam(model.parameters(), lr=config['lr']) if author else
                 torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']))
    scheduler = None if author else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'])
    datasets = {split: CachedDataset([caches[s][split] for s in ('llama', 'bert', 'roberta')], rows)
                for split, rows in data['splits'].items() if rows}
    if not author and 'val' not in datasets:
        raise ValueError('Controlled protocol requires a train-derived validation split.')
    batch_size = config['micro_batch']
    effective = config['batch_size']
    if batch_size < 1 or effective % batch_size:
        raise ValueError('Effective batch size must be divisible by micro batch size.')
    val_loader = DataLoader(datasets['val'], batch_size=batch_size, collate_fn=collate) if 'val' in datasets else None
    history, first_epoch, best, best_epoch, elapsed = [], 0, -1.0, 0, 0.0
    last_path, best_path = target / 'last.pt', target / 'best.pt'
    if last_path.exists():
        state = torch.load(last_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['rng'])
        if device.type == 'cuda' and state['cuda_rng']:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        first_epoch, history, best, best_epoch, elapsed = state['epoch'], state['history'], state['best'], state['best_epoch'], state['seconds']
    write_json(target / 'config.json', signature)
    for epoch in range(first_epoch, config['epochs']):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        generator = torch.Generator().manual_seed(config['seed'] + epoch)
        loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, generator=generator, collate_fn=collate)
        model.train()
        seen, loss_sum, correct = 0, 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for batch in loader:
            features, masks, labels = to_device(batch, device)
            if seen % effective == 0:
                group_size = min(effective, len(datasets['train']) - seen)
            logits, _, _ = model(features, masks)
            loss = nn.functional.cross_entropy(logits, labels, reduction='sum')
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss; cached data/checkpoint preserved.')
            (loss / group_size).backward()
            seen += len(labels)
            loss_sum += loss.item()
            correct += (logits.argmax(-1) == labels).sum().item()
            if seen % effective == 0 or seen == len(datasets['train']):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        validation = evaluate(model, val_loader, device, classes)[0] if val_loader else None
        if scheduler is not None:
            scheduler.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - started
        history.append(dict(epoch=epoch + 1, loss=loss_sum / seen, accuracy=correct / seen, validation=validation))
        score = validation['accuracy'] if validation else float(epoch)
        if score > best:
            best, best_epoch = score, epoch + 1
            save_checkpoint(best_path, model.state_dict())
        save_checkpoint(last_path, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict() if scheduler else None, rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [], epoch=epoch + 1,
            history=history, best=best, best_epoch=best_epoch, seconds=elapsed))
        print(f'{data["dataset"]}/{experiment["name"]} seed={config["seed"]} epoch={epoch + 1}: '
              f'train={correct / seen:.4f} val={validation["accuracy"] if validation else "unused"}', flush=True)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_loader = DataLoader(datasets['test'], batch_size=batch_size, collate_fn=collate)
    started = time.perf_counter()
    metrics, predictions, gate_mean, chunks = evaluate(model, test_loader, device, classes)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started
    stores = [store for source in caches.values() for store in source.values()]
    original = sum(s.state['original_tokens'] for s in stores)
    retained = sum(s.state['retained_tokens'] for s in stores)
    result = dict(dataset=data['dataset'], experiment=experiment['name'], seed=config['seed'],
        benchmark_status=data['benchmark_status'], sample_limit=data['limit'], protocol=config['protocol'],
        precision=caches['llama']['train'].state['signature']['spec']['precision'],
        data_fingerprint=data['fingerprint'], comparison_key=digest([data['fingerprint'], dict(config, seed=None), implementation]),
        run_group=digest(dict(signature, training=dict(config, seed=None))),
        study_key=digest([dict(config, seed=None), data['limit'], data['val_fraction'], data['split_seed'], implementation]),
        metrics=metrics, best_epoch=best_epoch, training_seconds=elapsed, inference_seconds=inference_seconds,
        extraction_seconds=sum(s.state['seconds'] for s in stores), token_retention=retained / original,
        retention_by_source={s: sum(v.state['retained_tokens'] for v in splits.values()) /
                               sum(v.state['original_tokens'] for v in splits.values()) for s, splits in caches.items()},
        parameters=sum(p.numel() for p in model.parameters()), gate_mean=gate_mean,
        feature_dirs={source: {split: str(store.path) for split, store in splits.items()} for source, splits in caches.items()},
        source_names=[f'LLM layer {d}' for d in caches['llama']['train'].state['signature']['spec']['layers']] +
                     [caches['bert']['train'].state['signature']['spec']['model'], caches['roberta']['train'].state['signature']['spec']['model']],
        chunk_examples=chunks, result_dir=str(target), history=history)
    if config.get('power_watts') is not None:
        result['estimated_gpu_kwh'] = config['power_watts'] * (result['extraction_seconds'] + elapsed + inference_seconds) / 3_600_000
        result['energy_method'] = f'assumed total GPU power={config["power_watts"]} W; not measured'
    else:
        result.update(estimated_gpu_kwh=None, energy_method='not measured')
    with (target / 'predictions.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'label', 'prediction'])
        writer.writerows((r['id'], r['label'], p) for r, p in zip(data['splits']['test'], predictions))
    write_json(result_path, result)
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result
