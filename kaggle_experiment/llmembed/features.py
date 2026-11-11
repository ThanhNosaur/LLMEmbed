"""Masked extraction and append-only, resumable float16 feature caches."""
import gc
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import digest, write_json


def window_ranges(length, size, overlap, max_chunks=0):
    if length < 1 or size < 1 or not 0 <= overlap < size or max_chunks < 0:
        raise ValueError('Require nonempty text, size > overlap >= 0 and max_chunks >= 0.')
    ranges, start = [], 0
    while start < length:
        end = min(start + size, length)
        ranges.append((start, end))
        if end == length:
            break
        start = end - overlap
    if max_chunks and len(ranges) > max_chunks:
        ranges = [ranges[i] for i in np.linspace(0, len(ranges) - 1, max_chunks, dtype=int)]
    return ranges


def make_windows(tokenizer, text, spec):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError('Text tokenized to an empty sequence.')
    prefix, suffix = [], []
    if spec['pooling'] == 'cot':
        sentiment = spec['task'] in ('sst2', 'mr')
        prefix = tokenizer.encode('Text: "' if sentiment else 'Article: "', add_special_tokens=False)
        suffix = tokenizer.encode('". After analyzing the key sentiment clues step by step, the overall sentiment of this text in one word is:' if sentiment else
                                  '". After identifying the core domain clues and reasoning step by step, the topic category of this article in one word is:', add_special_tokens=False)
    budget = spec['max_length'] - len(prefix) - len(suffix) - tokenizer.num_special_tokens_to_add(pair=False)
    if budget < 1:
        raise ValueError('max_length cannot fit the prompt and one content token.')
    ranges = (window_ranges(len(ids), min(spec['chunk_size'], budget), spec['overlap'], spec['max_chunks'])
              if spec['chunked'] else [(0, min(len(ids), budget))])
    entries = []
    for start, end in ranges:
        body = prefix + ids[start:end] + suffix
        full = tokenizer.build_inputs_with_special_tokens(body)
        # Supported single-sequence tokenizers add specials only at the boundaries.
        offset = next((i for i in range(len(full) - len(body) + 1) if full[i:i + len(body)] == body), None)
        if offset is None:
            raise ValueError('Tokenizer inserted special tokens inside the input sequence.')
        mask = [False] * len(full)
        mask[offset + len(prefix):offset + len(prefix) + end - start] = [True] * (end - start)
        entries.append(dict(input_ids=full, content_mask=mask, anchor=offset + len(body) - 1,
                            start=start, end=end))
    retained, previous = 0, 0
    for start, end in ranges:
        retained += max(0, end - max(start, previous))
        previous = end
    return entries, len(ids), retained


def token_pool(hidden, attention, content, anchors, mode, gamma=0.5, tau=0.1):
    if tau <= 0 or not 0 <= gamma <= 1:
        raise ValueError('Require tau > 0 and 0 <= gamma <= 1.')
    hidden = hidden.float()
    attention = attention.to(hidden.device).bool()
    if mode == 'legacy_mean':
        return hidden.mean(1), None
    if mode == 'mean':
        return (hidden * attention.unsqueeze(-1)).sum(1) / attention.sum(1, keepdim=True).clamp_min(1), None
    if mode != 'cot':
        raise ValueError(f'Unknown pooling: {mode}')
    content = content.to(hidden.device).bool() & attention
    if not content.any(1).all():
        raise ValueError('Attention pooling needs at least one original-text token.')
    anchor = hidden[torch.arange(len(hidden), device=hidden.device), anchors.to(hidden.device)]
    scores = (F.normalize(hidden, dim=-1) * F.normalize(anchor, dim=-1).unsqueeze(1)).sum(-1) / tau
    weights = scores.masked_fill(~content, -torch.inf).softmax(1)
    return gamma * anchor + (1 - gamma) * (weights.unsqueeze(-1) * hidden).sum(1), weights


class FeatureStore:
    def __init__(self, path, signature, shape, documents):
        self.path, self.shape = Path(path), tuple(shape)
        self.path.mkdir(parents=True, exist_ok=True)
        self.binary, self.manifest = self.path / 'values.bin', self.path / 'state.json'
        self.array = None
        if self.manifest.exists():
            self.state = json.loads(self.manifest.read_text(encoding='utf-8'))
            if self.state['signature'] != signature or self.state['shape'] != list(shape) or self.state['documents'] != documents:
                raise ValueError('Cache signature mismatch; select another output directory.')
        else:
            self.state = dict(signature=signature, shape=list(shape), documents=documents, offsets=[0],
                              seconds=0.0, original_tokens=0, retained_tokens=0, explanations=[], complete=False)
        expected = self.state['offsets'][-1] * math.prod(shape) * 2
        actual = self.binary.stat().st_size if self.binary.exists() else 0
        if actual < expected or (self.state['complete'] and actual != expected):
            raise ValueError(f'Truncated/corrupt feature cache: {self.path}')
        if actual > expected and not self.state['complete']:
            with self.binary.open('r+b') as stream:
                stream.truncate(expected)

    @property
    def done(self):
        return len(self.state['offsets']) - 1

    def append(self, documents, seconds, original, retained, explanations):
        if self.state['complete'] or self.done + len(documents) > self.state['documents']:
            raise ValueError('Cannot append beyond the planned document count.')
        arrays = [np.asarray(a, dtype=np.float16) for a in documents]
        if any(a.ndim != 3 or a.shape[1:] != self.shape or len(a) == 0 or not np.isfinite(a).all() for a in arrays):
            raise ValueError('Invalid/non-finite extracted feature array.')
        with self.binary.open('ab') as stream:
            for array in arrays:
                array.tofile(stream)
                self.state['offsets'].append(self.state['offsets'][-1] + len(array))
            stream.flush()
            os.fsync(stream.fileno())
        self.state['seconds'] += seconds
        self.state['original_tokens'] += original
        self.state['retained_tokens'] += retained
        self.state['explanations'].extend(explanations)
        write_json(self.manifest, self.state)

    def finish(self):
        if self.done != self.state['documents']:
            raise ValueError('Cannot finish an incomplete feature cache.')
        self.state['complete'] = True
        write_json(self.manifest, self.state)

    def __getitem__(self, index):
        if not self.state['complete']:
            raise ValueError('Training requires a completed feature cache.')
        if self.array is None:
            self.array = np.memmap(self.binary, dtype=np.float16, mode='r',
                                   shape=(self.state['offsets'][-1], *self.shape))
        return self.array[self.state['offsets'][index]:self.state['offsets'][index + 1]]

    def close(self):
        self.array = None


def encode_batch(model, tokenizer, entries, spec):
    length = spec['max_length'] if spec['pooling'] == 'legacy_mean' else max(len(e['input_ids']) for e in entries)
    ids = torch.full((len(entries), length), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    content = torch.zeros_like(ids, dtype=torch.bool)
    for i, entry in enumerate(entries):
        n = len(entry['input_ids'])
        ids[i, :n] = torch.tensor(entry['input_ids'])
        attention[i, :n] = 1
        content[i, :n] = torch.tensor(entry['content_mask'])
    device = model.get_input_embeddings().weight.device
    with torch.inference_mode():
        kwargs = dict(input_ids=ids.to(device), attention_mask=attention.to(device), return_dict=True)
        if spec['source'] == 'llama':
            kwargs.update(output_hidden_states=True, use_cache=False)
        outputs = model(**kwargs)
        if spec['source'] != 'llama':
            values = outputs.pooler_output if spec['pooling'] == 'pooler' else outputs.last_hidden_state[:, 0]
            return values[:, None].float().cpu().numpy(), None
        anchors = torch.tensor([e['anchor'] for e in entries])
        features, alpha = [], None
        for depth in spec['layers']:
            value, weights = token_pool(outputs.hidden_states[depth], attention, content, anchors,
                                        spec['pooling'], spec['gamma'], spec['tau'])
            features.append(value.cpu())
            if alpha is None and weights is not None:
                alpha = weights.cpu().numpy()
        return torch.stack(features, 1).numpy(), alpha


def extract_rows(model, tokenizer, rows, spec, store, batch_size=2, shard_docs=32, explain=4):
    while store.done < len(rows):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        start = store.done
        selected = rows[start:start + shard_docs]
        entries, counts, original, retained = [], [], 0, 0
        for row in selected:
            windows, n, kept = make_windows(tokenizer, row['text'], spec)
            entries.extend(windows)
            counts.append(len(windows))
            original, retained = original + n, retained + kept
        vectors, alphas = [], []
        cursor = 0
        while cursor < len(entries):
            try:
                values, alpha = encode_batch(model, tokenizer, entries[cursor:cursor + batch_size], spec)
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                print(f'CUDA OOM: retry extraction with batch_size={batch_size}', flush=True)
                continue
            vectors.extend(values)
            alphas.extend(alpha if alpha is not None else [None] * len(values))
            cursor += len(values)
        documents, explanations, cursor = [], [], 0
        for i, (row, count) in enumerate(zip(selected, counts)):
            documents.append(np.stack(vectors[cursor:cursor + count]))
            if start + i < explain and spec['source'] == 'llama' and spec['pooling'] == 'cot':
                for j in range(count):
                    entry, alpha = entries[cursor + j], alphas[cursor + j]
                    positions = [k for k, yes in enumerate(entry['content_mask']) if yes]
                    explanations.append(dict(id=row['id'], chunk=j, depth=spec['layers'][0],
                        token_start=entry['start'], tokens=tokenizer.convert_ids_to_tokens([entry['input_ids'][k] for k in positions]),
                        weights=[float(alpha[k]) for k in positions]))
            cursor += count
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        store.append(documents, time.perf_counter() - started, original, retained, explanations)
        print(f'{spec["source"]}: cached {store.done}/{len(rows)} documents', flush=True)
    store.finish()


def extract_source(data, source, spec, cache_dir, reuse_cache=None, batch_size=2, read_only=False):
    from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig, __version__
    config = AutoConfig.from_pretrained(spec['model'], revision=spec['revision'], trust_remote_code=False)
    spec = dict(spec, source=source, model_commit=getattr(config, '_commit_hash', None), transformers=__version__,
                feature_format=1, extraction_seed=42)
    if source == 'llama':
        spec['layers'] = [config.num_hidden_layers - i * spec['stride'] for i in range(5)]
        if min(spec['layers']) < 1:
            raise ValueError('Backbone is too shallow for five layers at this stride.')
    else:
        spec['layers'] = [config.num_hidden_layers]
    context = min(getattr(config, 'max_position_embeddings', 512), 512) if source != 'llama' else config.max_position_embeddings
    # RoBERTa reserves two positional indices for special/padding handling.
    if config.model_type == 'roberta':
        context = min(context, config.max_position_embeddings - 2)
    spec['max_length'] = min(spec['max_length'], context)
    stores = {}
    for split, rows in data['splits'].items():
        if not rows:
            continue
        signature = dict(spec=spec, data=data['fingerprint'], split=split)
        key = digest(signature)[:24]
        path = Path(cache_dir) / key
        if reuse_cache and (Path(reuse_cache) / key / 'state.json').exists():
            saved = json.loads((Path(reuse_cache) / key / 'state.json').read_text())
            if saved.get('complete'):
                path = Path(reuse_cache) / key
        stores[split] = FeatureStore(path, signature, (len(spec['layers']), config.hidden_size), len(rows))
    if all(store.state['complete'] for store in stores.values()):
        return stores
    if read_only:
        raise FileNotFoundError('Complete embeddings missing. Run --stage extract (with the same configuration) first.')
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(spec['model'], revision=spec['revision'], trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError('Tokenizer must define a pad or EOS token.')
    tokenizer.padding_side = 'right'
    gpu = torch.cuda.is_available()
    kwargs = dict(config=config, revision=spec['revision'], trust_remote_code=False,
                  torch_dtype=torch.float16 if gpu and spec['precision'] != 'fp32' else torch.float32)
    if source == 'llama':
        kwargs['attn_implementation'] = 'sdpa'
        if spec['precision'] == 'nf4':
            if not gpu:
                raise ValueError('NF4 extraction needs CUDA; use fp32 for CPU smoke tests.')
            kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        if gpu:
            kwargs['device_map'] = 'auto'
            kwargs['max_memory'] = {i: int(torch.cuda.get_device_properties(i).total_memory * 0.80)
                                    for i in range(torch.cuda.device_count())}
    model = AutoModel.from_pretrained(spec['model'], **kwargs).eval()
    if source != 'llama' and gpu:
        model.to('cuda:0')
    model.requires_grad_(False)
    try:
        for split, store in stores.items():
            if not store.state['complete']:
                extract_rows(model, tokenizer, data['splits'][split], spec, store, batch_size,
                             explain=4 if split == 'test' else 0)
    finally:
        del model
        gc.collect()
        if gpu:
            torch.cuda.empty_cache()
    return stores
