import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmembed.data import split_indices, normalize_records, prepare_data
import torch
from llmembed.models import BaselineFusion, AdaptiveFusion, ChunkPool, ExperimentModel
from llmembed.features import token_pool, window_ranges, make_windows, FeatureStore
from llmembed.run import main, parser, experiment_matrix, extraction_spec

torch.set_num_threads(1)


class DataTests(unittest.TestCase):
    def test_split_preserves_rare_classes_and_never_overlaps(self):
        labels = [0] * 20 + [1] * 3 + [2]
        train, val = split_indices(labels, 0.2, 42)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(sorted(train + val), list(range(len(labels))))
        self.assertEqual(set(np.array(labels)[train]), {0, 1, 2})
        self.assertEqual((train, val), split_indices(labels, 0.2, 42))

    def test_bad_local_data_fails_before_extraction(self):
        with self.assertRaises(ValueError):
            normalize_records({'text': ['hello'], 'label': [0, 1]})
        with self.assertRaises(ValueError):
            normalize_records([{'text': ' ', 'label': 0}])
        with self.assertRaises(ValueError):
            normalize_records([{'text': 'hello', 'label': -1}])
        with self.assertRaises(ValueError):
            normalize_records({'text': 'abc', 'label': [0, 1, 0]})

    def test_local_data_shared_label_map_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input' / 'r8'
            source.mkdir(parents=True)
            rows = [{'text': f'doc {i}', 'label': 'b' if i % 2 else 'a'} for i in range(20)]
            (source / 'train.json').write_text(json.dumps(rows))
            (source / 'test.json').write_text(json.dumps(rows[:2]))
            first = prepare_data('r8', root / 'input', root / 'out', limit=0)
            second = prepare_data('r8', root / 'input', root / 'out', limit=0)
            self.assertEqual(first['fingerprint'], second['fingerprint'])
            self.assertEqual(first['label_names'], ['a', 'b'])
            import shutil
            shutil.copytree(root / 'input', root / 'moved')
            moved = prepare_data('r8', root / 'moved', root / 'out', limit=0)
            self.assertEqual(first['fingerprint'], moved['fingerprint'])
            self.assertEqual(len(first['splits']['test']), 2)
            self.assertFalse(set(r['id'] for r in first['splits']['train']) &
                             set(r['id'] for r in first['splits']['val']))
            rows[0]['text'] = 'changed document'
            (source / 'train.json').write_text(json.dumps(rows))
            third = prepare_data('r8', root / 'input', root / 'out', limit=0)
            self.assertNotEqual(first['fingerprint'], third['fingerprint'])


class ModelTests(unittest.TestCase):
    def test_all_fifteen_fusions_and_adaptive_backward(self):
        inputs = (torch.randn(3, 5, 12), torch.randn(3, 8), torch.randn(3, 8))
        for strategy in range(1, 16):
            model = BaselineFusion(4, (12, 8, 8), strategy=strategy, hidden=8)
            logits, _ = model(*inputs)
            self.assertEqual(logits.shape, (3, 4))
            torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2])).backward()
        model = AdaptiveFusion(4, (12, 8, 6), hidden=8, heads=2)
        logits, gate = model(inputs[0], inputs[1], torch.randn(3, 6))
        torch.testing.assert_close(gate.sum(1), torch.ones(3))
        logits.sum().backward()
        self.assertIsNotNone(model.projections[0][0].weight.grad)

    def test_cooccurrence_matches_author_normalization(self):
        model = BaselineFusion(2, (12, 8, 8), strategy=15, hidden=8).eval()
        l, b, r = torch.randn(2, 5, 12), torch.randn(2, 8), torch.randn(2, 8)
        sources = torch.stack([model.compress_layers[i](l[:, i]) for i in range(5)] + [b, r], 1)
        expected = torch.cat([2 * torch.sigmoid(0.1 * (sources @ sources.transpose(1, 2)).flatten(1)) - 1,
                              l.mean(1)], 1)
        torch.testing.assert_close(model.fuse(l, b, r), expected)

    def test_chunk_padding_has_no_effect_and_learns(self):
        pool = ChunkPool(8)
        x = torch.randn(2, 4, 1, 8)
        mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
        result, weights = pool(x, mask)
        modified = x.clone()
        modified[~mask] = 100000
        other, _ = pool(modified, mask)
        torch.testing.assert_close(result, other)
        torch.testing.assert_close(weights.sum(1), torch.ones(2, 1))
        self.assertEqual(weights[~mask].sum().item(), 0)
        result.square().sum().backward()
        self.assertIsNotNone(pool.score.weight.grad)


class FeatureTests(unittest.TestCase):
    def test_truncation_preserves_prompt_anchor_and_content_only_mask(self):
        from transformers import BertTokenizerFast
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'vocab.txt'
            path.write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'film', ':', '"', '.']))
            tokenizer = BertTokenizerFast(vocab_file=str(path))
            spec = dict(pooling='cot', task='sst2', max_length=48, chunked=False,
                        chunk_size=16, overlap=2, max_chunks=0)
            entries, original, kept = make_windows(tokenizer, 'film ' * 200, spec)
            entry = entries[0]
            self.assertEqual(len(entry['input_ids']), 48)
            self.assertLess(kept, original)
            self.assertEqual(sum(entry['content_mask']), kept)
            self.assertEqual(entry['input_ids'][entry['anchor']], tokenizer.convert_tokens_to_ids(':'))
            self.assertFalse(entry['content_mask'][entry['anchor']])
            entries, original, kept = make_windows(tokenizer, 'film ' * 200, dict(spec, chunked=True))
            self.assertEqual(original, kept)
            self.assertTrue(all(len(e['input_ids']) <= 48 for e in entries))

    def test_windows_cover_every_token_including_tail(self):
        for n in (1, 224, 256, 257, 448, 2470):
            intervals = window_ranges(n, 256, 32)
            covered = {t for a, b in intervals for t in range(a, b)}
            self.assertEqual(covered, set(range(n)))
            self.assertTrue(all(0 < b - a <= 256 for a, b in intervals))
        with self.assertRaises(ValueError):
            window_ranges(50, 32, 32)

    def test_pool_masks_prompt_padding_and_anchor(self):
        h = torch.tensor([[[100., 100.], [1., 0.], [0., 1.], [1., 1.], [999., 999.]]])
        attention = torch.tensor([[1, 1, 1, 1, 0]])
        content = torch.tensor([[0, 1, 1, 0, 0]], dtype=torch.bool)
        pooled, weights = token_pool(h, attention, content, torch.tensor([3]), 'cot', 0.5, 0.1)
        torch.testing.assert_close(weights, torch.tensor([[0., .5, .5, 0., 0.]]))
        torch.testing.assert_close(pooled, torch.tensor([[.75, .75]]))
        mean, _ = token_pool(h, attention, content, torch.tensor([3]), 'mean', 0.5, 0.1)
        torch.testing.assert_close(mean, h[:, :4].mean(1))
        with self.assertRaises(ValueError):
            token_pool(h, attention, content, torch.tensor([3]), 'cot', .5, 0)

    def test_feature_store_recovers_uncommitted_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FeatureStore(Path(directory), {'key': 'same'}, (5, 4), 2)
            store.append([np.ones((2, 5, 4), dtype=np.float16)], 1.0, 30, 30, [])
            with (Path(directory) / 'values.bin').open('ab') as f:
                f.write(b'interrupted write')
            resumed = FeatureStore(Path(directory), {'key': 'same'}, (5, 4), 2)
            resumed.append([np.zeros((1, 5, 4), dtype=np.float16)], 2.0, 10, 8, [])
            resumed.finish()
            self.assertEqual(resumed[0].shape, (2, 5, 4))
            self.assertEqual(resumed[1].sum(), 0)
            self.assertEqual(resumed.state['original_tokens'], 40)
            self.assertEqual(resumed.state['retained_tokens'], 38)
            with self.assertRaises(ValueError):
                FeatureStore(Path(directory), {'key': 'different'}, (5, 4), 2)
            resumed.close()


class PipelineTests(unittest.TestCase):
    def test_experiments_do_not_confound_exp2_and_exp3(self):
        args = parser().parse_args([])
        exps = {e['name']: e for e in experiment_matrix(args)}
        self.assertTrue(exps['exp2']['cot'])
        self.assertFalse(exps['exp2']['chunked'])
        self.assertTrue(exps['exp3']['chunked'])
        self.assertFalse(exps['exp3']['adaptive'])
        self.assertTrue(exps['exp4']['adaptive'])
        self.assertEqual(extraction_spec(args, 'sst2', exps['exp3'], 'llama'),
                         extraction_spec(args, 'sst2', exps['exp4'], 'llama'))
        self.assertEqual(extraction_spec(args, 'sst2', exps['exp0'], 'bert'),
                         extraction_spec(args, 'sst2', exps['exp2'], 'bert'))

    def test_offline_all_experiments_resume_and_exports(self):
        from transformers import BertConfig, BertModel, BertTokenizerFast, LlamaConfig, LlamaModel
        from llmembed import train
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'good', 'bad', 'film', 'news', 'text', ':', '"', '.']
            vocab_path = root / 'vocab.txt'
            vocab_path.write_text('\n'.join(vocab), encoding='utf-8')
            tokenizer = BertTokenizerFast(vocab_file=str(vocab_path))
            models = {}
            for name, dim, depth in [('llama', 16, 32), ('bert', 8, 2), ('roberta', 8, 2),
                                     ('deberta', 12, 2), ('next', 12, 28)]:
                path = root / name
                if name in ('llama', 'next'):
                    model = LlamaModel(LlamaConfig(vocab_size=len(vocab), hidden_size=dim, intermediate_size=dim * 2,
                        num_hidden_layers=depth, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128))
                else:
                    model = BertModel(BertConfig(vocab_size=len(vocab), hidden_size=dim, intermediate_size=dim * 2,
                        num_hidden_layers=depth, num_attention_heads=2, max_position_embeddings=128))
                model.save_pretrained(path)
                tokenizer.save_pretrained(path)
                models[name] = path
            source = root / 'data' / 'sst2'
            source.mkdir(parents=True)
            for split, count in [('train', 16), ('test', 4)]:
                rows = [dict(text=(('good' if i % 2 else 'bad') + ' film ') * (i + 2), label=i % 2) for i in range(count)]
                (source / f'{split}.json').write_text(json.dumps(rows))
            args = ['--data-dir', str(root / 'data'), '--output', str(root / 'run'), '--datasets', 'sst2',
                '--experiments', 'exp0', 'exp1', 'exp2', 'exp3', 'exp4', 'exp5b', 'exp5c', '--precision', 'fp32',
                '--seeds', '7', '--epochs', '2', '--hidden', '16', '--heads', '2', '--micro-batch', '2',
                '--batch-size', '4', '--chunk-size', '8', '--overlap', '2']
            for name, path in models.items():
                args.extend([f'--{name}-model', str(path)])
            original_save = train.save_checkpoint

            def interrupt_after_first_epoch(path, content):
                original_save(path, content)
                if path.name == 'last.pt' and content['epoch'] == 1:
                    raise RuntimeError('simulated session interruption')

            with patch.object(train, 'save_checkpoint', side_effect=interrupt_after_first_epoch):
                with self.assertRaisesRegex(RuntimeError, 'simulated session'):
                    main(args)
            results = main(args)
            self.assertEqual(len(results), 7)
            self.assertEqual(len(results[0]['history']), 2)
            for result in results:
                self.assertTrue(0 <= result['metrics']['accuracy'] <= 1)
                self.assertEqual(result['estimated_gpu_kwh'], None)
            self.assertEqual(results[4]['retention_by_source'], dict(llama=1., bert=1., roberta=1.))
            self.assertTrue((root / 'run' / 'metrics.csv').exists())
            self.assertTrue((Path(results[4]['result_dir']) / 'source_weights.png').exists())
            self.assertTrue(list((root / 'run' / 'explanations').glob('*/tokens_0.png')))
            self.assertFalse((root / 'run' / 'average_five_datasets.csv').exists())
            states = list((root / 'run' / 'features').glob('*/state.json'))
            before = {str(p): p.stat().st_mtime_ns for p in states}
            again = main(args + ['--stage', 'train'])
            self.assertEqual(before, {str(p): p.stat().st_mtime_ns for p in states})
            self.assertEqual(results[0]['metrics'], again[0]['metrics'])
            clean = main(args + ['--output', str(root / 'clean'), '--experiments', 'exp0',
                                 '--reuse-cache', str(root / 'run' / 'features')])
            resumed = torch.load(Path(results[0]['result_dir']) / 'last.pt', map_location='cpu', weights_only=True)
            uninterrupted = torch.load(Path(clean[0]['result_dir']) / 'last.pt', map_location='cpu', weights_only=True)
            for key, tensor in resumed['model'].items():
                torch.testing.assert_close(tensor, uninterrupted['model'][key])


if __name__ == '__main__':
    unittest.main()
