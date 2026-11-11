import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmembed.data import split_indices, normalize_records, prepare_data


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
            self.assertEqual(len(first['splits']['test']), 2)
            self.assertFalse(set(r['id'] for r in first['splits']['train']) &
                             set(r['id'] for r in first['splits']['val']))
            rows[0]['text'] = 'changed document'
            (source / 'train.json').write_text(json.dumps(rows))
            third = prepare_data('r8', root / 'input', root / 'out', limit=0)
            self.assertNotEqual(first['fingerprint'], third['fingerprint'])


if __name__ == '__main__':
    unittest.main()
