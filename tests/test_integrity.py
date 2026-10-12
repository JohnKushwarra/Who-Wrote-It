"""Meaningful guards against split leakage, lost responses and padding bugs."""
import json
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from experiment import LABELS, build_vocab, collate, encode, validate_data
from models import TextCNN, TextLSTM
from prepare_data import split_groups


class IntegrityTests(unittest.TestCase):
    def test_group_splits_are_disjoint_and_repeatable(self):
        ids = [f"p{i}" for i in range(100)]
        first = split_groups(ids, 448)
        self.assertEqual(first, split_groups(list(reversed(ids)), 448))
        self.assertEqual([list(first.values()).count(s) for s in ("train", "validation", "test")], [70, 15, 15])

    def test_vocabulary_cannot_see_held_out_text(self):
        vocab = build_vocab([{"LLM_Input": "training prompt", "LLM_output": "training answer"}], 100)
        self.assertNotIn("unseenword", vocab)
        self.assertEqual(encode({"LLM_Input": "unseenword"}, "input", vocab, 10), [1])

    def test_combined_view_keeps_response_budget(self):
        row = {"LLM_Input": "prompt " * 50, "LLM_output": "answer " * 50}
        vocab = build_vocab([row], 100)
        combined = encode(row, "both", vocab, 8)
        self.assertEqual(len(combined), 17)
        self.assertEqual(combined[9:], encode(row, "output", vocab, 8))

    def test_padding_does_not_change_logits(self):
        torch.manual_seed(10)
        x, lengths, _ = collate([([3,4,5,6,7,8],0), ([3,4,5],1)])
        larger = torch.nn.functional.pad(x, (0,11))
        for cls in (TextCNN, TextLSTM):
            with self.subTest(model=cls.__name__):
                model = cls(20, embedding_dim=8, hidden_dim=8).eval()
                with torch.no_grad():
                    self.assertTrue(torch.allclose(model(x,lengths),model(larger,lengths),atol=1e-6))

    @unittest.skipUnless((ROOT / "data/processed/dataset.jsonl").is_file(),
                         "Requires the dataset, which is not included in this submission")
    def test_actual_dataset_integrity(self):
        rows = [json.loads(line) for line in (ROOT / "data/processed/dataset.jsonl").read_text(encoding="utf-8").splitlines()]
        validate_data(rows)
        for split in ("train", "validation", "test"):
            counts = [sum(r["LLM_name"] == label and r["split"] == split for r in rows) for label in LABELS]
            self.assertEqual(len(set(counts)), 1)
        modified = [dict(r) for r in rows]
        modified[0]["split"] = "validation" if modified[0]["split"] != "validation" else "test"
        with self.assertRaises(ValueError):
            validate_data(modified)


if __name__ == "__main__":
    unittest.main()
