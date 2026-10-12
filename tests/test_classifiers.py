"""Checks for information boundaries and executable ensemble behavior."""
import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from compare_classifiers import (fit_temperature, hard_vote, prefix_text, probabilities,
                                 tune_weights, validation_groups, vectorizer, weighted_average)
from experiment import tokens
from extended_experiments import load_rows


class ClassifierTests(unittest.TestCase):
    def test_prefix_and_training_only_features(self):
        row = {'LLM_output': 'Gemma\n\t' + 'known ' * 140 + 'AFTER_BUDGET'}
        text = prefix_text(row)
        self.assertEqual(text.split(), tokens(row['LLM_output'])[:128])
        self.assertNotIn('Gemma', text)
        self.assertNotIn('AFTER_BUDGET', text)
        for rep in ('word', 'char', 'word_char'):
            v = vectorizer(rep)
            x = v.fit_transform(['alpha beta !', 'alpha beta !', 'beta gamma ?'])
            names = list(v.get_feature_names_out())
            z = v.transform(['UNSEEN_SENTINEL'])
            self.assertEqual(x.shape[1], z.shape[1])
            self.assertEqual(names, list(v.get_feature_names_out()))
            self.assertFalse(any('SENTINEL' in name for name in names))

    @unittest.skipUnless(
        (ROOT / "data/processed/dataset.jsonl").is_file()
        and (ROOT / "results/initial_v1/cnn_output/config.json").is_file(),
        "Requires local dataset and initial-run configuration, not included in this submission")
    def test_grouped_validation_partition(self):
        rows = load_rows()
        groups = validation_groups(rows)
        self.assertEqual(len(groups['calibration']), 62)
        self.assertEqual(len(groups['selection']), 62)
        self.assertFalse(set(groups['calibration']) & set(groups['selection']))
        self.assertEqual(set(groups['calibration'] + groups['selection']),
                         {r['prompt_group'] for r in rows if r['split'] == 'validation'})
        self.assertFalse(set(groups['selection']) & {r['prompt_group'] for r in rows if r['split'] != 'validation'})

    def test_temperature_preserves_class_and_probability_sum(self):
        scores=np.array([[9,1,-2],[-1,3,0],[2,4,8],[4,3,-1],[1,7,0],[1,-2,5]],dtype=float)
        y=np.array([0,1,2,1,1,0])
        t=fit_temperature(scores,y)
        p=probabilities(scores,t)
        self.assertTrue(.049 < t < 20.001)
        np.testing.assert_array_equal(scores.argmax(axis=1),p.argmax(axis=1))
        np.testing.assert_allclose(p.sum(axis=1),1)
        with self.assertRaises(ValueError): probabilities(scores,0)

    def test_weight_search_can_reject_harmful_member(self):
        a=np.array([[.95,.03,.02],[.03,.95,.02],[.03,.02,.95]])
        b=np.roll(a,1,axis=1)
        weights,trials=tune_weights([a,b],np.arange(3))
        self.assertEqual(weights,[1.,0.])
        self.assertEqual(len(trials),5)
        np.testing.assert_allclose(weighted_average([a,b],weights),a)
        with self.assertRaises(ValueError): weighted_average([a,b],[.8,.8])

    def test_hard_vote_majority_and_label_free_tie(self):
        eye=np.eye(3)
        a,b,c=eye[[2,0]],eye[[0,1]],eye[[1,1]]
        np.testing.assert_array_equal(hard_vote([a,b,c]).argmax(axis=1),[2,1])


if __name__=='__main__':
    unittest.main()
