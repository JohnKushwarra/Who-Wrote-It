import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from experiment import build_vocab, encode, tokens
from extended_experiments import load_rows, prefix_ids, select_rq3, task_label


class ExtensionTests(unittest.TestCase):
    def test_format_example_does_not_define_task(self):
        prompt="Answer in accordance with the format: title such as <<poem of joy>>.\n\nWhat causes rain?"
        self.assertEqual(task_label(prompt),"question_answering")
        self.assertEqual(task_label("Compose a poem about rain."),"drafting")
        self.assertEqual(task_label("How Road Infrastructure Shapes Our Lives: write an article."),"other")
        self.assertEqual(task_label("How about generating a poem?"),"other")

    @unittest.skipUnless(
        (ROOT / "data/processed/dataset.jsonl").is_file()
        and (ROOT / "results/initial_v1/cnn_output/config.json").is_file(),
        "Requires local dataset and initial-run configuration, not included in this submission")
    def test_rq3_equal_sizes_and_target_isolation(self):
        partitions,_=select_rq3(load_rows())
        tasks=list(partitions)
        for split in ("train","validation"):
            self.assertEqual(len(partitions[tasks[0]][split]),len(partitions[tasks[1]][split]))
        all_sets=[]
        for task in tasks:
            for split in ("train","validation","test"):
                current={r["prompt_group"] for r in partitions[task][split]}
                self.assertTrue(all(not (current & other) for other in all_sets))
                all_sets.append(current)
        source=partitions[tasks[0]]["train"]
        vocab=build_vocab(source,12000)
        source_tokens={t for r in source for f in ("LLM_Input","LLM_output") for t in tokens(r[f])}
        self.assertTrue(set(vocab)-{"<pad>","<unk>","<sep>"} <= source_tokens)

    def test_ablations_preserve_original_prefix_and_length(self):
        row={"LLM_Input":"Question?","LLM_output":"Hello, WORLD! "+"word "*150,"prompt_group":"synthetic-group"}
        vocab=build_vocab([row],100)
        baseline=prefix_ids(row,vocab)
        self.assertEqual(baseline,encode(row,"output",vocab,128))
        self.assertEqual(len(baseline),128)
        punctuation=prefix_ids(row,vocab,"punctuation_masked")
        words=prefix_ids(row,vocab,"words_masked")
        self.assertEqual([punctuation[i] for i in [0,2]],[baseline[i] for i in [0,2]])
        self.assertEqual([punctuation[i] for i in [1,3]],[1,1])
        self.assertEqual([words[i] for i in [0,2]],[1,1])
        self.assertEqual([words[i] for i in [1,3]],[baseline[i] for i in [1,3]])
        for condition in ("punctuation_masked","words_masked","order_shuffled"):
            changed=prefix_ids(row,vocab,condition)
            self.assertEqual(len(changed),len(baseline))
            self.assertNotIn(0,changed)
        self.assertCountEqual(prefix_ids(row,vocab,"order_shuffled"),baseline)
        self.assertEqual(prefix_ids(row,vocab,"order_shuffled"),prefix_ids(row,vocab,"order_shuffled"))


if __name__=="__main__":
    unittest.main()
