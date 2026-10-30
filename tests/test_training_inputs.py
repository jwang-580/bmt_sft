"""Synthetic input validation tests; the training dataset is not opened."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fine_tuning.train_gemma4_12b_unsloth import parse_args, validate_inputs
from prompt_eval.extraction_prompts import SFT_OUTPUT_FIELDS


class TrainingInputTests(unittest.TestCase):
    def test_external_telemetry_is_opt_in(self):
        with patch('sys.argv',['train']):
            self.assertEqual(parse_args().report_to,'none')

    def test_mixed_assistant_identities_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            (path/'reserved.json').write_text(json.dumps([f'synthetic-held-{i}' for i in range(20)]),encoding='utf-8')
            first={k:'synthetic' for k in SFT_OUTPUT_FIELDS}
            first.update(record_id='synthetic-train-1',field_id='1',answers='No')
            second={**first,'record_id':'synthetic-held-0','field_id':'2'}
            row={'messages':[{'role':'system','content':'Synthetic instruction'},
                             {'role':'user','content':'Synthetic input'},
                             {'role':'assistant','content':json.dumps([first,second])}]}
            (path/'train.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Mixed admission identities'):
                validate_inputs(path/'train.jsonl',path/'reserved.json')


if __name__=='__main__': unittest.main()
