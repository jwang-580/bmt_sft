"""Synthetic registry regression tests; no patient data or model downloads."""
import importlib
import json
import unittest
from unittest.mock import patch

from prompt_eval import registry_contract as contract
from prompt_eval.score_predictions import score_group, flatten_predictions

NOTE = {'record_id':'synthetic-001','record_id_type':'synthetic','date':'2000-01-01'}
META = {str(i):{'field_category':'synthetic','field_name':f'Synthetic question {i}',
               'field_instructions':"['Yes', 'No']"} for i in (1,2)}
BACKENDS = ['evaluate_gemma4_extraction','evaluate_qwen35_extraction',
            'evaluate_gptoss_extraction','evaluate_gptoss_extraction_unsloth']


class ContractTests(unittest.TestCase):
    def coerce(self, rows):
        return contract.coerce_records(rows, NOTE, ['1','2'], META, 'baseline')

    def test_missing_field_is_not_a_successful_na(self):
        rows=self.coerce([{'field_id':'1','answers':'N/A'}])
        self.assertTrue(rows[0]['extraction_success'])
        self.assertIsNone(rows[1]['answers'])
        self.assertFalse(rows[1]['extraction_success'])
        with self.assertRaises(ValueError): contract.require_complete_predictions(rows)

    def test_null_blank_and_missing_answer_fail(self):
        for raw in ({'field_id':'1'}, {'field_id':'1','answers':None}, {'field_id':'1','answers':''},
                    {'field_id':'1','answers':[]}, {'field_id':'1','answers':{'value':'No'}}):
            with self.subTest(raw=raw): self.assertFalse(self.coerce([raw])[0]['extraction_success'])

    def test_duplicate_unknown_and_conflicting_fields_rejected(self):
        cases=[[{'field_id':'1','answers':'No'}]*2,
               [{'field_id':'999','answers':'No'}],
               [{'field_id':'1','question_id':'2','answers':'No'}],
               [{'answers':'No'}]]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError): self.coerce(rows)

    def test_reordered_fields_keep_binding_and_trusted_metadata(self):
        rows=self.coerce([{'field_id':'2','answers':'Yes','field_name':'untrusted'},
                          {'field_id':'1','answers':'No'}])
        self.assertEqual([r['answers'] for r in rows],['No','Yes'])
        self.assertEqual(rows[1]['field_name'],META['2']['field_name'])

    def test_unique_field_name_can_bind(self):
        rows=self.coerce([{'field_name':META['1']['field_name'],'answers':'No'}])
        self.assertEqual(rows[0]['answers'],'No')

    def test_plain_text_missing_and_duplicates(self):
        raw=contract.parse_non_json_generation('Field 1: No',['1','2'])
        self.assertEqual(len(raw),1)
        self.assertFalse(self.coerce(raw)[1]['extraction_success'])
        with self.assertRaises(ValueError):
            contract.parse_non_json_generation('Field 1: Yes\nField 1: No',['1','2'])

    def test_nan_is_not_a_valid_answer(self):
        self.assertFalse(contract.valid_answer(float('nan')))

    def test_resume_rejects_historical_contract(self):
        with self.assertRaises(ValueError):
            contract.validate_resume_predictions([{'field_id':'1','answers':'N/A'}])
        contract.validate_resume_predictions(self.coerce([]))

    def test_all_local_backends_share_contract_and_failures_are_wrong(self):
        for name in BACKENDS:
            with self.subTest(backend=name):
                module=importlib.import_module('prompt_eval.'+name)
                output=module.coerce_records([], NOTE, ['1','2'], META, 'baseline')
                self.assertTrue(all(r['answers'] is None for r in output))
                score=module.evaluate_predictions(output,{('synthetic-001','1'):'N/A',('synthetic-001','2'):'N/A'})
                self.assertEqual(score['exact'],0)

    def test_gemma_incomplete_response_retries_and_stays_failed(self):
        from prompt_eval import evaluate_gemma4_extraction as backend
        with patch.object(backend,'generate',return_value=('[{"field_id":"1","answers":"No"}]','')) as generate:
            _,_,rows,error,attempts=backend.generate_predictions(
                None,None,None,NOTE,['1','2'],META,'baseline','synthetic prompt',10,0,1,1,1)
        self.assertEqual(generate.call_count,2)
        self.assertEqual(len(attempts),2)
        self.assertTrue(error)
        self.assertTrue(all(r['answers'] is None and not r['extraction_success'] for r in rows))

    def test_hosted_backend_uses_same_retry_contract(self):
        from prompt_eval import evaluate_api_extraction as backend
        with patch.object(backend,'generate',return_value='[{"field_id":"1","answers":"No"}]') as generate:
            _,rows,error,attempts=backend.generate_predictions(
                None,'gpt-6-sol',NOTE,['1','2'],META,'baseline','synthetic prompt',10,1)
        self.assertEqual(generate.call_count,2)
        self.assertTrue(error)
        self.assertEqual(len(attempts),2)
        self.assertTrue(all(r['answers'] is None for r in rows))


class ScoringTests(unittest.TestCase):
    def score(self, rows, policy='valid-response-v2'):
        return score_group(rows,{('synthetic-001','1'):'N/A',('synthetic-001','2'):'No'},
                           {'synthetic-001'},'all-gold-fields',META,scoring_policy=policy)[0]

    def test_failed_fallback_never_earns_default_credit(self):
        row={'record_id':'synthetic-001','field_id':'1','answers':'N/A','parse_error':'failed'}
        self.assertEqual(self.score([row])['correct'],0)
        self.assertEqual(self.score([row])['failed_responses'],1)
        self.assertEqual(self.score([row],'legacy-answer-only')['correct'],1)

    def test_missing_answers_do_not_gain_credit_after_flattening(self):
        rows=flatten_predictions([{'record_id':'synthetic-001','field_id':'1'}])
        self.assertNotIn('answers',rows[0])
        self.assertEqual(self.score(rows)['correct'],0)

    def test_missing_targets_remain_in_denominator(self):
        m=self.score([{'record_id':'synthetic-001','field_id':'2','answers':'No'}])
        self.assertEqual(m['expected_gold_cells'],2)
        self.assertEqual(m['missing_predictions'],1)
        self.assertEqual(m['valid_answer_accuracy'],.5)

    def test_explicit_na_and_negative_equivalence_are_retained(self):
        rows=[{'record_id':'synthetic-001','field_id':'1','answers':'N/A'},
              {'record_id':'synthetic-001','field_id':'2','answers':'No'}]
        self.assertEqual(self.score(rows)['correct'],2)

    def test_failed_status_overrides_matching_answer(self):
        row={'record_id':'synthetic-001','field_id':'2','answers':'No','extraction_success':False}
        self.assertEqual(self.score([row])['correct'],0)

    def test_duplicate_scoring_rows_are_rejected(self):
        row={'record_id':'synthetic-001','field_id':'2','answers':'No'}
        with self.assertRaises(ValueError): self.score([row,row])


if __name__=='__main__': unittest.main()
