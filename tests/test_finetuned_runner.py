"""Starter-runner tests with invented text and mocked GPU/model boundaries."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from prompt_eval import run_finetuned as runner


class RunnerTests(unittest.TestCase):
    def test_demo_dry_run_never_loads_model_or_writes_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'predictions.json'
            with patch.object(runner,'load_model') as load,contextlib.redirect_stdout(io.StringIO()) as log:
                self.assertEqual(runner.main(['--demo','--dry_run','--output',str(output)]),0)
            load.assert_not_called()
            self.assertFalse(output.exists())
            self.assertIn('initial calls: 1',log.getvalue())
            self.assertNotIn(runner.DEMO_NOTE,log.getvalue())

    def test_all_fields_has_26_initial_calls(self):
        with contextlib.redirect_stdout(io.StringIO()) as log:
            runner.main(['--demo','--all_fields','--dry_run'])
        self.assertIn('fields per record: 129; initial calls: 26',log.getvalue())

    def test_csv_extra_labels_do_not_enter_model_note(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'notes.csv'
            path.write_text('record_id,date,note,reference_answer\nsynthetic_1,2000-01-01,Invented note,LABEL_SENTINEL\n',encoding='utf-8')
            notes=runner.load_notes(runner.parse_args(['--notes_csv',str(path)]))
            self.assertNotIn('LABEL_SENTINEL',json.dumps(notes))
            self.assertEqual(notes[0]['record_id'],'synthetic_1')

    def test_csv_duplicate_identifiers_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'notes.csv'
            path.write_text('record_id,note\nsynthetic_1,Invented one\nsynthetic_1,Invented two\n',encoding='utf-8')
            with self.assertRaises(ValueError):
                runner.load_notes(runner.parse_args(['--notes_csv',str(path)]))

    def test_field_selection_rejects_unknown_or_duplicate_ids(self):
        metadata=runner.common.load_question_metadata(str(runner.ROOT/'resources/CIBMTR_Form_2400_R11_Items_simplified.csv'))
        for fields in ['99999','32,32']:
            with self.subTest(fields=fields),self.assertRaises(ValueError):
                runner.selected_fields(runner.parse_args(['--demo','--fields',fields]),metadata)

    def test_local_adapter_layout_detected_without_hub(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            (path/'adapter_config.json').write_text(json.dumps({'base_model_name_or_path':'synthetic-base'}),encoding='utf-8')
            self.assertEqual(runner.checkpoint_layout(str(path),None,True)['base_model_name_or_path'],'synthetic-base')

    def mocked_runtime(self):
        torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True),bfloat16='bf16',float32='fp32')
        processor=SimpleNamespace(chat_template='synthetic template')
        transformers=SimpleNamespace(AutoModelForMultimodalLM=Mock(),AutoProcessor=Mock(),AutoTokenizer=Mock())
        transformers.AutoProcessor.from_pretrained.return_value=processor
        peft=SimpleNamespace(PeftModel=Mock())
        return torch,transformers,peft

    def test_adapter_loads_base_then_attaches_finetuned_weights(self):
        torch,transformers,peft=self.mocked_runtime()
        args=runner.parse_args(['--demo','--revision','adapter-revision','--base_revision','base-revision'])
        with patch.dict(sys.modules,{'torch':torch,'transformers':transformers,'peft':peft}),\
             patch.object(runner,'checkpoint_layout',return_value={'base_model_name_or_path':'synthetic-base'}):
            *_,layout=runner.load_model(args)
        transformers.AutoModelForMultimodalLM.from_pretrained.assert_called_once_with(
            'synthetic-base',device_map='auto',dtype='bf16',local_files_only=False,revision='base-revision')
        peft.PeftModel.from_pretrained.assert_called_once_with(
            transformers.AutoModelForMultimodalLM.from_pretrained.return_value,runner.MODEL,
            revision='adapter-revision',local_files_only=False,is_trainable=False)
        self.assertEqual(layout['checkpoint_type'],'adapter')

    def test_merged_checkpoint_does_not_attach_an_adapter(self):
        torch,transformers,peft=self.mocked_runtime()
        with patch.dict(sys.modules,{'torch':torch,'transformers':transformers,'peft':peft}),\
             patch.object(runner,'checkpoint_layout',return_value=None):
            *_,layout=runner.load_model(runner.parse_args(['--demo']))
        peft.PeftModel.from_pretrained.assert_not_called()
        self.assertEqual(layout['checkpoint_type'],'merged')

    def test_local_base_override_for_offline_adapter(self):
        torch,transformers,peft=self.mocked_runtime()
        with patch.dict(sys.modules,{'torch':torch,'transformers':transformers,'peft':peft}),\
             patch.object(runner,'checkpoint_layout',return_value={'base_model_name_or_path':'synthetic-base'}):
            *_,layout=runner.load_model(runner.parse_args(['--demo','--base_model_path','local-base','--local_files_only']))
        self.assertEqual(layout['base_model'],'local-base')
        self.assertTrue(transformers.AutoModelForMultimodalLM.from_pretrained.call_args.kwargs['local_files_only'])

    def test_incomplete_fields_retry_then_remain_failed(self):
        metadata={'1':{'field_category':'synthetic','field_name':'Invented question','field_instructions':''}}
        note={'record_id':'synthetic_1','record_id_type':'synthetic','date':'','note':'Invented note'}
        generate=Mock(return_value='[]')
        rows=runner.extract_note(note,['1'],metadata,5,1,generate)
        self.assertEqual(generate.call_count,2)
        self.assertFalse(rows[0]['extraction_success'])
        self.assertIsNone(rows[0]['answers'])

    def test_existing_output_rejected_before_model_download(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'predictions.json'
            output.write_text('preserve',encoding='utf-8')
            with patch.object(runner,'load_model') as load,contextlib.redirect_stdout(io.StringIO()),self.assertRaises(ValueError):
                runner.main(['--demo','--output',str(output)])
            load.assert_not_called()
            self.assertEqual(output.read_text(),'preserve')

    def test_successful_run_writes_validated_answers_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'predictions.json'
            processor=SimpleNamespace(apply_chat_template=lambda *a,**kw:{'input_ids':SimpleNamespace(shape=(1,10))})
            response=json.dumps([{'field_id':fid,'answers':'N/A'} for fid in runner.DEFAULT_FIELDS])
            with patch.object(runner,'load_model',return_value=(None,None,processor,{'checkpoint_type':'synthetic'})),\
                 patch.object(runner.common,'generate',return_value=(response,'')),\
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(['--demo','--output',str(output)]),0)
            rows=json.loads(output.read_text())
            self.assertEqual(len(rows),5)
            self.assertTrue(all(row['extraction_success'] for row in rows))
            self.assertEqual(json.loads(output.with_suffix('.manifest.json').read_text())['failed_fields'],0)


if __name__=='__main__': unittest.main()
