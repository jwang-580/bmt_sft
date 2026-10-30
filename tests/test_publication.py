"""Release-boundary tests using only synthetic marker strings."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.publication import permitted_path, inspect_bytes, validate_answers, build_archive, FIGURE_CELLS
from scripts.export_public_answers import export_rows


class PublicationTests(unittest.TestCase):
    def test_private_artifact_paths_blocked(self):
        for name in ['data/notes.csv','fine_tuning/outputs/train.jsonl','fine_tuning/mimic_eval_record_ids.json',
                     'prompt_eval/results/answers.json','research_eval/results/cache.json','tmp/report.md',
                     '.env','outputs/manuscript.docx','notes.txt','scripts/clinical_notes.txt','../README.md']:
            with self.subTest(name=name): self.assertFalse(permitted_path(name))

    def test_analysis_release_rejects_unknown_data_and_figures(self):
        for name in ['data_analysis/data/notes.json', 'data_analysis/data/analysis.json',
                     'data_analysis/figs/patient_chart.png', 'data_analysis/bmt_analysis_results.xlsx']:
            with self.subTest(name=name): self.assertFalse(permitted_path(name))

    def test_public_metadata_paths_allowed(self):
        for name in ['README.md','scripts/publication.py','tests/test_registry.py',
                     'resources/CIBMTR_Form_2400_R11_Items_simplified.csv',
                     'research_eval/research_questions.csv','.github/workflows/checks.yml']:
            with self.subTest(name=name): self.assertTrue(permitted_path(name))

    def test_analysis_assets_and_metadata_are_allowed(self):
        for name in ['resources/README.md', 'data_analysis/README.md', FIGURE_CELLS,
                     'data_analysis/Figs_scripts/figS2_question_accuracy.R',
                     'data_analysis/figs/fig1_study_design.png',
                     'data_analysis/figs/fig1_study_design.pdf']:
            with self.subTest(name=name): self.assertTrue(permitted_path(name))

    def test_figure_cells_reject_raw_fields_and_invalid_flags(self):
        def payload():
            cell = {'case': 'M01', 'field_id': '32', 'field_name': 'Specify donor',
                    'field_category': 'donor_information', 'base': {'answer_match': False},
                    'fine': {'answer_match': True}}
            return {'mimic': [cell], 'osu': [dict(cell)]}
        self.assertEqual(inspect_bytes(FIGURE_CELLS, json.dumps(payload()).encode()), [])
        for target, key, value in [('cell', 'note', 'synthetic note'),
                                   ('base', 'evidence_from_text', 'synthetic quote'),
                                   ('fine', 'answer_match', 'True')]:
            data = payload()
            cell = data['mimic'][0]
            (cell if target == 'cell' else cell[target])[key] = value
            with self.subTest(key=key):
                self.assertTrue(inspect_bytes(FIGURE_CELLS, json.dumps(data).encode()))

    def test_figure_cell_duplicates_and_invalid_figure_signatures_are_rejected(self):
        cell = {'case': 'M01', 'field_id': '32', 'field_name': 'Specify donor',
                'field_category': 'donor_information', 'base': {'answer_match': False},
                'fine': {'answer_match': True}}
        self.assertTrue(inspect_bytes(FIGURE_CELLS, json.dumps({'mimic': [cell, cell], 'osu': [cell]}).encode()))
        self.assertTrue(inspect_bytes('data_analysis/figs/fig1_study_design.png', b'synthetic text'))

    def test_archive_includes_analysis_assets_but_excludes_extra_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public = {
                'README.md': b'Synthetic source',
                'resources/README.md': b'Question metadata',
                'data_analysis/README.md': b'Figure instructions',
                'data_analysis/Figs_scripts/figS2_question_accuracy.R': b'print(1)',
                'data_analysis/figs/fig1_study_design.png': b'\x89PNG\r\n\x1a\n',
            }
            cell = {'case': 'M01', 'field_id': '32', 'field_name': 'Specify donor',
                    'field_category': 'donor_information', 'base': {'answer_match': False},
                    'fine': {'answer_match': True}}
            public[FIGURE_CELLS] = json.dumps({'mimic': [cell], 'osu': [cell]}).encode()
            for name, content in public.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            private = root / 'data_analysis/data/notes.json'
            private.parent.mkdir(parents=True, exist_ok=True)
            private.write_text('SYNTHETIC_PRIVATE_NARRATIVE', encoding='utf-8')
            output = root / 'releases/source.zip'
            self.assertEqual(build_archive(output, root), len(public))
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(set(archive.namelist()), set(public) | {'RELEASE_MANIFEST.json'})

    def test_answer_export_drops_all_narrative_columns(self):
        raw={'record_id':'SYNTHETIC_PRIVATE_ID','date':'2000-01-01','field_id':'1','answers':'No',
             'note':'SYNTHETIC_PRIVATE_NARRATIVE','evidence_from_text':'SYNTHETIC_PRIVATE_QUOTE',
             'reasoning_trace':'SYNTHETIC_PRIVATE_RATIONALE','raw_generation':'SYNTHETIC_PRIVATE_OUTPUT',
             'supporting_note_ids':['SYNTHETIC_PRIVATE_SOURCE']}
        result=export_rows([raw],b'synthetic test salt')
        serialized=json.dumps(result)
        self.assertEqual(result[0]['answers'],'No')
        self.assertNotIn('SYNTHETIC_PRIVATE',serialized)
        self.assertEqual(set(result[0]),{'record_id','field_id','answers'})

    def test_same_export_keeps_pairing_but_different_salts_do_not(self):
        rows=[{'record_id':'synthetic','field_id':str(i),'answers':'No'} for i in (1,2)]
        first=export_rows(rows,b'one')
        self.assertEqual(first[0]['record_id'],first[1]['record_id'])
        self.assertNotEqual(first[0]['record_id'],export_rows(rows,b'two')[0]['record_id'])

    def test_answer_only_validation_rejects_quoted_evidence(self):
        public=export_rows([{'record_id':'synthetic','field_id':'1','answers':'No'}],b'one')
        public[0]['evidence_from_text']='synthetic quote'
        with self.assertRaises(ValueError): validate_answers(public)

    def test_multiline_and_long_answers_rejected(self):
        for answer in ['synthetic\ntext','x'*501,{'note':'synthetic'}]:
            with self.subTest(answer_type=type(answer).__name__),self.assertRaises(ValueError):
                export_rows([{'record_id':'synthetic','field_id':'1','answers':answer}],b'one')

    def test_credential_detector_and_unknown_paths(self):
        fake=('hf_'+'a'*32).encode()
        self.assertIn('possible credential material',inspect_bytes('README.md',fake))
        self.assertTrue(inspect_bytes('notes/clinical.json',b'{}'))

    def test_archive_never_includes_private_files_or_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'README.md').write_text('Synthetic code release',encoding='utf-8')
            for relative in ['data/note.csv','fine_tuning/outputs/train.jsonl','.git/private-history',
                             'tmp/note.md','prompt_eval/results/raw.json']:
                p=root/relative;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text('SYNTHETIC_PRIVATE_NARRATIVE',encoding='utf-8')
            output=root/'releases/source.zip'
            self.assertEqual(build_archive(output,root),1)
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(set(archive.namelist()),{'README.md','RELEASE_MANIFEST.json'})
                self.assertNotIn(b'SYNTHETIC_PRIVATE_NARRATIVE',b''.join(archive.read(n) for n in archive.namelist()))

    def test_archive_rejects_secret_in_allowed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'README.md').write_text('hf_'+'a'*32,encoding='utf-8')
            with self.assertRaises(ValueError): build_archive(root/'releases/source.zip',root)
            self.assertFalse((root/'releases/source.zip').exists())

    def test_export_missing_answer_is_rejected(self):
        with self.assertRaises(ValueError): export_rows([{'record_id':'synthetic','field_id':'1'}],b'one')

    def test_non_finite_answer_rejected_before_writing(self):
        with self.assertRaises(ValueError):
            export_rows([{'record_id':'synthetic','field_id':'1','answers':float('nan')}],b'one')

    def test_clinical_csv_cannot_masquerade_as_question_catalog(self):
        self.assertTrue(inspect_bytes('resources/CIBMTR_Form_2400_R11_Items_simplified.csv',
                                     b'patient_id,note\nsynthetic,SYNTHETIC_PRIVATE_NARRATIVE\n'))


if __name__=='__main__': unittest.main()
