"""Shared live/reference dimensions, denominators, and chart selection."""

from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'apps'))
from modules.basic_stats import _analyze_text, analyze_basic_metrics, analyze_additional_tools
from modules.pos_highlighting import POS_TAGS, POS_MODELS, load_pos_model
from modules.sentiment_analysis import sentiment_dimensions, analyze_sentiment_report
from modules.corpus_reference import comparison_values, METRIC_KEYS
from modules.corpus_comparison import build_scatter, load_reference


class DimensionTests(unittest.TestCase):
    def test_shared_schema_and_counts_in_both_languages(self):
        for language, text in [('en', 'Alice loves London. This awful book is not good!'),
                               ('fr', 'Alice aime Paris. Ce livre est très mauvais !')]:
            with self.subTest(language=language):
                values = comparison_values(analyze_basic_metrics(text, language), analyze_additional_tools(text, language),
                                           sentiment_dimensions(text, language))
                self.assertEqual(set(values), set(METRIC_KEYS))
                self.assertEqual(sum(values[f'pos_{tag}_count'] for tag in POS_TAGS), values['token_count'])
                self.assertAlmostEqual(sum(values[f'pos_{tag}_ratio'] for tag in POS_TAGS), 1)
                self.assertAlmostEqual(sum(values[f'sentiment_{tag}_sentence_ratio'] for tag in ('positive', 'neutral', 'negative')), 1)
                report = analyze_sentiment_report(text, language)
                self.assertEqual(values['sentiment_compound'], report.overall['compound'])
                self.assertEqual(values['sentiment_sentence_mean'], report.mean_sentence_compound)
                self.assertEqual(values['sentiment_sentence_min'], min(row['compound'] for row in report.sentences))
                tokens = _analyze_text(text, language)['spacy']['tokens']
                self.assertEqual(len(tokens), values['token_count'])
                self.assertTrue(all(text[t['start']:t['end']] == t['text'] for t in tokens))
                self.assertTrue(POS_MODELS[language].endswith('_sm'))

    def test_unsupported_model_labels_are_missing_not_zero(self):
        values = analyze_additional_tools('Les enfants lisent un livre.', 'fr')
        self.assertIsNone(values['entity_PERSON_count'])
        self.assertEqual(values['entity_PER_count'], 0)
        self.assertIsNone(values['dep_dobj_count'])
        self.assertIsInstance(values['dep_obj_count'], int)

    def test_all_parser_and_entity_labels_have_dimensions(self):
        for model in POS_MODELS.values():
            nlp = load_pos_model(model)
            for pipe, kind in [('parser', 'dep'), ('ner', 'entity')]:
                for label in nlp.get_pipe(pipe).labels:
                    self.assertIn(f'{kind}_{label}_count', METRIC_KEYS)

    def test_scatter_supports_same_axis_and_colon_labels(self):
        frame = load_reference().query('language == "fr"').head(10)
        current = frame.iloc[0].to_dict()
        for x, y in [('sentiment_compound', 'sentiment_compound'), ('dep_aux:pass_ratio', 'pos_NOUN_count')]:
            chart = build_scatter(frame, current, x, y, 'fr').to_dict()
            self.assertEqual(len(chart['layer']), 2)
            self.assertEqual(chart['layer'][0]['encoding']['x']['field'], 'x')

    def test_new_dimension_controls_work(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'apps/writer_dashboard.py'), default_timeout=60).run()
        app.text_area(key='writing_text').set_value('Les enfants lisent des livres. Ils aiment cette excellente école.')
        app.button(key='analyze_writing').click().run()
        for group, metric in [('sentiment', 'sentiment_compound'), ('pos', 'pos_NOUN_ratio'),
                              ('dep', 'dep_aux:pass_ratio'), ('entity', 'entity_PER_count'), ('morph', 'morph_Tense=Past_count')]:
            app.session_state['corpus_group_fr'] = group
            app.run()
            control = app.radio if group in ('sentiment', 'entity') else app.selectbox
            control(key=f'corpus_metric_fr_{group}').set_value(metric).run()
            self.assertFalse(app.exception)
        app.selectbox(key='corpus_x_fr').set_value('dep_aux:pass_ratio').run()
        app.selectbox(key='corpus_y_fr').set_value('sentiment_compound').run()
        self.assertFalse(app.exception)
