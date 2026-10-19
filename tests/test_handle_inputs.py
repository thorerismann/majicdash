"""Submission gating, collapse behavior, and local-only Analyze execution."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'apps'))
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'apps/writer_dashboard.py'
TEXT = 'The public library will open every Sunday starting in November.'


class InputTests(unittest.TestCase):
    def test_blank_and_unsubmitted_text_never_analyzes_or_calls_api(self):
        with patch('modules.basic_stats.analyze_basic_metrics') as analyze, patch('modules.inference.urlopen') as request:
            app = AppTest.from_file(str(APP), default_timeout=60).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.session_state['input_expanded'])
            self.assertEqual(len(app.tabs), 0)
            app.button(key='analyze_writing').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.warning)
            self.assertTrue(app.session_state['input_expanded'])
            app.text_area(key='writing_text').set_value('  \n\t ')
            app.button(key='analyze_writing').click().run()
            self.assertEqual(len(app.tabs), 0)
            app.text_area(key='writing_text').set_value(TEXT).run()
            self.assertEqual(len(app.tabs), 0)
            analyze.assert_not_called()
            request.assert_not_called()

    def test_valid_submit_collapses_preserves_text_and_never_calls_api(self):
        with patch('modules.inference.urlopen') as request:
            app = AppTest.from_file(str(APP), default_timeout=60).run()
            app.text_area(key='writing_text').set_value(TEXT)
            app.button(key='analyze_writing').click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.session_state['input_expanded'])
            self.assertEqual(app.text_area(key='writing_text').value, TEXT)
            self.assertTrue(app.tabs)
            self.assertFalse(any(row.value == 'Suggested tags' for row in app.subheader))
            app.run()
            # Reopening the input and analyzing again must collapse it again.
            app.session_state['input_expanded'] = True
            app.run()
            app.button(key='analyze_writing').click().run()
            self.assertFalse(app.session_state['input_expanded'])
            app.text_area(key='writing_text').set_value(' ').run()
            self.assertEqual(len(app.tabs), 0)
            app.button(key='analyze_writing').click().run()
            self.assertTrue(app.session_state['input_expanded'])
            request.assert_not_called()
