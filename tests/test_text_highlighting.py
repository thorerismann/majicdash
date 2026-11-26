"""Source integrity, overlapping spans, citation validation, and local highlighting."""

import json
from html.parser import HTMLParser
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))
from modules.basic_stats import _analyze_text, analyze_style
from modules.text_highlighting import highlighted_html, match_evidence
from modules.ai_summary import parse_summary_response


class TextCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = ""
        self.mark_depth = 0
        self.max_mark_depth = 0
        self.tags = []

    def handle_data(self, data):
        self.text += data

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        if tag == "mark":
            self.mark_depth += 1
            self.max_mark_depth = max(self.max_mark_depth, self.mark_depth)

    def handle_endtag(self, tag):
        if tag == "mark":
            self.mark_depth -= 1


class HighlightTests(unittest.TestCase):
    def test_overlaps_and_unsafe_text_preserve_every_character(self):
        text = '  L’école & « été »\n\n<script>alert("x")</script> 😀  '
        spans = [{"start": 2, "end": 14, "label": 'Long <sentence> "'},
                 {"start": 5, "end": 18, "label": 'Repeated & quoted'},
                 {"start": -1, "end": 20, "label": "Invalid"}]
        html = highlighted_html(text, spans)
        collector = TextCollector()
        collector.feed(html)
        self.assertEqual(collector.text, text)
        self.assertEqual(collector.max_mark_depth, 1)
        self.assertEqual(collector.mark_depth, 0)
        self.assertEqual(set(collector.tags), {"div", "mark"})
        self.assertIn('Long &lt;sentence&gt; &quot; · Repeated &amp; quoted', html)
        self.assertNotIn('Invalid', html)
        self.assertNotIn('<script>', html)

    def test_repeated_sentences_use_ids_and_ambiguous_quotes_are_not_guessed(self):
        text = 'Hello world. Hello world. Again again.'
        rows = [
            {"sentence": 1, "start": 0, "end": 12, "text": 'Hello world.'},
            {"sentence": 2, "start": 13, "end": 25, "text": 'Hello world.'},
            {"sentence": 3, "start": 26, "end": 38, "text": 'Again again.'},
        ]
        spans, unmatched = match_evidence(text, rows, [
            {"sentence": 2, "quote": 'Hello world'},
            {"sentence": 2, "quote": 'Hello world'},
            {"sentence": 1, "quote": 'HELLO'},
            {"sentence": 9, "quote": 'Hello world'},
            {"sentence": True, "quote": 'Hello world'},
            {"sentence": 3, "quote": 'gain'},
        ], 'Finding')
        self.assertEqual(spans, [{"start": 13, "end": 24, "label": 'Finding'}])
        self.assertEqual(unmatched, 4)

    def test_offsets_use_original_french_apostrophes_and_whitespace(self):
        text = '  L’école ouvre.\n\nL’école ferme.  '
        rows = _analyze_text(text, 'fr')['sentence_spans']
        for row in rows:
            self.assertEqual(text[row['start']:row['end']], row['text'])
        spans, unmatched = match_evidence(text, rows, [{"sentence": 2, "quote": 'L’école'}], 'Citation')
        self.assertEqual(unmatched, 0)
        self.assertEqual(spans[0]['start'], text.rindex('L’école'))
        self.assertEqual(match_evidence(text, rows, [{"sentence": 2, "quote": "L'école"}], 'Citation'), ([], 1))

    def test_local_repetition_highlights_inflected_content_words(self):
        text = 'She writes reports. They are writing articles. He wrote a book.'
        style = analyze_style(text, 'en')
        self.assertIn(('write', 3), style['repeated_terms'])
        words = [text[row['start']:row['end']] for row in style['highlights'] if row['kind'] == 'repetition']
        self.assertEqual(words, ['writes', 'writing', 'wrote'])

    def test_all_long_sentences_have_spans_even_beyond_example_limit(self):
        sentence = ' '.join(['word'] * 28) + '.'
        text = '\n\n'.join([sentence] * 6)
        style = analyze_style(text, 'en')
        self.assertEqual(len(style['long_sentences']), 5)
        spans = [row for row in style['highlights'] if row['kind'] == 'long_sentences']
        self.assertEqual(len(spans), 6)
        self.assertTrue(all(text[row['start']:row['end']] == sentence for row in spans))

    def test_invalid_structured_reviews_fail_and_unmatched_evidence_keeps_finding(self):
        finding = {"title": "Issue", "metric_evidence": "Metric", "why_it_matters": "Impact",
                   "suggestion": "Fix", "evidence": [{"sentence": 99, "quote": "invented"}]}
        valid = {"diagnosis": "Review", "findings": [finding]}
        self.assertEqual(parse_summary_response(json.dumps(valid), 'en'), valid)
        self.assertEqual(parse_summary_response('```json\n' + json.dumps(valid) + '\n```', 'en'), valid)
        for raw in ('not JSON', 'null', '{}', json.dumps({"diagnosis": "Review", "findings": [finding] * 6}),
                    json.dumps({"diagnosis": "Review", "findings": [{"title": "Missing fields"}]})):
            with self.subTest(raw=raw), self.assertRaises(RuntimeError):
                parse_summary_response(raw, 'en')
        empty = {"diagnosis": "No substantial weaknesses.", "findings": []}
        self.assertEqual(parse_summary_response(json.dumps(empty), 'fr'), empty)

    def test_writing_view_filters_highlights_without_altering_text(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'apps/writer_dashboard.py'), default_timeout=60).run()
        text = 'She writes reports. They are writing articles. He wrote a book.'
        app.sidebar.radio(key='analysis_view').set_value('basic_stats')
        app.text_area(key='writing_text').set_value(text)
        app.button(key='analyze_writing').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.html[-1].value.count('<mark '), 3)
        app.segmented_control(key='style_highlight_kind').set_value('long_sentences').run()
        self.assertFalse(app.exception)
        self.assertNotIn('<mark ', app.html[-1].value)
        collector = TextCollector()
        collector.feed(app.html[-1].value)
        self.assertEqual(collector.text, text)
