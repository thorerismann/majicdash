import sys
import unittest
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))

from modules.pos_highlighting import analyze_parts_of_speech, highlight_parts_of_speech, part_of_speech_counts


class HighlightParser(HTMLParser):
    def __init__(self, markup):
        super().__init__(convert_charrefs=True)
        self.text = []
        self.marked = []
        self.in_mark = False
        self.feed(markup)

    def handle_starttag(self, tag, attrs):
        self.in_mark = tag == "mark"

    def handle_endtag(self, tag):
        if tag == "mark":
            self.in_mark = False

    def handle_data(self, data):
        self.text.append(data)
        if self.in_mark:
            self.marked.append(data)


class PosHighlightingTests(unittest.TestCase):
    def test_french_apostrophes_tag_verbs_and_nouns_consistently(self):
        for apostrophe in ("'", "’", "ʼ"):
            text = (
                f"La seconde main s{apostrophe}ancre dans les habitudes suisses. "
                f"Mais dans l{apostrophe}essor des friperies vintages, certains prix "
                "peuvent parfois sembler élevés."
            )
            with self.subTest(apostrophe=apostrophe):
                markup, counts = highlight_parts_of_speech(text, "fr_core_news_md", {"VERB"})
                parsed = HighlightParser(markup)
                self.assertIn("ancre", parsed.marked)
                self.assertNotIn("essor", parsed.marked)
                self.assertEqual("".join(parsed.text), text)
                self.assertEqual(counts["VERB"], part_of_speech_counts(text, "fr_core_news_md")["VERB"])
                nouns, _ = highlight_parts_of_speech(text, "fr_core_news_md", {"NOUN"})
                self.assertIn("essor", HighlightParser(nouns).marked)

    def test_auxiliaries_are_included_in_verbs(self):
        for model, text, auxiliary in (
            ("fr_core_news_md", "Les enfants sont heureux.", "sont"),
            ("en_core_web_md", "The children are happy.", "are"),
        ):
            with self.subTest(model=model):
                markup, counts = highlight_parts_of_speech(text, model, {"VERB"})
                self.assertIn(auxiliary, HighlightParser(markup).marked)
                self.assertEqual(counts["VERB"], 1)
                self.assertEqual(part_of_speech_counts(text, model)["VERB"], 1)

    def test_compound_verbs_and_counts_when_highlights_are_disabled(self):
        for model, text, verbs in (
            ("fr_core_news_md", "Les enfants ont mangé.", {"ont", "mangé"}),
            ("en_core_web_md", "The children have eaten.", {"have", "eaten"}),
        ):
            with self.subTest(model=model):
                selected = analyze_parts_of_speech(text, model, {"VERB"})
                unselected = analyze_parts_of_speech(text, model, set())
                self.assertEqual(set(HighlightParser(selected.html).marked), verbs)
                self.assertEqual(selected.counts["VERB"], 2)
                self.assertEqual(selected.highlighted_counts["VERB"], 2)
                self.assertEqual(selected.counts, unselected.counts)
                self.assertEqual(unselected.highlighted_counts, {})
                self.assertEqual(HighlightParser(unselected.html).marked, [])

    def test_original_text_and_html_escaping_are_preserved(self):
        text = "  L’école s’ancre ici.\n\n\t<script>alert('x')</script> & lʼœuvre.  "
        for selected in (set(), {"VERB", "NOUN", "ADJ", "ADV"}):
            markup, _ = highlight_parts_of_speech(text, "fr_core_news_md", selected)
            self.assertEqual("".join(HighlightParser(markup).text), text)
            self.assertNotIn("<script>", markup)


if __name__ == "__main__":
    unittest.main()
