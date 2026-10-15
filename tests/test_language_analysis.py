import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))

import writer_dashboard as app
from modules.sentiment_analysis import get_sentiment_analyzer
from modules.style_analysis import analyze_style
from modules.text_utils import tokenize_words


FRENCH = (
    "L’école présente une œuvre excellente. Les élèves sont heureux et les parents "
    "sont très satisfaits. Cependant, le projet pourrait peut-être être amélioré."
)
ENGLISH = "The students enjoy reading books. Their teacher explains the lesson clearly."


class LanguageAnalysisTests(unittest.TestCase):
    def test_detection_and_manual_override(self):
        self.assertEqual(app.detect_language(FRENCH), "fr")
        self.assertEqual(app.detect_language(ENGLISH), "en")
        self.assertEqual(app.select_analysis_language("fr", "Auto-detect"), "fr")
        self.assertEqual(app.select_analysis_language("fr", "English"), "en")
        self.assertEqual(app.select_analysis_language("unknown", "French"), "fr")
        self.assertIsNone(app.select_analysis_language("de", "Auto-detect"))

    def test_unicode_words_and_apostrophes(self):
        self.assertEqual(tokenize_words("L’école œuvre très"), ["l'école", "œuvre", "très"])
        self.assertEqual(tokenize_words("l’école"), tokenize_words("l'école"))

    def test_language_specific_readability_and_cache(self):
        french = app.analyze_basic_metrics(FRENCH, "fr")
        english = app.analyze_basic_metrics(FRENCH, "en")
        self.assertIsNone(french["fk_grade"])
        self.assertIsInstance(english["fk_grade"], float)
        self.assertNotEqual(french["reading_ease"], english["reading_ease"])
        self.assertEqual(french, app.analyze_basic_metrics(FRENCH, "fr"))

    def test_french_repetition_ignores_stopwords_and_elisions(self):
        style = analyze_style("Avec avec avec l'école école l’école", "fr")
        self.assertEqual(style["repeated_terms"], [("école", 3)])

    def test_french_sentiment_and_summary(self):
        self.assertIn("vaderSentiment_fr", type(get_sentiment_analyzer("fr")).__module__)
        self.assertNotIn("vaderSentiment_fr", type(get_sentiment_analyzer("en")).__module__)
        sentiment = app.analyze_sentiment(FRENCH, "fr")
        self.assertGreater(sentiment["compound"], 0)
        summary = app.build_summary(FRENCH, app.analyze_basic_metrics(FRENCH, "fr"), sentiment)
        self.assertIn("French reading ease", summary[1])
        self.assertIn("Expected response language: French", app.build_ai_prompt(FRENCH, "fr"))

    def test_diagnostics_use_language_and_whole_words(self):
        french = app.analyze_additional_tools(FRENCH, "fr")
        self.assertEqual(french["transition_hits"], ["cependant"])
        self.assertEqual(french["hedge_count"], 1)
        self.assertIsNone(french["passive_count"])
        english = app.analyze_additional_tools("The secondhand book was borrowed. Perhaps it helps.", "en")
        self.assertEqual(english["transition_hits"], [])
        self.assertEqual(english["hedge_count"], 1)
        self.assertEqual(english["passive_count"], 1)

    def test_streamlit_switches_language_with_text_and_override(self):
        from streamlit.testing.v1 import AppTest

        dashboard = AppTest.from_file(str(Path(app.__file__)), default_timeout=30).run()
        self.assertFalse(dashboard.exception)
        dashboard.text_area[0].set_value(FRENCH)
        dashboard.button[0].click().run()
        self.assertFalse(dashboard.exception)
        self.assertFalse(dashboard.error)
        self.assertIn("Using French analysis", dashboard.success[0].value)
        self.assertTrue(any(m.label == "Document compound" for m in dashboard.metric))
        self.assertEqual(set(dashboard.dataframe[0].value["language"]), {"fr"})
        self.assertEqual(next(m.value for m in dashboard.metric if m.label == "F-K Grade"), "N/A")
        self.assertTrue(any("<mark " in m.value for m in dashboard.markdown))

        dashboard.text_area[0].set_value(ENGLISH).run()
        self.assertFalse(dashboard.exception)
        self.assertFalse(dashboard.error)
        self.assertIn("Using English analysis", dashboard.success[0].value)
        self.assertEqual(set(dashboard.dataframe[0].value["language"]), {"en"})
        self.assertNotEqual(next(m.value for m in dashboard.metric if m.label == "F-K Grade"), "N/A")

        dashboard.sidebar.selectbox[0].select("French").run()
        self.assertFalse(dashboard.exception)
        self.assertFalse(dashboard.error)
        self.assertIn("Using French analysis", dashboard.success[0].value)


if __name__ == "__main__":
    unittest.main()
