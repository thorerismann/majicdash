"""Regression checks for the English/French spaCy-backed writing statistics."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))

from modules import basic_stats as stats


class BasicStatsTests(unittest.TestCase):
    def test_abbreviation_does_not_create_an_extra_sentence(self):
        text = "Dr. Smith writes reports. She edits them."
        metrics = stats.analyze_basic_metrics(text, "en")
        self.assertEqual(metrics["word_count"], 7)
        self.assertEqual(metrics["sentence_count"], 2)
        self.assertEqual(metrics["avg_sentence_length"], 3.5)
        self.assertEqual(metrics["char_count"], len(text))
        self.assertEqual(metrics["reading_minutes"], 7 / 200)
        self.assertEqual(stats.split_sentences(text, "en"), [
            "Dr. Smith writes reports.", "She edits them.",
        ])

    def test_french_elisions_and_original_sentence_spelling(self):
        for apostrophe in ("'", "’", "ʼ"):
            text = f"L{apostrophe}école ouvre."
            with self.subTest(apostrophe=apostrophe):
                self.assertEqual(stats.tokenize_words(text, "fr"), ["l'", "école", "ouvre"])
                self.assertEqual(stats.analyze_style(text, "fr")["short_sentences"], [text])

    def test_paragraph_counts_use_the_same_word_definition(self):
        text = "  First paragraph.\n\nSecond paragraph has 42 numbers.\n \n  "
        extra = stats.analyze_additional_tools(text, "en")
        self.assertEqual(extra["paragraph_count"], 2)
        self.assertEqual(extra["paragraph_lengths"], [2, 4])
        self.assertEqual(extra["longest_paragraph"], 4)
        self.assertEqual(sum(extra["paragraph_lengths"]), stats.analyze_basic_metrics(text, "en")["word_count"])

    def test_passive_predicates_include_irregular_verbs_and_count_once(self):
        text = "The report was written by Alice. The ball was thrown by Bob. The door was opened by Eve."
        self.assertEqual(stats.analyze_additional_tools(text, "en")["passive_count"], 3)
        self.assertEqual(stats.analyze_additional_tools(
            "The door is open. Alice has written a report.", "en",
        )["passive_count"], 0)
        self.assertEqual(stats.analyze_additional_tools("Le rapport est écrit par Alice.", "fr")["passive_count"], 1)
        self.assertEqual(stats.analyze_additional_tools("Alice a écrit un rapport.", "fr")["passive_count"], 0)

    def test_phrase_matches_use_whole_tokens_and_normalize_apostrophes(self):
        english = stats.analyze_additional_tools("The secondhand book was borrowed. Perhaps it helps.", "en")
        self.assertEqual(english["transition_hits"], [])
        self.assertEqual(english["hedge_count"], 1)
        french = stats.analyze_additional_tools(
            "D’abord, l’école semble utile. Cependant, elle pourrait peut-être changer.", "fr",
        )
        self.assertEqual(french["transition_hits"], ["cependant", "d'abord"])
        self.assertEqual(french["hedge_count"], 2)

    def test_repetition_groups_inflections_in_both_languages(self):
        english = stats.analyze_style("She writes reports. They are writing articles. He wrote a book.", "en")
        french = stats.analyze_style("Les écoles ferment. Une école ouvre. Cette école grandit.", "fr")
        self.assertEqual(english["repeated_terms"], [("write", 3)])
        self.assertEqual(french["repeated_terms"], [("école", 3)])
        french_short = stats.analyze_style("La loi est la loi et reste une loi.", "fr")
        self.assertEqual(french_short["repeated_terms"], [("loi", 3)])

    def test_lexical_diversity_still_counts_surface_forms(self):
        result = stats.analyze_additional_tools("Reports report reports.", "en")
        self.assertEqual(result["unique_words"], 2)
        self.assertEqual(result["lexical_diversity"], 2 / 3)

    def test_empty_input_and_readability_language(self):
        for language in ("en", "fr"):
            with self.subTest(language=language):
                metrics = stats.analyze_basic_metrics("", language)
                self.assertEqual(metrics["word_count"], 0)
                self.assertEqual(metrics["sentence_count"], 0)
                self.assertEqual(metrics["reading_ease"], 0)
                extra = stats.analyze_additional_tools(" \n\n ", language)
                self.assertEqual(extra["paragraph_count"], 0)
                self.assertEqual(extra["lexical_diversity"], 0)
                self.assertEqual(stats.analyze_style("", language)["repeated_terms"], [])
        french_metrics = stats.analyze_basic_metrics("Les enfants lisent un livre.", "fr")
        self.assertIsNone(french_metrics["fk_grade"])
        self.assertEqual(french_metrics["reading_ease"], 113.61)
        self.assertIsInstance(stats.analyze_basic_metrics("The children read a book.", "en")["fk_grade"], float)

    def test_basic_style_and_diagnostics_share_one_parse(self):
        for function in (stats._analyze_text, stats.analyze_basic_metrics,
                         stats.analyze_style, stats.analyze_additional_tools):
            function.clear()
        with patch.object(stats, "parse_pos_text", wraps=stats.parse_pos_text) as parse:
            text = "The journalist carefully checks each source."
            stats.analyze_basic_metrics(text, "en")
            stats.analyze_style(text, "en")
            stats.analyze_additional_tools(text, "en")
            self.assertEqual(parse.call_count, 1)


if __name__ == "__main__":
    unittest.main()
