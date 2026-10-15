import csv
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))

from modules.sentiment_analysis import (
    analyze_sentiment,
    analyze_sentiment_report,
    classify_sentiment,
    sentiment_csv,
)


class SentimentAnalysisTests(unittest.TestCase):
    def test_documented_threshold_boundaries(self):
        for score, expected in ((-0.05, "Negative"), (-0.049, "Neutral"), (0, "Neutral"),
                                (0.049, "Neutral"), (0.05, "Positive")):
            with self.subTest(score=score):
                self.assertEqual(classify_sentiment(score), expected)

    def test_negation_emphasis_and_curly_apostrophes(self):
        positive = analyze_sentiment("The book is good.")
        negated = analyze_sentiment("The book is not good.")
        emphatic = analyze_sentiment("The book is VERY GOOD!!!")
        self.assertGreater(positive["compound"], 0)
        self.assertLess(negated["compound"], 0)
        self.assertGreater(emphatic["compound"], positive["compound"])
        self.assertEqual(analyze_sentiment("The book isn’t good."), analyze_sentiment("The book isn't good."))

    def test_sentence_report_exposes_mixed_sentiment(self):
        text = "The service was wonderful! The food was awful. The book is on the table."
        report = analyze_sentiment_report(text)
        self.assertEqual(len(report.sentences), 3)
        self.assertEqual(report.sentence_counts, {"Positive": 1, "Negative": 1, "Neutral": 1})
        self.assertEqual(report.overall, analyze_sentiment(text))
        self.assertAlmostEqual(report.mean_sentence_compound, sum(row["compound"] for row in report.sentences) / 3)
        for row in report.sentences:
            self.assertAlmostEqual(row["positive"] + row["negative"] + row["neutral"], 1, places=2)
            self.assertGreaterEqual(row["compound"], -1)
            self.assertLessEqual(row["compound"], 1)

    def test_abbreviations_and_french_csv_preserve_original_text(self):
        text = 'M. Dupont est heureux. Il dit : « C’est excellent, vraiment ! »'
        report = analyze_sentiment_report(text, "fr")
        self.assertEqual(len(report.sentences), 2)
        self.assertEqual(report.sentences[0]["text"], "M. Dupont est heureux.")
        self.assertIn("C’est", report.sentences[1]["text"])
        rows = list(csv.DictReader(io.StringIO(sentiment_csv(report))))
        self.assertEqual([row["text"] for row in rows], [row["text"] for row in report.sentences])
        self.assertEqual([row["language"] for row in rows], ["fr", "fr"])
        self.assertEqual([row["sentence"] for row in rows], ["1", "2"])
        english = analyze_sentiment_report("Dr. Smith is happy. The food is awful.")
        self.assertEqual(len(english.sentences), 2)

    def test_empty_and_unsupported_input(self):
        report = analyze_sentiment_report("  \n\t", "fr")
        self.assertEqual(report.sentences, [])
        self.assertEqual(report.sentence_counts, {})
        self.assertIsNone(report.mean_sentence_compound)
        self.assertEqual(list(csv.DictReader(io.StringIO(sentiment_csv(report)))), [])
        with self.assertRaises(ValueError):
            analyze_sentiment("Hallo", "de")


if __name__ == "__main__":
    unittest.main()
