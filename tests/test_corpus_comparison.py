"""Metric alignment, population selection and distribution edge cases."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))

from modules.corpus_comparison import (
    _load_reference, build_chart, distribution_summary, load_reference, select_cohort,
)
from modules.corpus_reference import (
    METRIC_KEYS, REFERENCE_PATH, comparison_values,
)
from modules.basic_stats import analyze_additional_tools, analyze_basic_metrics


class ComparisonTests(unittest.TestCase):
    def test_paragraph_average_uses_all_paragraphs(self):
        text = "\n\n".join(["The reporter writes."] * 7 + ["The journalist carefully checks each source."])
        metrics = analyze_basic_metrics(text, "en")
        extra = analyze_additional_tools(text, "en")
        values = comparison_values(metrics, extra)
        self.assertEqual(len(extra["paragraph_lengths"]), 6)
        self.assertEqual(values["paragraph_count"], 8)
        self.assertEqual(values["avg_paragraph_words"], 27 / 8)
        self.assertEqual(values["syllables_per_word"], metrics["syllable_count"] / 27)

    def test_percentile_ties_missing_values_and_extremes(self):
        self.assertEqual(distribution_summary([1, 2, 2, 3, np.nan, np.inf], 2),
                         {"n": 4, "median": 2, "percentile": 50})
        self.assertEqual(distribution_summary([1, 2], 0)["percentile"], 0)
        self.assertEqual(distribution_summary([1, 2], 3)["percentile"], 100)
        self.assertEqual(distribution_summary([1 / 3], .333333333333)["percentile"], 50)
        self.assertEqual(distribution_summary([], 0)["n"], 0)

    def test_filters_intersect_and_keep_languages_separate(self):
        frame = pd.DataFrame([
            {"article_id": "a", "language": "fr", "medium": "Ecrit", "year": 2024},
            {"article_id": "b", "language": "en", "medium": "Ecrit", "year": 2024},
            {"article_id": "c", "language": "fr", "medium": "Vidéo", "year": 2024},
            {"article_id": "d", "language": "fr", "medium": "Ecrit", "year": 2025},
        ])
        self.assertEqual(select_cohort(frame, "fr", "Ecrit", (2024, 2024)).article_id.tolist(), ["a"])
        self.assertEqual(len(select_cohort(frame, "fr")), 3)
        self.assertTrue(select_cohort(frame, "en", "Vidéo").empty)

    def test_packaged_reference_is_current_and_compact(self):
        frame = load_reference()
        self.assertEqual(len(frame), 805)
        self.assertEqual(frame.language.value_counts().to_dict(), {"fr": 802, "en": 3})
        self.assertEqual(len(select_cohort(frame, "fr", "Ecrit")), 253)
        self.assertTrue(set(METRIC_KEYS).issubset(frame.columns))
        self.assertNotIn("body_text", frame)
        self.assertLess(REFERENCE_PATH.stat().st_size, 2_000_000)
        self.assertTrue(frame.loc[frame.language == "fr", "fk_grade"].isna().all())

    def test_single_csv_loads_and_invalid_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "reference.csv"
            data.write_bytes(REFERENCE_PATH.read_bytes())
            self.assertEqual(len(_load_reference(str(data), 0)), 805)
            data.write_text("wrong data")
            with self.assertRaisesRegex(ValueError, "schema"):
                _load_reference(str(data), 1)

    def test_builder_uses_cleaned_text_and_writes_only_one_csv(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from build_dashboard_reference import build

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "cleaned.csv"
            output = root / "output" / "reference.csv"
            records = [
                {"article_id": "001", "body_text": "L’école accueille les enfants. Ils lisent des livres.",
                 "analysis_language": "fr", "medium": "Ecrit", "year": 2025,
                 "strict_benchmark_eligible": True, "word_count": 999999, "reading_ease": -999999},
                {"article_id": "002", "body_text": "The children read books.",
                 "analysis_language": "en", "medium": "Ecrit", "year": 2025,
                 "strict_benchmark_eligible": True, "word_count": 999999, "reading_ease": -999999},
            ]
            excluded = dict(records[0], article_id="003", strict_benchmark_eligible=False)
            unknown = dict(records[0], article_id="004", analysis_language="unknown")
            pd.DataFrame([*records, excluded, unknown]).to_csv(source, index=False)
            original_bytes = source.read_bytes()
            build(source, output)
            self.assertEqual(list(output.parent.iterdir()), [output])
            self.assertEqual(source.read_bytes(), original_bytes)
            frame = _load_reference(str(output), 0).set_index("article_id")
            self.assertEqual(list(frame.index), ["001", "002"])
            for row in records:
                text, language = row["body_text"], row["analysis_language"]
                expected = comparison_values(analyze_basic_metrics(text, language),
                                             analyze_additional_tools(text, language))
                for key, value in expected.items():
                    actual = frame.loc[row["article_id"], key]
                    if value is None:
                        self.assertTrue(pd.isna(actual))
                    else:
                        self.assertAlmostEqual(actual, value, places=7)

    def test_chart_includes_extreme_input_and_all_reference_observations(self):
        for values, current, metric in [
            ([1, 1, 1], 100, "paragraph_count"),
            ([-20, 0, 110], -80, "reading_ease"),
        ]:
            chart = build_chart(values, current, metric, "en").to_dict()
            domain = chart["layer"][1]["encoding"]["x"]["scale"]["domain"]
            self.assertLess(domain[0], min(current, min(values)))
            self.assertGreater(domain[1], max(current, max(values)))
            counts = [row["count"] for rows in chart["datasets"].values() for row in rows if "count" in row]
            self.assertEqual(sum(counts), len(values))
            self.assertEqual(len(chart["layer"]), 5)

    def test_streamlit_switches_metrics_filters_and_language(self):
        from streamlit.testing.v1 import AppTest

        app_path = Path(__file__).resolve().parents[1] / "apps/writer_dashboard.py"
        app = AppTest.from_file(str(app_path), default_timeout=60).run()
        app.text_area(key="writing_text").set_value(
            "Les écoles ferment. Une école ouvre. Cette école grandit.\n\nLes enfants lisent des livres."
        )
        app.button(key="analyze_writing").click().run()
        self.assertFalse(app.exception)

        def displayed(label):
            return next(metric.value for metric in app.metric if metric.label == label)

        self.assertEqual(displayed("Articles de référence"), "253")
        app.selectbox(key="corpus_medium_fr").set_value("__all__").run()
        self.assertEqual(displayed("Articles de référence"), "802")
        for group, metric in [("paragraphs", "paragraph_count"), ("syllables", "syllable_count"),
                              ("characters", "char_count"), ("readability", "reading_ease")]:
            app.session_state["corpus_group_fr"] = group
            app.run()
            app.radio(key=f"corpus_metric_fr_{group}").set_value(metric).run()
            self.assertFalse(app.exception)
        frame = load_reference()
        first_year = int(frame.loc[frame.language == "fr", "year"].min())
        app.slider(key="corpus_years_fr").set_value((first_year, first_year)).run()
        expected = len(select_cohort(frame, "fr", years=(first_year, first_year)))
        self.assertEqual(displayed("Articles de référence"), str(expected))

        app.selectbox(key="language_choice").set_value("en").run()
        app.text_area(key="writing_text").set_value("The journalist writes an article. She checks every source.")
        app.button(key="analyze_writing").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(displayed("Reference articles"), "3")
        self.assertEqual(displayed("Percentile"), "—")
        self.assertTrue(any("Only 3 reference" in warning.value for warning in app.warning))
        app.session_state["corpus_group_en"] = "readability"
        app.run()
        app.radio(key="corpus_metric_en_readability").set_value("fk_grade").run()
        self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
