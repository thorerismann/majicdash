"""Corpus-wide invariants and filter/distribution behavior on a small fixture."""

import csv
import gzip
import json
from pathlib import Path
import sys
import tempfile
import subprocess
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets"))
from corpus_tools.distributions import select_articles, distribution
from corpus_tools.pipeline import build


class DistributionTests(unittest.TestCase):
    def test_filter_deduplicates_overlapping_topics_and_intersects_facets(self):
        articles = pd.DataFrame([
            {"article_id": "a", "year": 2024, "analysis_language": "fr", "benchmark_eligible": True},
            {"article_id": "b", "year": 2025, "analysis_language": "fr", "benchmark_eligible": True},
            {"article_id": "c", "year": 2025, "analysis_language": "en", "benchmark_eligible": False},
        ])
        articles["strict_benchmark_eligible"] = articles["benchmark_eligible"]
        terms = pd.DataFrame([
            {"article_id": "a", "dimension": "topic", "term_key": "topic:culture", "label": "Culture"},
            {"article_id": "a", "dimension": "topic", "term_key": "topic:société", "label": "Société"},
            {"article_id": "b", "dimension": "topic", "term_key": "topic:culture", "label": "Culture"},
            {"article_id": "a", "dimension": "tag", "term_key": "tag:livre", "label": "livre"},
        ])
        result = select_articles(articles, terms, topics=["Culture", "Société"])
        self.assertEqual(list(result.article_id), ["a", "b"])
        result = select_articles(articles, terms, topics=["Culture"], tags=["livre"])
        self.assertEqual(list(result.article_id), ["a"])
        self.assertEqual(list(select_articles(articles, terms, year_min=2025).article_id), ["b"])

    def test_nulls_and_small_samples_are_explicit(self):
        result = distribution([1, None, 9], [0, 5, 10], article_count=3, missing_count=1)
        self.assertEqual(result["observation_count"], 2)
        self.assertEqual(result["median"], 5)
        self.assertEqual(result["missing_count"], 1)
        self.assertEqual(result["histogram"]["counts"], [1, 1])
        self.assertTrue(result["small_sample"])
        empty = distribution([], [0, 1], article_count=0)
        self.assertIsNone(empty["median"])


class PipelineTests(unittest.TestCase):
    def test_sql_source_export_does_not_overwrite_benchmarks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "exports"
            output.mkdir()
            (output / "cleaned.csv").write_text("existing benchmark\n")
            fields = [""] * 23
            for index, value in {0: "1", 1: "2", 2: "2025-01-01", 4: "<p>Un court article.</p>",
                                 5: "Exemple", 7: "publish", 17: "0", 20: "post"}.items():
                fields[index] = value
            dump = root / "source.gz"
            with gzip.open(dump, "wt", encoding="utf-8") as stream:
                stream.write("INSERT INTO `wp_posts` VALUES (" + ",".join("'" + v + "'" for v in fields) + ");\n")
            script = Path(__file__).resolve().parents[1] / "assets" / "article_exports" / "export_wordpress_articles.py"
            subprocess.run([sys.executable, str(script), str(dump), "--output-dir", str(output)], check=True, capture_output=True)
            self.assertTrue((output / "rough.csv").exists())
            self.assertTrue((output / "intermediate.csv").exists())
            self.assertEqual((output / "cleaned.csv").read_text(), "existing benchmark\n")

    def test_roundtrip_retains_short_media_duplicate_and_typed_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.csv"
            fields = ["id", "raw_content", "title", "published_at", "author", "categories_json", "tags_json"]
            text = '<h3>Le prix du vintage</h3><p>Par exemple, les boutiques proposent des vêtements de seconde main aux habitants de cette ville.</p><p>Non.</p>'
            records = [
                ["1", text, "Premier article", "2024-01-01", "Sara Benito", '["Société", "Ecrit"]', '["vintage"]'],
                ["2", text, "Premier article", "2025-01-01", "Sara Benito", '["Culture"]', '[]'],
                ["3", '<iframe src="example"></iframe>', "Vidéo seule", "2025-01-02", "Sara Benito", '["Vidéo"]', '[]'],
                ["4", '<p>Pourquoi ?</p>', "Question brève", "2025-01-03", "Sara Benito", '["Culture"]', '[]'],
            ]
            with source.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(fields)
                writer.writerows(records)
            articles, blocks, terms, sections, manifest = build(source, root / "output", with_nlp=False)
            self.assertEqual(len(articles), 4)
            self.assertEqual(manifest["benchmark_article_count"], 2)
            duplicate = articles.set_index("article_id").loc["2"]
            self.assertEqual(duplicate.duplicate_of, "1")
            self.assertEqual(articles.set_index("article_id").loc["4"].paragraph_count, 1)
            self.assertEqual(len(blocks[blocks.kind == 'heading']), 2)
            self.assertEqual(len(blocks[blocks.kind == 'paragraph']), 5)
            self.assertTrue(all(manifest["checks"].values()))
            restored = pd.read_parquet(root / "output" / "articles.parquet")
            self.assertEqual(restored.article_id.tolist(), articles.article_id.tolist())
            distributions = json.loads((root / "output" / "distributions.json").read_text())
            self.assertEqual(distributions["overall"]["article_metrics"]["word_count"]["observation_count"], 2)
            self.assertEqual(distributions["overall"]["block_metrics"]["paragraph_words"]["observation_count"], 3)
            self.assertEqual(distributions["overall"]["article_metrics"]["word_count"]["histogram"]["bin_edges"],
                             distributions["by_year"]["2025"]["article_metrics"]["word_count"]["histogram"]["bin_edges"])


if __name__ == "__main__":
    unittest.main()
