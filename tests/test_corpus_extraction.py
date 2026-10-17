"""Structural extraction regressions: real failure modes, not whitespace snapshots."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'assets'))
from corpus_tools.extraction import extract_article, tokenize_words
from corpus_tools.metrics import taxonomy


class CorpusExtractionTests(unittest.TestCase):
    def test_short_paragraphs_and_par_de_sentences_survive(self):
        text = '<p>Pourquoi ?</p><p>Par exemple, cette boutique propose des vêtements de seconde main.</p><p>De plus, les prix sont élevés.</p>'
        result = extract_article(text)
        self.assertEqual(len(result['paragraphs']), 3)
        self.assertEqual(result['paragraphs'][0], 'Pourquoi ?')

    def test_heading_caption_and_nested_figure_are_separate(self):
        text = '<h3>Le prix du vintage</h3><p>Le texte principal.</p><figure><figure><img src="x"></figure><figcaption>Une légende assez longue.</figcaption></figure><p>Non.</p>'
        result = extract_article(text)
        self.assertEqual(result['paragraphs'], ['Le texte principal.', 'Non.'])
        self.assertEqual(result['subheadings'], ['Le prix du vintage'])
        self.assertEqual([b['text'] for b in result['blocks'] if b['kind']=='caption'], ['Une légende assez longue.'])

    def test_br_and_inline_tags_do_not_split_paragraphs_or_words(self):
        result = extract_article('<p>Voici un <strong>mot</strong> et inter<em>national</em>.<br>La suite reste ici.</p><p>Suite.</p>')
        self.assertEqual(result['paragraphs'], ['Voici un mot et international. La suite reste ici.', 'Suite.'])

    def test_classic_editor_blank_lines_define_paragraphs(self):
        result = extract_article('Un texte <a href="x">ancien</a> reste ici.\n\nUn autre paragraphe\nsur deux lignes.')
        self.assertEqual(result['paragraphs'], ['Un texte ancien reste ici.', 'Un autre paragraphe sur deux lignes.'])

    def test_quotes_attribution_and_lists_do_not_become_body_paragraphs(self):
        result = extract_article('<p>Corps.</p><blockquote><p>Une citation.</p><cite>Une personne</cite></blockquote><ul><li>Premier point</li><li>Deuxième point</li></ul>')
        self.assertEqual(result['paragraphs'], ['Corps.'])
        self.assertEqual([b['kind'] for b in result['blocks']], ['paragraph','quote','attribution','list_item','list_item'])
        self.assertIn('Une citation.', result['prose_text'])

    def test_chapo_rules_and_excerpt_provenance(self):
        lead = 'Ce texte présente une introduction suffisamment longue pour expliquer le sujet de cet article.'
        result = extract_article(f'<h4>{lead}</h4><p>Corps.</p>', excerpt='Une description différente.')
        self.assertEqual(result['chapo'], lead)
        self.assertEqual(result['paragraphs'], ['Corps.'])
        self.assertIn('inferred_chapo', result['quality_flags'])
        exact = extract_article(f'<p>{lead}</p><p>Corps.</p>', excerpt=lead)
        self.assertEqual(exact['chapo_source'], 'exact_excerpt_match')
        unknown = extract_article('<p>La première phrase reste un paragraphe normal.</p>', excerpt='Une description différente.')
        self.assertEqual(unknown['chapo'], '')
        self.assertEqual(len(unknown['paragraphs']), 1)

    def test_byline_matching_does_not_remove_author_mentions(self):
        result = extract_article('<p>Sara Benito raconte cette histoire avec précision.</p><h5>Par Sara Benito</h5>', author='Sara Benito')
        self.assertEqual(len(result['paragraphs']), 1)
        self.assertEqual(result['blocks'][-1]['kind'], 'byline')
        reverse = extract_article('<p>Corps.</p><h5>Tassin Jiyana</h5>', author='Jiyana Tassin')
        self.assertEqual(reverse['blocks'][-1]['kind'], 'byline')

    def test_known_shortcodes_and_social_embeds_keep_editorial_brackets(self):
        result = extract_article('<p>Elle rit [rires] et ajoute [ndlr] un détail.</p>[text-blocks id="123"]<blockquote class="twitter-tweet"><p>Un tweet.</p></blockquote>[caption id="a"]<img src="x">Une légende.[/caption]')
        self.assertEqual(result['paragraphs'], ['Elle rit [rires] et ajoute [ndlr] un détail.'])
        self.assertTrue(any(b['kind']=='credit' for b in result['blocks']))
        self.assertTrue(any(b['kind']=='caption' and b['text']=='Une légende.' for b in result['blocks']))
        self.assertNotIn('Un tweet.', result['prose_text'])

    def test_malformed_paragraph_endings_recover_without_merging(self):
        result = extract_article('<p>Premier paragraphe.<p>Deuxième paragraphe.<h3>Intertitre</h3>Suite.')
        self.assertEqual(result['paragraphs'], ['Premier paragraphe.', 'Deuxième paragraphe.', 'Suite.'])
        self.assertIn('html_implicit_closures', result['quality_flags'])

    def test_tokenization_matches_apostrophe_variants(self):
        self.assertEqual(tokenize_words('L’école sʼouvre à l\'enfant.'), ["l'école", "s'ouvre", 'à', "l'enfant"])

    def test_multi_author_footer_and_photo_credit_are_not_headings_or_paragraphs(self):
        result = extract_article('<p>Le récit reste ici.</p><p>Crédits photo © KEYSTONE ATS</p><h5>Par Stephen Mossaz et Fabien Lüthi.</h5><h5>Atelier rédactionnel II, 2013</h5>')
        self.assertEqual(result['paragraphs'], ['Le récit reste ici.'])
        self.assertEqual(result['subheadings'], [])
        self.assertEqual([b['kind'] for b in result['blocks']], ['paragraph','caption','byline','credit'])

    def test_navigation_spacers_and_unresolved_references_are_visible_in_audit(self):
        text = '<p>Un récit qui reste.</p><div class="wp-block-buttons"><a href="x">Chapitre précédent</a></div><p>' + 'Assasa' * 20 + '</p><!-- wp:block {"ref":123} /-->'
        result = extract_article(text)
        self.assertEqual(result['paragraphs'], ['Un récit qui reste.'])
        self.assertIn('unresolved_reusable_block', result['quality_flags'])
        self.assertEqual(sum(b['kind']=='decoration' for b in result['blocks']), 2)

    def test_bibliography_and_explicit_credits_do_not_create_paragraphs(self):
        result = extract_article('<p>Le corps reste ici.</p><p>Crédits photo mise en avant: Pixabay</p><p>Crédit immobilier: un vrai sujet de reportage.</p><p><strong>Références bibliographiques</strong></p><ul><li>Un livre consulté.</li></ul><p>Un autre document.</p>')
        self.assertEqual(result['paragraphs'], ['Le corps reste ici.', 'Crédit immobilier: un vrai sujet de reportage.'])
        self.assertEqual(sum(b['kind']=='reference' for b in result['blocks']), 2)
        self.assertEqual(sum(b['kind']=='caption' for b in result['blocks']), 1)
        self.assertEqual(result['subheadings'], [])

    def test_taxonomy_keeps_multi_membership_and_cohort_ranges(self):
        result = taxonomy(['Volées 1 à 8','Volée 17','Société','Ecrit','Vidéo','Portrait','Perugia2019'], ['politique'])
        self.assertEqual(result['volee'], ['Volée 17','Volées 1 à 8'])
        self.assertEqual(result['medium'], ['Ecrit','Vidéo'])
        self.assertEqual(result['topic'], ['Société'])
        self.assertEqual(result['format'], ['Portrait'])
        self.assertEqual(result['collection'], ['Perugia2019'])


if __name__ == '__main__':
    unittest.main()
