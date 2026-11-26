# majicdash

This folder is a self-contained package of the Streamlit app, prepared so you can move it and open it directly in Visual Studio Code.

## Included

- `apps/writer_dashboard.py`: French/English interface and text analysis
- `apps/modules/pos_highlighting.py`: POS model selection, tagging, counts, highlighting, and tab rendering
- `apps/modules/sentiment_analysis.py`: English/French VADER scoring, sentence analysis, CSV export, and sentiment tab rendering
- `apps/modules/basic_stats.py`: shared spaCy analysis, statistics, style rules and diagnostics
- `apps/modules/main_displays.py`: input layout and main display
- `apps/modules/handle_inputs.py`: submission validation and analysis state
- `apps/modules/corpus_comparison.py`: distributions, two-axis plots and comparison exports
- `assets/code_machine_angle.html`: angle-machine HTML asset
- `requirements.txt`: pip dependencies for the app
- `apps/requirements.txt`: Community Cloud entrypoint that includes the shared pip dependencies
- `environment.yml`: optional Conda environment

## Open In VS Code

1. Open this `majicdash` folder as the workspace root.
2. Create a virtual environment, or use the provided Conda file.
3. Install dependencies from `requirements.txt`.
4. Run `python -m streamlit run apps/writer_dashboard.py` in the integrated terminal.

## Pip Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Conda Setup

```bash
mamba env create -f environment.yml
conda activate majic2
```

## Streamlit Community Cloud

Deploy `apps/writer_dashboard.py` with Python 3.12. Community Cloud finds
`apps/requirements.txt` beside the entrypoint first; it includes the root
`requirements.txt`, including both small spaCy model wheels. The root
`environment.yml` remains available for local Mamba setup.

Push these dependency files to the branch selected in Cloud (for example,
`main`), then rebuild the app. If spaCy reports E050, check the installation
log for `en-core-web-sm` and `fr-core-news-sm` and any preceding install errors.
Do not download models during app startup.

See [Community Cloud dependency-file precedence](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

## spaCy Language Model

The English and French models (`en_core_web_sm` and `fr_core_news_sm` 3.8.0) are included in both setup files, alongside compatible spaCy 3.8.x and both sentiment libraries. Pip installs the models as regular Python packages; Mamba installs them through the `pip` section of `environment.yml`. No separate model download is needed.

The sidebar's **Text language** setting defaults to auto-detection, with English and French overrides for short or ambiguous text. This selects the matching grammar model, sentiment analyzer, readability language, stopwords, and diagnostic vocabulary. Flesch-Kincaid grade is English-only; passive predicates are detected from each language model’s dependencies. Style and executive-presence scores remain rough heuristics.

To update the existing `majic2` environment:

```bash
conda activate majic2
python -m pip install -r requirements.txt
```

Alternatively, run `mamba env update -n majic2 -f environment.yml`. Restart Streamlit after installing dependencies. The model powers the existing **Parts of Speech** tab.

The POS module uses the small (`sm`) models. Its Verb category includes both `VERB` and `AUX` tags. Apostrophes are normalized for analysis while the original spelling, punctuation, and spacing are preserved in the display. Counts and highlights come from the same model pass.

## Run Without VS Code

```bash
python -m streamlit run apps/writer_dashboard.py
```

## Journalistic Angle Analysis

The dashboard includes a tab that sends the submitted text and an editable prompt to the CSCS inference API. Set `CSCS_INFERENCE_API_KEY` in the ignored project `.env` file (see `.env.example`), or provide it through the process environment or Streamlit secrets. The API key is never shown in the interface. The request uses the `swiss-ai/Apertus-v1.5-8B` model.

## AI summary

The separate **AI summary** menu option sends the source text and computed analysis to Apertus to rank up to five evidence-based revision priorities. Each priority asks for evidence, why it matters, and a concrete edit. The existing journalistic-angle tools remain available.

Generation computes all local analysis families, whether or not their views have been opened: readability and counts, style flags and repetition, complete sentence/paragraph lengths, spaCy aggregate grammar/syntax/morphology/entity data, document and sentence sentiment, and corpus comparisons. Raw token-export rows are omitted; the original text, aggregate metrics and named-entity spans are included. Corpus comparisons use same-language written articles when available (otherwise all types), across all years, independently of chart filters. Missing reference data is explicitly marked, and percentiles are suppressed below five samples.

The model is asked to ground its priorities in quotations and measurements, treat heuristics and corpus differences as context rather than quality scores, and return fewer than five issues if needed. The analysis report sent to Apertus is available below the review. Generation only runs on an explicit click. Results stay in the current session across view changes and are hidden when the input text, language, model or review instructions change. A live model response still needs editorial judgment.

## Findings in the original text

**Writing analysis** reproduces the original text with selectable highlights for long sentences (28 words or more), repeated content-word lemmas, or both. Repetition uses the same eligible nouns, verbs and adjectives and the same top-eight repeated lemmas as the style analysis, including inflected forms. All matching occurrences and long sentences are highlighted, including sentences beyond the five-example preview.

**AI summary** requests structured findings with a diagnosis, evidence, impact, suggested edit, and exact quotations tied to the supplied spaCy sentence IDs. Selecting a priority shows its explanation beside the highlighted original text. A quotation is highlighted only when it occurs exactly once inside its cited sentence. Non-matching quotes, ambiguous matches and invalid sentence IDs are counted and reported without guessed highlights; the finding remains readable. Document-wide findings can have no quotations. Malformed responses produce a retry message and retain any previous successful review for the same text. Downloads include all findings and the verified evidence quotations.

Both views use `apps/modules/text_highlighting.py`: source offsets are computed locally, original text and tooltip labels are escaped, and overlapping spans are split without nesting marks or repeating text. Paragraphs, punctuation, Unicode and original apostrophes are preserved. Sentence positions and repetition positions come from the existing shared spaCy parse; no additional model call is needed when selecting a highlight. The existing grammar highlighter remains available separately.

## Sentiment Analysis

VADER scores emotional wording with a dictionary and rules. The Sentiment tab shows a document compound score (−1 to +1), positive/neutral/negative proportions, and separate sentence scores. Proportions are not confidence probabilities. Labels now use the documented ±0.05 thresholds instead of the previous custom ±0.35 thresholds; raw compound scores retain their meaning.

The sentence chart, category counts, and strongest positive/negative passages help expose changes hidden by a document average. The mean sentence compound gives each sentence equal weight and is not the document compound. All sentence scores can be downloaded as CSV. Sentence boundaries use spaCy's language-specific tokenizer and a rule-based sentencizer, so unusual punctuation may still need review.

French text uses `vaderSentiment-fr`; English uses `vaderSentiment`. VADER does not determine author intent, objectivity, factual accuracy, or writing quality. Review quotations, sarcasm, and domain-specific wording in context. See the [VADER scoring documentation](https://github.com/cjhutto/vaderSentiment#about-the-scoring) and [French adaptation](https://github.com/thomas7lieues/vader_FR).

## Credits and packages

A collapsed credits section at the bottom of the home screen and analysis results describes Apertus generation, the local analysis and visualization packages, and Codex development assistance. Credits are available in French and English. The sidebar’s Analysis to run menu selects one of the seven analysis views, with the language selector and sentence-flag setting below it. Only the selected analysis is rendered; switching views keeps the submitted text and journalistic-angle drafts.

## Notes

- The app uses the angle-machine HTML asset in `assets/`.
- The part-of-speech highlighting tab uses spaCy and the matching language model installed during setup.
- The original source files outside this folder are not needed for the bundle to run.

## Submitting text

The input panel starts open. Analyze requires non-whitespace text, collapses the panel, and displays results. Editing the text requires another Analyze click. Chart and display controls keep the submitted analysis without sending new requests.

Analyze runs locally. Only the separate AI-summary and journalistic-angle generation buttons send requests to CSCS Apertus.

## Expanded corpus comparison

The packaged reference is recalculated from 805 eligible articles with the small models and the same functions as live input. It includes document/sentence VADER scores, lexical diversity, style diagnostics, every POS category, every dependency/entity label supported by the models, and common morphological features. Select a measure for its distribution or expand **Compare any two dimensions** to choose both scatterplot axes. Both plots use the selected language, article type and year filters. CSV downloads include either the current measure or all available dimensions.

POS, dependency and morphology shares divide by non-space tokens; entity shares divide by entity mentions. Entity frequency is also available per 100 words. Unsupported dependency/entity labels are missing, rather than zero, and excluded from controls. The detailed spaCy view separates VERB and AUX, includes all observed morphology features, and exports token text, offsets, lemmas, fine tags, dependency heads, morphology and entities.

To refresh the reference from the local cleaned corpus:

```bash
mamba run -n majic2 python scripts/build_dashboard_reference.py
```

`assets/cleaned.csv` is a local, ignored build input. The generated reference under `apps/data/` is packaged with the app; rebuilding does not call Apertus. Runtime input state uses [Streamlit’s controlled expander API](https://docs.streamlit.io/develop/api-reference/layout/st.expander), so setup requires Streamlit 1.66 or newer.
