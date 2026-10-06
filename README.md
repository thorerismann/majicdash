# majicdash

This folder is a self-contained package of the Streamlit app, prepared so you can move it and open it directly in Visual Studio Code.

## Included

- `apps/writer_dashboard_en.py`: English Streamlit app
- `apps/writer_dashboard_fr.py`: French Streamlit app
- `assets/code_machine_angle.html`: shared angle-machine HTML asset
- `requirements.txt`: pip dependencies for both app variants
- `environment.yml`: optional Conda environment
- `.vscode/launch.json`: VS Code run/debug profiles for both apps
- `.vscode/extensions.json`: recommended VS Code Python extensions

## Open In VS Code

1. Open this `majicdash` folder as the workspace root.
2. Create a virtual environment, or use the provided Conda file.
3. Install dependencies from `requirements.txt`.
4. Run either launch profile from the VS Code Run and Debug panel.

## Pip Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Conda Setup

```bash
mamba env create -f environment.yml
conda activate majicdash
```

## spaCy Language Models

The part-of-speech highlighter needs a language model for each dashboard you plan to use:

```bash
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

## Run Without VS Code

```bash
streamlit run apps/writer_dashboard_en.py
streamlit run apps/writer_dashboard_fr.py
```

## Notes

- The English and French apps both use the same HTML asset in `assets/`.
- The French app requires `vaderSentiment-fr`, which is already included in the bundled dependency files.
- The part-of-speech highlighting tabs use spaCy. Install the matching English and French language models with the commands above before running the apps.
- The original source files outside this folder are not needed for the bundle to run.
