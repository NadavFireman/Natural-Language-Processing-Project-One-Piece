# Natural Language Processing Final Project - One Piece

**Final Project (M.Sc. Data Science, HIT). Predicting the bounties of *One Piece* characters — a quantitative threat score — from wiki text combined with a character-relationship network. Data collected independently from the One Piece Fandom Wiki API; no Kaggle or pre-built dataset.**

## Key Features
- **Independent Data Collection:** 1,734 character pages scraped via the public Fandom API — **218 characters with a known bounty** (the target) plus hundreds more for the relationship graph.
- **Text Features:** TF-IDF + Truncated SVD on cleaned page text — bounty figures scrubbed before modeling, and TF-IDF/SVD fit on the training fold only within each CV split (leakage control).
- **Network Analysis:** Character-relationship graph with structural features, visualized via a layered k-core layout (~1,000 nodes).
- **Error Analysis:** Systematic errors investigated (e.g. weak characters on strong crews over-predicted), plus six visualizations.

## Results
Three-way comparison with CatBoost — **combined model R² = 0.42** vs. **0.37** text-only and **0.13** network-only (median baseline: −0.05): text and network carry complementary signal, and their combination beats either alone.

## Repository Structure
- `one_piece_nlp_project.ipynb`: Full end-to-end notebook (collection, features, modeling, evaluation).
- `data_collection.py` / `data_collection.ipynb`: Fandom-API scraper producing `onepiece_raw.csv`.
- `text_features.py`: Text cleaning and TF-IDF/SVD extraction.
- `network_plots.py`: Graph construction and visualization.
- `features.py`: Feature assembly.
- `modeling.py`: CatBoost training, cross-validation and evaluation.
- `onepiece_raw.csv`: Raw collected dataset (~27MB, 1,734 rows) — regenerate with `data_collection.py`.
- `one_piece_report.pdf` / `one_piece_bounty_presentation.pdf`: Full report and presentation.
- `Project_Proposal.pdf` / `Final_Project_Instructions.pdf`: Approved proposal and course guidelines.
- `DATA.md` / `ETHICS.md` / `REFLECTION.md` / `AI_USAGE.md`: Data provenance, ethics, reflection, and AI-usage documentation.
