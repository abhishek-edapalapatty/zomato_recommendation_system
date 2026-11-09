# Zomato Recommendation System

Content based restaurant recommendation from the historical Bengaluru Zomato reviews dataset. It uses NLP (TF-IDF) and cosine similarity to find restaurants with similar review language, then sorts candidates by rating.

## Dataset

The provided Kaggle notebook uses the Zomato Bengaluru restaurant data. Download it with a Kaggle API token:

1. Install the Kaggle CLI and configure `kaggle.json` using the official Kaggle account settings. Keep the token outside source control.
2. From this project folder, run `python src/download_dataset.py`. The loader downloads `himanshupoddar/zomato-bangalore-restaurants` and stores `zomato.csv` in `data/`.
3. Review the dataset terms and cite the source in any submission.

Reference notebook: https://www.kaggle.com/code/chirag9073/zomato-recommendation-system
Dataset: https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/download_dataset.py
python src/build_model.py
streamlit run app.py
```

The app can build its in-memory TF-IDF index on first launch. `build_model.py` writes a reusable joblib artifact and a reproducibility summary under `reports/`.

The downloaded dataset (`data/zomato.csv`) and generated model (`models/*.joblib`) are excluded from GitHub because of their size and the dataset's distribution terms. Download the dataset and build the model locally using the commands above.

## Method

Review strings are safely parsed when serialized as review tuples, grouped by normalized restaurant name, and combined with cuisine and neighbourhood text. TF-IDF uses English stop words, unigrams and bigrams. Cosine similarity retrieves similar profiles. Similarity is a candidate filter; the final requested order is highest rating first, with similarity and vote count as tie-breakers. This is content based and unsupervised; it does not learn user preferences or predict ratings.

## Limitations

The source is a historical Bengaluru snapshot, not live data. Text quality and coverage vary, restaurant names can be ambiguous, and ratings may reflect a different date or context. Review similarity is not proof of quality. This project is an educational demonstration.
