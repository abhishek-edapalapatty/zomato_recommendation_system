from pathlib import Path
import json, sys
import joblib
import pandas as pd
from recommender import prepare_restaurants, build_index
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'data'/'zomato.csv'
if not source.exists():
    raise SystemExit('Dataset not found. Run: python src/download_dataset.py')
raw=pd.read_csv(source, low_memory=False)
restaurants=prepare_restaurants(raw)
vectorizer,matrix=build_index(restaurants)
(ROOT/'models').mkdir(exist_ok=True)
joblib.dump({'restaurants':restaurants,'vect`orizer':vectorizer,'matrix':matrix}, ROOT/'models'/'zomato_tfidf.joblib', compress=3)
summary={'source_rows':int(len(raw)),'restaurants':int(len(restaurants)),'restaurants_with_reviews':int(restaurants.review_text.str.len().gt(0).sum()),'ratings_available':int(restaurants.rating.notna().sum()),'vocabulary_size':int(len(vectorizer.vocabulary_)),'matrix_shape':list(matrix.shape),'method':'TF-IDF unigrams+bigrams, cosine similarity; top results sorted by rating'}
(ROOT/'reports').mkdir(exist_ok=True)
(ROOT/'reports'/'model_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
