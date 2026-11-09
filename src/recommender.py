from __future__ import annotations
import ast, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ALIASES = {
    'name': ['name', 'restaurant', 'restaurant_name'],
    'reviews': ['reviews_list', 'reviews', 'review', 'review_text'],
    'rating': ['rate', 'rating', 'aggregate_rating'],
    'votes': ['votes', 'vote_count'],
    'cuisines': ['cuisines', 'cuisine'],
    'location': ['location', 'listed_in(city)', 'city'],
    'cost': ['approx_cost(for two people)', 'approx_cost', 'cost'],
}

def find_column(df, key):
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for candidate in ALIASES[key]:
        if candidate in normalized:
            return normalized[candidate]
    return None

def clean_rating(value):
    if pd.isna(value): return np.nan
    match = re.search(r'\d+(?:\.\d+)?', str(value))
    return float(match.group()) if match else np.nan

def extract_review_text(value):
    if pd.isna(value): return ''
    text = str(value).strip()
    if not text: return ''
    # Kaggle reviews_list is commonly serialized as [(rating, 'review'), ...].
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            parts = []
            for item in parsed:
                if isinstance(item, (list, tuple)) and len(item) > 1:
                    parts.append(str(item[-1]))
                elif isinstance(item, str):
                    parts.append(item)
            return ' '.join(parts)
    except (ValueError, SyntaxError, TypeError, MemoryError):
        pass
    return text

def prepare_restaurants(df):
    name_col, review_col = find_column(df, 'name'), find_column(df, 'reviews')
    if name_col is None: raise ValueError('Could not find a restaurant-name column (expected name).')
    if review_col is None: raise ValueError('Could not find review text (expected reviews_list or reviews).')
    rating_col, votes_col = find_column(df, 'rating'), find_column(df, 'votes')
    cuisines_col, location_col, cost_col = (find_column(df, k) for k in ('cuisines','location','cost'))
    data = pd.DataFrame({
        'name': df[name_col].fillna('').astype(str).str.strip(),
        'review_text': df[review_col].map(extract_review_text),
        'rating': df[rating_col].map(clean_rating) if rating_col else np.nan,
        'votes': pd.to_numeric(df[votes_col], errors='coerce').fillna(0) if votes_col else 0,
        'cuisines': df[cuisines_col].fillna('').astype(str) if cuisines_col else '',
        'location': df[location_col].fillna('').astype(str) if location_col else '',
        'cost': df[cost_col].fillna('').astype(str) if cost_col else '',
    })
    data = data[data.name.ne('')].copy()
    data['review_text'] = data.review_text.fillna('').astype(str)
    data['combined_text'] = (data.review_text + ' ' + data.cuisines + ' ' + data.location).str.strip()
    # Collapse outlet/category duplicates under a normalized restaurant name.
    data['_key'] = data.name.str.casefold().str.replace(r'\s+', ' ', regex=True)
    rows=[]
    for _, group in data.groupby('_key', sort=False):
        first = group.iloc[0]
        valid = group.dropna(subset=['rating'])
        rating = float(np.average(valid.rating, weights=np.maximum(valid.votes, 1))) if len(valid) else np.nan
        rows.append({'name': first['name'], 'review_text': ' '.join(group.review_text),
                     'combined_text': ' '.join(group.combined_text), 'rating': rating,
                     'votes': int(group.votes.sum()), 'cuisines': first.cuisines,
                     'location': first.location, 'cost': first.cost})
    result = pd.DataFrame(rows)
    result['name'] = result['name'].fillna('').astype(str).str.strip()
    result = result[result.combined_text.str.strip().ne('')].reset_index(drop=True)
    return result

def build_index(restaurants, max_features=60000):
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, strip_accents='unicode',
        max_features=max_features, min_df=2, max_df=0.98, ngram_range=(1,2), sublinear_tf=True,
        token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]{1,}\b')
    matrix = vectorizer.fit_transform(restaurants.combined_text)
    return vectorizer, matrix

def recommend(query, restaurants, matrix, limit=10, min_similarity=0.01):
    query = str(query).strip()
    if not query: raise ValueError('Enter a restaurant name.')
    matches = restaurants.index[restaurants.name.str.casefold().eq(query.casefold())].tolist()
    if not matches:
        matches = restaurants.index[restaurants.name.str.casefold().str.contains(re.escape(query.casefold()), regex=True)].tolist()
    if not matches: raise KeyError(query)
    source = matches[0]
    similarities = cosine_similarity(matrix[source], matrix).ravel()
    candidates = restaurants.copy()
    candidates['similarity'] = similarities
    candidates = candidates.drop(index=source, errors='ignore')
    candidates = candidates[(candidates.similarity >= min_similarity) & candidates.rating.notna()]
    # Similarity filters candidates; the requested final ordering prioritizes higher ratings.
    candidates = candidates.sort_values(['rating','similarity','votes'], ascending=[False,False,False])
    return source, candidates.head(limit).reset_index(drop=True)
