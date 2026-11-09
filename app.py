from pathlib import Path
import joblib, pandas as pd, streamlit as st
from src.recommender import prepare_restaurants, build_index, recommend
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'/'zomato.csv'; MODEL=ROOT/'models'/'zomato_tfidf.joblib'
st.set_page_config(page_title='Zomato Restaurant Recommender',page_icon='🍽️',layout='wide')
st.markdown('''<style>
:root{--ink:#241b19;--red:#e23744;--cream:#fffaf6} .stApp{background:var(--cream);color:var(--ink)}
.hero{padding:1.8rem 2rem;border-radius:20px;background:linear-gradient(115deg,#fff0e6,#ffe4e1);margin-bottom:1.2rem}
.hero h1{color:#321f1c;margin:0}.hero p{color:#70524d;margin:.35rem 0 0}
div[data-testid="stMetric"]{background:white;padding:1rem;border-radius:14px;border:1px solid #f2d8d2}
</style>''',unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>Find your next favourite restaurant</h1><p>Review driven recommendations for Bengaluru · Similarity first, then highest rated</p></div>',unsafe_allow_html=True)

@st.cache_resource(show_spinner='Preparing review language model…')
def load_system():
    if MODEL.exists():
        obj=joblib.load(MODEL)
        restaurants=obj['restaurants']
        # Older builds accidentally stored the pandas row index as each restaurant name.
        # Rebuild from source instead of exposing those invalid numeric labels in the UI.
        if restaurants['name'].map(lambda value: isinstance(value, str)).all():
            return restaurants, obj['matrix']
    raw=pd.read_csv(DATA,low_memory=False)
    restaurants=prepare_restaurants(raw)
    vectorizer,matrix=build_index(restaurants)
    joblib.dump({'restaurants':restaurants,'vectorizer':vectorizer,'matrix':matrix}, MODEL, compress=3)
    return restaurants,matrix

if not DATA.exists():
    st.warning('Add the Kaggle dataset to get started.')
    st.code('python -m pip install kaggle\n# Configure a Kaggle API token (kaggle.json)\npython src/download_dataset.py\npython src/build_model.py\nstreamlit run app.py',language='powershell')
    st.markdown('[Open the referenced Kaggle notebook](https://www.kaggle.com/code/chirag9073/zomato-recommendation-system) · [Dataset source](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants)')
    st.stop()
try:
    restaurants,matrix=load_system()
except Exception as exc:
    st.error(f'Could not prepare recommendation data: {exc}')
    st.info('Check the CSV and its review and restaurant name columns, then run `python src/build_model.py`.')
    st.stop()

c1,c2,c3=st.columns(3)
c1.metric('Restaurants indexed',f'{len(restaurants):,}')
c2.metric('With ratings',f'{restaurants.rating.notna().sum():,}')
c3.metric('With review text',f'{restaurants.review_text.str.len().gt(0).sum():,}')
left,right=st.columns([2,1])
with left:
    query=st.selectbox('Choose a restaurant',options=restaurants['name'].astype(str).sort_values(key=lambda s:s.str.casefold()).tolist(),index=None,placeholder='Start typing a restaurant name…')
with right:
    count=st.slider('Recommendations',5,20,10)
    threshold=st.slider('Minimum review similarity',0.0,0.2,0.01,0.01,help='Filters out restaurants with very little review and profile overlap.')
if query:
    try:
        source,results=recommend(query,restaurants,matrix,count,threshold)
        st.subheader(f'Because you selected {restaurants.loc[source,"name"]}')
        st.caption('We compare TF-IDF vectors made from customer reviews, cuisines, and neighbourhood. Recommendations are then ranked by restaurant rating.')
        if results.empty: st.info('No restaurants passed the similarity and rating filters. Lower the similarity threshold.')
        else:
            view=results[['name','rating','similarity','votes','cuisines','location','cost']].rename(columns={'name':'Restaurant','rating':'Rating / 5','similarity':'Review similarity','votes':'Votes','cuisines':'Cuisines','location':'Neighbourhood','cost':'Approx cost for two'})
            st.dataframe(view,use_container_width=True,hide_index=True,column_config={'Review similarity':st.column_config.ProgressColumn('Review similarity',format='%.0f%%',min_value=0,max_value=1),'Rating / 5':st.column_config.NumberColumn(format='%.1f ⭐')})
    except KeyError:
        st.error('Could not find a matching restaurant. Try another name.')

with st.expander('How the recommendations work'):
    st.markdown('''1. Clean and combine the review text with cuisine and neighbourhood descriptions.\n2. Convert language to TF-IDF unigrams and bigrams, downweighting terms common across restaurants.\n3. Measure cosine similarity between restaurant vectors.\n4. Remove the selected restaurant and low-similarity/unrated candidates.\n5. Sort the remaining restaurants by rating, then similarity and vote count.''')
    st.caption('This is an educational prototype using a historical Kaggle snapshot. Similar review language does not prove similar food quality or current availability.')
st.sidebar.header('About this project')
st.sidebar.write('Content based recommendation · NLP · TF-IDF · cosine similarity · unsupervised text similarity')
st.sidebar.caption('Dataset snapshot is historical and limited to Bengaluru. Ratings are not live.')

