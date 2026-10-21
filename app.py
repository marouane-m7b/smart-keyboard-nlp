"""Run with uv run streamlit run app.py."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from src.pipeline import EOS, NGramModel, InterpolatedModel, load_documents, split_by_document, backoff_top_k

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Smart Keyboard | NLP Lab", page_icon="⌨️", layout="wide")
st.markdown('<style>.block-container{max-width:1100px;padding-top:3rem}h1{letter-spacing:-.04em!important}[data-testid="stMetric"]{background:var(--secondary-background-color);padding:1rem;border-radius:14px}</style>', unsafe_allow_html=True)

@st.cache_resource
def get_models(k):
    docs = load_documents(ROOT / 'data/corpus.csv')
    train, test = split_by_document(docs)
    return [NGramModel(train, n, k) for n in (3, 2, 1)], InterpolatedModel(train, k=k), docs, train, test

def set_sentence(text):
    st.session_state.sentence = text

def append_word(word):
    text = st.session_state.sentence.rstrip()
    set_sentence('' if word == EOS else (text + ('' if word == '.' else ' ') + word).lstrip())

with st.sidebar:
    st.markdown('### ⌨️ Smart Keyboard')
    st.caption('ENSET · NLP · Lab 02')
    strategy = st.selectbox('Prediction strategy', ['Hard back-off', 'Interpolated', 'Trigram', 'Bigram', 'Unigram'])
    k = st.select_slider('Add-k smoothing', [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0], value=0.1)
    st.caption('Back-off uses the longest observed context. Interpolation mixes trigram, bigram and unigram probabilities with weights 60%, 30%, 10%.')
    st.divider()
    st.caption('Training: first 80% of documents. Evaluation: last 20%, held out from training.')

(tri, bi, uni), interp, docs, train, test = get_models(k)
st.caption('LANGUAGE MODELS / NEXT-WORD PREDICTION')
st.title('A little help with your next word.')
st.write('Start a sentence, explore the predictions, and tap a suggestion to keep writing.')
keyboard, evaluation, about = st.tabs(['⌨️ Keyboard', '📊 Evaluation', '📖 How it works'])
with keyboard:
    if 'sentence' not in st.session_state:
        st.session_state.sentence = 'the happy girls'
    st.text_area('Your sentence', key='sentence', height=140)
    words = st.session_state.sentence.lower().replace('.', ' . ').split()
    unknown = sorted({w for w in words if w not in uni.vocab})
    if unknown:
        st.warning('Outside the training vocabulary (mapped to <unk>): ' + ', '.join(unknown))
    if strategy == 'Hard back-off':
        suggestions, order = backoff_top_k(tri, bi, uni, words, 3)
    else:
        model = {'Interpolated': interp, 'Trigram': tri, 'Bigram': bi, 'Unigram': uni}[strategy]
        suggestions, order = model.top_k(words, 3), strategy.lower()
    st.subheader('Up next')
    st.caption(f'Using {order} · k = {k} · click a word to append it')
    for col, (word, prob) in zip(st.columns(3), suggestions):
        col.button('↵ New sentence' if word == EOS else word, key='suggest_' + word, on_click=append_word, args=(word,), width='stretch')
        col.caption(f'{prob:.2%} model probability')
    st.divider()
    st.caption('TRY A STARTER')
    for col, example in zip(st.columns(3), ['the cats eat the', 'the teachers', 'the children find']):
        col.button(example, on_click=set_sentence, args=(example,), width='stretch')
    st.button('Clear sentence', on_click=set_sentence, args=('',))
    st.caption('A tiny synthetic English corpus: these predictions demonstrate n-grams, not general English fluency. Probabilities are model estimates, not renormalized over the three suggestions.')
with evaluation:
    path = ROOT / 'results/metrics.json'
    if path.exists():
        raw = path.read_text(encoding='utf-8')
        metrics = json.loads(raw)
        a, b, c = st.columns(3)
        a.metric('Training sentences', f'{len(train):,}')
        b.metric('Held-out sentences', f'{len(test):,}')
        c.metric('Back-off top-3 accuracy', f"{metrics['backoff_top3']:.1%}")
        st.caption('Saved evaluation uses the fixed document split and the k values below. Sidebar settings affect the keyboard only. End-of-sentence tokens count as prediction targets.')
        frame = pd.DataFrame(metrics['rows'])
        frame.columns = ['Model', 'Perplexity', 'Zero-probability share', 'Top-3 accuracy']
        st.dataframe(frame, hide_index=True, width='stretch')
        st.caption('Accuracy and zero-probability share are fractions from 0 to 1. Lower perplexity and higher accuracy are better. Infinite perplexity means at least one target had zero probability.')
        st.image(str(ROOT / 'results/ngram_results.png'), width='stretch')
        with st.expander('Inspect prediction failures'):
            st.code((ROOT / 'results/failures.txt').read_text(encoding='utf-8'), language=None)
        st.download_button('Download evaluation JSON', raw, 'metrics.json', 'application/json')
    else:
        st.info('Run uv run python -m src.evaluate to generate results.')
with about:
    st.subheader('Count the context. Predict the next word.')
    st.latex(r'P(w\mid c)=\frac{\operatorname{count}(c,w)+k}{\operatorname{count}(c)+k|V|}')
    st.write('Unigrams ignore context, bigrams use one preceding word, and trigrams use two. Add-k smoothing assigns nonzero probabilities to unseen combinations. Hard back-off falls back to shorter observed contexts.')
    st.write(f'The corpus has {len(docs)} documents and {len(train)+len(test):,} sentences (seed 42). Its grammar is much more regular than real English, making perplexities unusually low. Vocabulary groups cover animals, foods, people, places, and objects. Gold POS tags are included; watch can be a noun or verb.')
    st.code('uv sync\nuv run python -m src.evaluate\nuv run streamlit run app.py', language='bash')
