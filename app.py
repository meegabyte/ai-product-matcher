#import sys
#print(sys.executable)

import streamlit as st
import pandas as pd
import pypdf
import os
import io
import ollama
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata
from rapidfuzz import fuzz 

#konfig
st.set_page_config(page_title="AI Prodkt Matching v4 ", layout="wide", page_icon="✨")

#style
st.markdown("""
<style>
    .stDataFrame { font-size: 14px; }
    div[data-testid="metric-container"] { background-color: #262730; border: 1px solid #464b5f; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)
#UI

st.title("✨ AI Product Matching")
st.info(" © Miłosz Mielcarek")

