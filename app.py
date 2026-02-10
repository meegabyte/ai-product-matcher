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

# 1.Overlap, ekstrakcja przez llm 
def clean_product_line(line):
    line = re.sub(r'^\d+[\.\s]+', '', line)
    match = re.search(r'\s+\d+\s+(op|szt|kpl|opak)', line)
    if match:
        line = line[:match.start()]
    return line.strip()

def query_llama_overlap(chunk_text):
    prompt = f"""
    Zadanie: Wypisz listę produktów z fragmentu faktury.
    1. Przepisz nazwy 1:1.
    2. Pomiń nagłówki, daty i podsumowania.
    3. Jeden produkt w jednej linii.
    
    FRAGMENT:
    {chunk_text}
    """
    try:
        response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt}])
        content = response['message']['content']
        cleaned = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('- ') or line.startswith('* '): line = line[2:]

            forbidden_words = [
                'strona', 'faktura', 'razem', 'netto', 'brutto', 'kwota', 
                'sprzedawca', 'nabywca', 'data', 'termin', 
                'lista', 'produktów', 'poniżej', 'oto', 'jeśli', 'chcesz', 'mogę', 'fragment'
            ]

            if len(line) > 3:
                #Sprawdzenie czy linia nie zawiera zakazanych słów (case insensitive)
                if any(bad_word in line.lower() for bad_word in forbidden_words):
                    continue #pomijanie, śmieci z faktury

                final_name = clean_product_line(line)

                if len(final_name) > 2:
                    cleaned.append(final_name)
        return cleaned
    except:
        return []

def extract_products_overlap(text):
    
    lines = text.split('\n')
    all_products = []
    
    
    chunk_size = 45
    overlap = 15
    step = chunk_size - overlap
    chunks = []
    
    
    
    for i in range(0, len(lines), step):
        chunk_lines = lines[i : i + chunk_size]
        if len(chunk_lines) < 5: continue
        chunks.append("\n".join(chunk_lines))
    
    progress_bar = st.progress(0)
    
    # 1. Pobieranie wszystkiego
    for i, chunk in enumerate(chunks):
        found = query_llama_overlap(chunk)
        all_products.extend(found)
        progress_bar.progress((i + 1) / len(chunks))
        
    progress_bar.empty()
    
    unique_products = []
    
    for item in all_products:
        item_clean = item.strip()
        if not item_clean: continue
        
        #Sprawdzanie, czy ten produkt już jest na liście (podobieństwo > 85%)
        is_duplicate = False
        for existing in unique_products:
            # fuzz.ratio porównuje dwa napisy i daje wynik 0-100
            if fuzz.ratio(item_clean.lower(), existing.lower()) > 85:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_products.append(item_clean)
            
    return unique_products

#2.DOPASOWANIE 

def clean_text_for_search(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text

def find_matches_tfidf(query, vectorizer, tfidf_matrix, df_db, name_col):
    try:
        query_clean = clean_text_for_search(query)
        if not query_clean.strip(): return []
        
        query_vec = vectorizer.transform([query_clean])
        sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = sims.argsort()[-3:][::-1]
        
        matches = []
        for idx in top_indices:
            score = sims[idx] * 100
            name = df_db.iloc[idx][name_col]
            matches.append((name, score))
        return matches
    except:
        return []



current_dir = os.path.dirname(os.path.abspath(__file__))
DB_FILENAME = os.path.join(current_dir, 'wyeksportowane-produkty.csv')

@st.cache_resource
def load_db():
    if not os.path.exists(DB_FILENAME): return None
    try:
        #Próba 1: utf-8
        df = pd.read_csv(DB_FILENAME, sep=';', on_bad_lines='skip')
        if df.shape[1] < 2: df = pd.read_csv(DB_FILENAME, sep=',', on_bad_lines='skip')
        return df.fillna('').astype(str)
    except:
        try:
            #Próba 2: polski (cp1250)
            df = pd.read_csv(DB_FILENAME, sep=';', encoding='cp1250', on_bad_lines='skip')
            if df.shape[1] < 2: df = pd.read_csv(DB_FILENAME, sep=',', encoding='cp1250', on_bad_lines='skip')
            return df.fillna('').astype(str)
        except: return None

df_db = load_db()

if df_db is not None:
    cols = df_db.columns
    name_col = next((c for c in cols if 'name' in c.lower() or 'nazwa' in c.lower()), cols[1])
    
    
    vectorizer = TfidfVectorizer(
        preprocessor=clean_text_for_search,
        analyzer='word',
        token_pattern=r'(?u)\b\w+\b',
        ngram_range=(1, 3),  #Szukaj też trójek słów (np. "Rękawice nitrylowe L")
        min_df=1,
        stop_words=['op', 'szt', 'opak', 'kpl'] #ignoruj jednostki miary
    )
    tfidf_matrix = vectorizer.fit_transform(df_db[name_col])
    
    st.sidebar.success(f" Pomyślnie wczytano wyeksportowane-produkty.csv. Baza: {len(df_db)} poz.")
    
    uploaded_file = st.file_uploader("Wgraj fakturę (PDF)", type=['pdf'])

    if uploaded_file and st.button("🚀 URUCHOM ANALIZĘ"):
        status = st.empty()
        
        # 1.ODCZYT
        pdf_reader = pypdf.PdfReader(io.BytesIO(uploaded_file.getvalue()))
        raw_text = ""
        for page in pdf_reader.pages:
            raw_text += page.extract_text() + "\n"
            
        # 2.EKSTRAKCJA
        status.text("Krok 1: Analiza Llama z inteligentnym usuwaniem duplikatów...")
        products = extract_products_overlap(raw_text)
        
        if not products:
            st.error("Nie znaleziono produktów. Sprawdź plik.")
        else:
            st.success(f"Znaleziono {len(products)} unikalnych pozycji.")
            
            with st.sidebar:
                 st.markdown("---")
                 st.write("🕵️ Podgląd odczytu AI ")
                 st.caption("Wyszukane 10 pierwszych produktów przez SI:")
                 for p in products[:10]:
                     st.code(p, language="text")
                 st.markdown("---")

            results = []
            total = len(products)
            progress_bar = st.progress(0)
            
            # 3.DOPASOWANIE
            for i, item in enumerate(products):
                status.text(f"Dopasowywanie ({i+1}/{total}): {item}")
                
                matches = find_matches_tfidf(item, vectorizer, tfidf_matrix, df_db, name_col)
                
                match_name = ""
                score = 0
                alternatives_str = ""
                final_status = "BRAK"
                
                if matches:
                    top_match = matches[0]
                    if top_match[1] > 15:
                        match_name = top_match[0]
                        score = top_match[1]
                        
                        alts = [m[0] for m in matches[1:]]
                        alternatives_str = " | ".join(alts)
                        
                        if score >= 50: final_status = "PEWNE"
                        elif score >= 25: final_status = "WERYFIKACJA"
                        else: final_status = "BRAK"
                    else:
                        match_name = ""
                        final_status = "BRAK"

                results.append({
                    "Nazwa z Faktury": item,
                    "Dopasowanie Systemu": match_name,
                    "Alternatywy": alternatives_str,
                    "Pewność": f"{score:.1f}%" if match_name else "-",
                    "Status": final_status
                })
                progress_bar.progress((i+1)/total)
            
            status.empty()
            
            # 4.wyniki
            df_res = pd.DataFrame(results)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("✅ Pewne", len(df_res[df_res['Status']=='PEWNE']))
            c2.metric("⚠️ Do Weryfikacji", len(df_res[df_res['Status']=='WERYFIKACJA']))
            c3.metric("⛔ Brak", len(df_res[df_res['Status']=='BRAK']))

            def color_rows(row):
                s = row['Status']
                if s == 'PEWNE': return ['background-color: #d4edda; color: black'] * len(row)
                if s == 'WERYFIKACJA': return ['background-color: #fff3cd; color: black'] * len(row)
                return ['background-color: #f8d7da; color: black'] * len(row)

            st.dataframe(df_res.style.apply(color_rows, axis=1), use_container_width=True, height=700)
            
            csv = df_res.to_csv(sep=';', index=False).encode('utf-8-sig')
            st.download_button("💾 Pobierz Wyniki (CSV)", csv, "wynik_fixed.csv")
else:
    st.sidebar.error(f"Nie znaleziono bazy danych. Sprawdź, czy plik istnieje.")