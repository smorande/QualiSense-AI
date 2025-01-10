import streamlit as st
from openai import OpenAI
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from wordcloud import WordCloud
import PyPDF2
import docx
import matplotlib.pyplot as plt
import io
import json
from nltk.tokenize import sent_tokenize
from collections import Counter, defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import nltk
import os

# Create NLTK data directory if it doesn't exist
nltk_data_dir = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

# Set NLTK data path
nltk.data.path.append(nltk_data_dir)

# Download punkt if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    except Exception as e:
        st.error(f"Error downloading NLTK data: {str(e)}")
        # Fallback tokenizer function
        def sent_tokenize(text):
            return text.split('.')

@st.cache_data(ttl=3600)
def extract_text(file_content, file_type):
    try:
        if file_type == 'pdf':
            return "".join(page.extract_text() for page in PyPDF2.PdfReader(io.BytesIO(file_content)).pages)
        elif file_type == 'docx':
            return "\n".join(p.text for p in docx.Document(io.BytesIO(file_content)).paragraphs)
        return file_content.decode('utf-8')
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return ""

@st.cache_data(ttl=3600)
def chunk_text(text, max_tokens=4000):
    chunks = []
    sentences = sent_tokenize(text)
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_tokens:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

async def analyze_chunks(client, chunks):
    sem = asyncio.Semaphore(3)
    
    async def process_chunk(chunk):
        async with sem:
            try:
                sanitized_chunk = chunk.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                response = await asyncio.to_thread(
                    lambda: client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a text analysis expert. Return a valid JSON object with exactly this structure, ensuring all strings are properly escaped: {\"themes\":[{\"name\":\"string\",\"description\":\"string\",\"codes\":[{\"name\":\"string\",\"frequency\":1}]}],\"relationships\":[{\"source\":\"string\",\"target\":\"string\",\"weight\":1,\"type\":\"string\"}]}"},
                            {"role": "user", "content": f"Analyze and return strictly valid JSON:\n{sanitized_chunk}"}
                        ],
                        temperature=0,
                        response_format={"type": "json_object"}
                    )
                )
                content = response.choices[0].message.content
                return json.loads(content)
            except json.JSONDecodeError as e:
                st.warning(f"JSON parsing error: {str(e)}")
                return {"themes": [], "relationships": []}
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
                return {"themes": [], "relationships": []}

    return await asyncio.gather(*[process_chunk(chunk) for chunk in chunks])

@st.cache_data(ttl=3600)
def merge_analyses(analyses):
    code_data = []
    relationships = []
    code_counter = Counter()
    
    for analysis in analyses:
        if isinstance(analysis, str):
            try:
                analysis = json.loads(analysis)
            except json.JSONDecodeError:
                continue
                
        for theme in analysis.get('themes', []):
            theme_name = theme.get('name', '')
            theme_desc = theme.get('description', '')
            
            for code in theme.get('codes', []):
                if not isinstance(code, dict):
                    continue
                    
                try:
                    freq = int(code.get('frequency', 1))
                    code_name = str(code.get('name', ''))
                    code_data.append({
                        'theme': theme_name,
                        'code': code_name,
                        'value': freq,
                        'theme_description': theme_desc
                    })
                    code_counter[code_name] += freq
                except (ValueError, TypeError):
                    continue
        
        for r in analysis.get('relationships', []):
            if not isinstance(r, dict):
                continue
                
            try:
                weight = int(r.get('weight', 1))
                relationships.append({
                    'source': str(r.get('source', '')),
                    'target': str(r.get('target', '')),
                    'weight': weight,
                    'type': str(r.get('type', ''))
                })
            except (ValueError, TypeError):
                continue
    
    code_df = pd.DataFrame(code_data) if code_data else pd.DataFrame(columns=['theme', 'code', 'value', 'theme_description'])
    rel_df = pd.DataFrame(relationships) if relationships else pd.DataFrame(columns=['source', 'target', 'weight', 'type'])
    
    codes = list(code_counter)
    matrix = pd.DataFrame(0, index=codes, columns=codes) if codes else pd.DataFrame()
    
    for rel in relationships:
        src, tgt = rel['source'], rel['target']
        if src in codes and tgt in codes:
            matrix.loc[src, tgt] = matrix.loc[tgt, src] = rel['weight']
    
    return code_df, rel_df, matrix

def create_visualizations(code_data, relationship_data, co_occurrence_matrix, text):
    if code_data.empty or relationship_data.empty:
        return [create_empty_figure() for _ in range(5)]

    visualizations = []
    
    # Treemap
    treemap = px.treemap(
        code_data,
        path=['theme', 'code'],
        values='value',
        title='Code Distribution',
        color='value'
    ).update_layout(height=600)
    visualizations.append(treemap)
    
    # Sunburst
    sunburst = px.sunburst(
        code_data,
        path=['theme', 'code'],
        values='value',
        title='Thematic Analysis'
    ).update_layout(height=600)
    visualizations.append(sunburst)
    
    # Word Cloud
    text_for_cloud = ' '.join(code_data['code'].astype(str))
    wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis', 
                  max_words=100, min_word_length=2).generate(text_for_cloud)
    fig_wc = plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Frequency Analysis')
    visualizations.append(fig_wc)
    
    # Heatmap
    heatmap = px.imshow(
        co_occurrence_matrix,
        title='Code Co-occurrence Matrix',
        color_continuous_scale='Viridis'
    ).update_layout(height=600)
    visualizations.append(heatmap)
    
    # Network Graph
    if len(relationship_data) > 0:
        G = nx.from_pandas_edgelist(
            relationship_data.nlargest(20, 'weight'),
            'source', 'target', 'weight'
        )
        pos = nx.spring_layout(G, k=1)
        
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=list(G.nodes()),
            textposition="top center",
            marker=dict(size=10, color='blue')
        )
        
        network = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              title='Code Relationship Network',
                              showlegend=False,
                              height=600,
                              margin=dict(b=20,l=5,r=5,t=40)
                          ))
    else:
        network = create_empty_figure("No network data available")
    
    visualizations.append(network)
    return visualizations

def create_empty_figure(message="No data available"):
    return go.Figure().update_layout(
        annotations=[{"text": message, "showarrow": False, "font": {"size": 20}}]
    )

async def main():
    st.set_page_config(layout="wide")
    st.title("Qualitative Data Analysis Tool")
    
    api_key = st.secrets.get("OPENAI_API_KEY") or st.text_input("Enter OpenAI API key:", type="password")
    if not api_key:
        st.warning("Please enter an API key to continue.")
        return

    client = OpenAI(api_key=api_key)
    
    uploaded_files = st.file_uploader("Upload documents (max 7)", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
    
    if uploaded_files:
        if len(uploaded_files) > 7:
            st.error("Maximum 7 files allowed.")
            return
            
        if st.button("Analyze"):
            progress = st.progress(0)
            status = st.empty()
            
            try:
                texts = []
                for file in uploaded_files:
                    if file.size > 12 * 1024 * 1024:
                        st.error(f"File {file.name} exceeds 12MB limit.")
                        return
                    texts.append(extract_text(file.read(), file.name.split('.')[-1].lower()))
                
                chunks = chunk_text("\n".join(texts))
                status.text("Analyzing chunks...")
                
                analyses = await analyze_chunks(client, chunks)
                code_data, relationship_data, co_occurrence = merge_analyses(analyses)
                
                status.text("Creating visualizations...")
                visualizations = create_visualizations(code_data, relationship_data, co_occurrence, "\n".join(texts))
                
                tabs = st.tabs(["Coding", "Themes", "Word Frequency", "Co-Occurrence", "Triangulation"])
                
                for tab, viz in zip(tabs, visualizations):
                    with tab:
                        if isinstance(viz, plt.Figure):
                            st.pyplot(viz)
                        else:
                            st.plotly_chart(viz, use_container_width=True)
                
                with io.BytesIO() as buffer:
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        code_data.to_excel(writer, 'Coding Analysis', index=False)
                        relationship_data.to_excel(writer, 'Theme Relationships', index=False)
                        co_occurrence.to_excel(writer, 'Co-Occurrence Matrix')
                        
                        summary_data = pd.DataFrame({
                            'Metric': ['Total Themes', 'Total Codes', 'Total Relationships', 'Network Density'],
                            'Value': [
                                len(code_data['theme'].unique()),
                                len(code_data['code'].unique()),
                                len(relationship_data),
                                len(relationship_data)/(len(code_data['code'].unique())*(len(code_data['code'].unique())-1)/2) if len(code_data['code'].unique()) > 1 else 0
                            ]
                        })
                        summary_data.to_excel(writer, 'Analysis Summary', index=False)
                    
                    st.download_button(
                        "Download Complete Analysis",
                        buffer.getvalue(),
                        "qualitative_analysis_results.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                status.text("Analysis complete!")
                progress.empty()
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                status.text("Analysis failed!")
                progress.empty()

if __name__ == "__main__":
    asyncio.run(main())