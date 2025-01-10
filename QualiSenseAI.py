import streamlit as st
st.set_page_config(page_title="QualiSense-AI", page_icon="🔍", layout="wide")

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
from collections import Counter
import asyncio
from concurrent.futures import ThreadPoolExecutor

def tokenize_text(text):
    """Simple tokenizer using basic punctuation."""
    return [s.strip() + '.' for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]

tokenizer = tokenize_text

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
    sentences = tokenizer(text)
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
    
    # High-resolution Word Cloud
    text_for_cloud = ' '.join(code_data['code'].astype(str))
    wc = WordCloud(
        width=1600, 
        height=800, 
        background_color='white',
        colormap='viridis',
        max_words=150,
        min_word_length=2,
        prefer_horizontal=0.7,
        relative_scaling=0.5,
        min_font_size=10,
        max_font_size=120,
        random_state=42
    ).generate(text_for_cloud)
    
    fig_wc = plt.figure(figsize=(20, 10), dpi=300)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Frequency Analysis', pad=20, size=16)
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

def main():
    st.title("QualiSense-AI")
    
    st.markdown("""
        <style>
        .main { padding: 2rem; }
        .stTitle {
            font-size: 3rem !important;
            color: #1E3D59 !important;
            margin-bottom: 2rem !important;
        }
        .info-box {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #1E3D59;
            margin-bottom: 2rem;
        }
        .st-emotion-cache-1v0mbdj > img { width: 100px; }
        .upload-section {
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 10px;
            background-color: #ffffff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stButton>button {
            width: 100%;
            margin-top: 1rem;
            background-color: #1E3D59;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='info-box'>
        <h2>Advanced Qualitative Data Analysis Platform</h2>
        <p>QualiSense-AI leverages cutting-edge AI to provide deep insights from your qualitative data:</p>
        <ul>
            <li><strong>Automated Analysis:</strong> Advanced theme extraction and pattern recognition</li>
            <li><strong>Visual Insights:</strong> Interactive visualizations and relationship mapping</li>
            <li><strong>Multi-format Support:</strong> PDF, DOCX, and TXT file analysis</li>
            <li><strong>Comprehensive Reports:</strong> Detailed Excel reports with key metrics</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
        api_key = st.secrets.get("OPENAI_API_KEY") or st.text_input(
            "Enter OpenAI API key:",
            type="password",
            help="Your API key will be used securely and not stored"
        )
        
        if not api_key:
            st.warning("Please enter an API key to continue.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        client = OpenAI(api_key=api_key)
        
        uploaded_files = st.file_uploader(
            "Upload your documents for analysis",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Select up to 7 files to analyze"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background-color: #e9ecef; padding: 1.5rem; border-radius: 10px;'>
                <h4 style='color: #1E3D59;'>📄 Supported Formats</h4>
                <ul style='list-style-type: none; padding-left: 0;'>
                    <li>📌 PDF Documents (.pdf)</li>
                    <li>📌 Word Documents (.docx)</li>
                    <li>📌 Text Files (.txt)</li>
                </ul>
                <div style='margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #dee2e6;'>
                    <small>
                        <strong>Limits:</strong><br>
                        • Maximum 7 files<br>
                        • 12MB per file
                    </small>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    if uploaded_files:
        if len(uploaded_files) > 7:
            st.error("Maximum 7 files allowed.")
            return
        
        analyze_button = st.button(
            "🔍 Start Analysis",
            type="primary",
            help="Click to begin processing your documents"
        )
        
        if analyze_button:
            progress = st.progress(0)
            status = st.empty()
            
            try:
                texts = []
                for idx, file in enumerate(uploaded_files):
                    if file.size > 12 * 1024 * 1024:
                        st.error(f"File {file.name} exceeds 12MB limit.")
                        return
                    texts.append(extract_text(file.read(), file.name.split('.')[-1].lower()))
                    progress.progress((idx + 1) / len(uploaded_files) * 0.3)
                
                chunks = chunk_text("\n".join(texts))
                status.text("🔍 Analyzing content...")
                progress.progress(0.4)
                
                async def run_analysis():
                    analyses = await analyze_chunks(client, chunks)
                    return analyses
                
                analyses = asyncio.run(run_analysis())
                progress.progress(0.7)
                
                code_data, relationship_data, co_occurrence = merge_analyses(analyses)
                
                status.text("📊 Creating visualizations...")
                progress.progress(0.9)
                
                visualizations = create_visualizations(code_data, relationship_data, co_occurrence, "\n".join(texts))
                
                tabs = st.tabs([
                    "📊 Code Distribution",
                    "🎯 Theme Analysis",
                    "☁️ Word Frequency",
                    "🔄 Co-Occurrence",
                    "🔍 Network Analysis"
                ])
                
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
                        "📥 Download Complete Analysis",
                        buffer.getvalue(),
                        "QualiSense_Analysis_Report.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Download a comprehensive Excel report of all analyses"
                    )
                
                status.text("✅ Analysis complete!")
                progress.empty()
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                status.text("❌ Analysis failed!")
                progress.empty()

if __name__ == "__main__":
    st.set_page_config(page_title="QualiSense-AI", page_icon="🔍", layout="wide")
    main()