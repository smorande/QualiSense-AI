import streamlit as st
st.set_page_config(layout="wide", page_title="QualiSense-AI", page_icon="🔍")

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

# Enhanced UI Configuration
st.markdown("""
<style>
    .app-title {
    background-color: rgba(33, 150, 243, 0.3);
    padding: 10px 20px;
    border-radius: 8px;
    display: inline-block;
    margin-bottom: 20px;
    font-weight: 600;
    box-shadow: 0 2px 4px rgba(33, 150, 243, 0.2);
}
    
    .donate-button {
        background-color: #0070ba;
        border: 2px solid #169BD7;
        color: #ffffff !important;
        padding: 8px 16px;
        border-radius: 4px;
        text-decoration: none;
        display: inline-block;
        font-weight: 500;
        font-size: 14px;
        margin: 5px 0;
        text-align: center;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .donate-button:hover {
        background-color: #005ea6;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .donate-section {
        padding: 8px;
        margin: 8px 0;
        border-radius: 4px;
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

def create_welcome_page():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<h1 class="app-title">🔍 QualiSense-AI</h1>', unsafe_allow_html=True)
        st.markdown("""
        ### Advanced Qualitative Data Analysis Platform
        
        QualiSense-AI leverages cutting-edge AI to transform your qualitative data into actionable insights:
        
        ✨ **Key Features**:
        - Multi-document analysis support (PDF, DOCX, TXT)
        - AI-powered thematic analysis
        - High-resolution visualizations
        - Advanced relationship mapping
        - Comprehensive Excel reports
        
        🎯 **Perfect for**: Researchers, Data Scientists, and Analysts
        """)
    
    with col2:
        st.markdown("""
        ### Quick Start Guide
        1. Enter your OpenAI API key
        2. Upload documents (max 7 files)
        3. Click 'Analyze' to begin
        4. Download comprehensive results
        
        ⚡ **Supported File Types**:
        - PDF (max 12MB)
        - DOCX
        - TXT
        """)
        
        # Add Donate Section
        st.markdown("""
        <div class="donate-section">
            <h3>💝 Support QualiSense-AI</h3>
            <p>If you find this tool valuable, consider supporting its development:</p>
            <div class="donation-options">
                <a href="https://www.paypal.com/paypalme/smorande90" target="_blank" class="donate-button">
                    Support via PayPal
                </a>
                <p class="donation-note" style="font-size: 12px; color: #666; margin-top: 8px;">
                    Your support helps maintain and improve QualiSense-AI ❤️
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def tokenize_text(text):
    """Simple tokenizer using basic punctuation."""
    return [s.strip() + '.' for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]

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
    sentences = tokenize_text(text)
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
                return json.loads(response.choices[0].message.content)
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
    
    # Enhanced Treemap
    treemap = px.treemap(
        code_data,
        path=['theme', 'code'],
        values='value',
        title='Code Distribution Analysis',
        color='value',
        color_continuous_scale='Viridis'
    ).update_layout(height=600, title_x=0.5, title_font_size=20)
    visualizations.append(treemap)
    
    # Enhanced Sunburst
    sunburst = px.sunburst(
        code_data,
        path=['theme', 'code'],
        values='value',
        title='Thematic Analysis Overview',
        color='value',
        color_continuous_scale='Viridis'
    ).update_layout(height=600, title_x=0.5, title_font_size=20)
    visualizations.append(sunburst)
    
    # High-Resolution WordCloud
    plt.figure(figsize=(16, 8), dpi=300)
    text_for_cloud = ' '.join(code_data['code'].astype(str))
    wc = WordCloud(
        width=1600, 
        height=800, 
        background_color='white',
        colormap='viridis',
        max_words=150,
        min_word_length=2,
        prefer_horizontal=0.7,
        collocations=True,
        contour_width=1,
        contour_color='steelblue',
        font_path=None,
        random_state=42
    ).generate(text_for_cloud)
    
    fig_wc = plt.figure(figsize=(16, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Frequency Analysis', fontsize=20, pad=20)
    visualizations.append(fig_wc)
    
    # Enhanced Heatmap
    heatmap = px.imshow(
        co_occurrence_matrix,
        title='Code Co-occurrence Matrix',
        color_continuous_scale='Viridis',
        aspect='auto'
    ).update_layout(
        height=600,
        title_x=0.5,
        title_font_size=20,
        xaxis_title="Codes",
        yaxis_title="Codes"
    )
    visualizations.append(heatmap)
    
    # Enhanced Network Graph
    if len(relationship_data) > 0:
        G = nx.from_pandas_edgelist(
            relationship_data.nlargest(20, 'weight'),
            'source', 'target', 'weight'
        )
        pos = nx.spring_layout(G, k=1.5, iterations=50)
        
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='rgba(150,150,150,0.5)'),
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
            marker=dict(
                size=15,
                color='#17a2b8',
                line=dict(color='white', width=1)
            )
        )
        
        network = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              title={
                                  'text': 'Code Relationship Network',
                                  'y': 0.95,
                                  'x': 0.5,
                                  'xanchor': 'center',
                                  'yanchor': 'top',
                                  'font': dict(size=20)
                              },
                              showlegend=False,
                              height=600,
                              margin=dict(b=20,l=5,r=5,t=40),
                              plot_bgcolor='white',
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                          ))
    else:
        network = create_empty_figure("No network data available")
    
    visualizations.append(network)
    return visualizations

def create_empty_figure(message="No data available"):
    return go.Figure().update_layout(
        annotations=[{
            "text": message,
            "showarrow": False,
            "font": {"size": 20},
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": 0.5
        }],
        height=600
    )

def main():
    create_welcome_page()
    
    api_key = st.secrets.get("OPENAI_API_KEY") or st.text_input(
        "🔑 Enter OpenAI API key:",
        type="password",
        help="Your API key will be used securely for analysis"
    )
    
    if not api_key:
        st.info("👆 Please enter an API key to begin your analysis journey.")
        return

    client = OpenAI(api_key=api_key)
    
    st.markdown('<div class="upload-message">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "📁 Upload your documents",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Upload up to 7 documents (PDF, DOCX, or TXT)"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_files:
        if len(uploaded_files) > 7:
            st.error("⚠️ Maximum 7 files allowed.")
            return
            
        if st.button("🚀 Start Analysis"):
            progress = st.progress(0)
            status = st.empty()
            
            try:
                with st.spinner("📚 Processing documents..."):
                    texts = []
                    for idx, file in enumerate(uploaded_files):
                        if file.size > 12 * 1024 * 1024:
                            st.error(f"⚠️ File {file.name} exceeds 12MB limit.")
                            return
                        texts.append(extract_text(file.read(), file.name.split('.')[-1].lower()))
                        progress.progress((idx + 1) / len(uploaded_files))
                
                chunks = chunk_text("\n".join(texts))
                status.info("🤖 AI analysis in progress...")
                
                async def run_analysis():
                    analyses = await analyze_chunks(client, chunks)
                    return analyses
                
                analyses = asyncio.run(run_analysis())
                code_data, relationship_data, co_occurrence = merge_analyses(analyses)
                
                status.success("📊 Creating visualizations...")
                visualizations = create_visualizations(code_data, relationship_data, co_occurrence, "\n".join(texts))
                
                # Create tabs with descriptive icons
                tabs = st.tabs([
                    "📊 Code Distribution",
                    "🎯 Thematic Analysis",
                    "📝 Word Frequency",
                    "🔄 Code Co-occurrence",
                    "🕸️ Network Analysis"
                ])
                
                for tab, viz in zip(tabs, visualizations):
                    with tab:
                        if isinstance(viz, plt.Figure):
                            st.pyplot(viz)
                        else:
                            st.plotly_chart(viz, use_container_width=True)
                
                # Generate downloadable report
                with io.BytesIO() as buffer:
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        # Write main data
                        workbook = writer.book
                        header_format = workbook.add_format({
                            'bold': True,
                            'fg_color': '#17a2b8',
                            'font_color': 'white',
                            'border': 1
                        })
                        
                        # Coding Analysis sheet
                        code_data.to_excel(writer, 'Coding Analysis', index=False)
                        worksheet = writer.sheets['Coding Analysis']
                        for col_num, value in enumerate(code_data.columns.values):
                            worksheet.write(0, col_num, value, header_format)
                            worksheet.set_column(col_num, col_num, 15)
                            
                        # Theme Relationships sheet
                        relationship_data.to_excel(writer, 'Theme Relationships', index=False)
                        worksheet = writer.sheets['Theme Relationships']
                        for col_num, value in enumerate(relationship_data.columns.values):
                            worksheet.write(0, col_num, value, header_format)
                            worksheet.set_column(col_num, col_num, 15)
                            
                        # Co-Occurrence Matrix sheet
                        co_occurrence.to_excel(writer, 'Co-Occurrence Matrix')
                        worksheet = writer.sheets['Co-Occurrence Matrix']
                        for col_num, value in enumerate(co_occurrence.columns.values):
                            worksheet.write(0, col_num + 1, value, header_format)
                            worksheet.set_column(col_num + 1, col_num + 1, 15)
                        
                        # Analysis Summary sheet with enhanced formatting
                        summary_data = pd.DataFrame({
                            'Metric': [
                                'Total Themes',
                                'Total Codes',
                                'Total Relationships',
                                'Network Density',
                                'Analysis Date',
                                'Documents Analyzed',
                                'Total Word Count'
                            ],
                            'Value': [
                                len(code_data['theme'].unique()),
                                len(code_data['code'].unique()),
                                len(relationship_data),
                                len(relationship_data)/(len(code_data['code'].unique())*(len(code_data['code'].unique())-1)/2) if len(code_data['code'].unique()) > 1 else 0,
                                pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                                len(uploaded_files),
                                sum(len(text.split()) for text in texts)
                            ]
                        })
                        summary_data.to_excel(writer, 'Analysis Summary', index=False)
                        worksheet = writer.sheets['Analysis Summary']
                        for col_num, value in enumerate(summary_data.columns.values):
                            worksheet.write(0, col_num, value, header_format)
                            worksheet.set_column(col_num, col_num, 20)
                    
                    # Add download button with styling
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.download_button(
                            "📥 Download Complete Analysis Report",
                            buffer.getvalue(),
                            "QualiSense_AI_Analysis_Report.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                
                status.success("✨ Analysis complete! Navigate through the tabs to explore your results.")
                progress.empty()
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                status.error("Analysis failed! Please try again.")
                progress.empty()

if __name__ == "__main__":
    main()