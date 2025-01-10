# README.md for Qualitative Data Analysis Tool

## Project Title: Qualitative Data Analysis Tool

### Description

The **Qualitative Data Analysis Tool** is a Streamlit application designed to assist researchers and analysts in performing qualitative data analysis. It leverages OpenAI's GPT-4 model to extract themes, relationships, and codes from textual data, providing visualizations to aid in understanding and interpreting qualitative information.

### Features

- **File Upload**: Supports uploading multiple document types (PDF, DOCX, TXT) for analysis.
- **Text Extraction**: Extracts text content from uploaded files for further processing.
- **Chunking**: Splits large texts into manageable chunks to comply with API token limits.
- **Data Analysis**: Utilizes OpenAI's GPT-4 for analyzing text and extracting themes and relationships.
- **Visualizations**: Generates various visual representations including treemaps, sunbursts, word clouds, heatmaps, and network graphs.
- **Downloadable Reports**: Provides an option to download the analysis results in Excel format.

### Installation

To run this application locally, ensure you have Python 3.7 or higher installed. Follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/username/my_streamlit_app.git
   cd my_streamlit_app
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set your OpenAI API key in the Streamlit secrets management:
   - Create a file named `.streamlit/secrets.toml` in your project directory and add your API key:
     ```toml
     [general]
     OPENAI_API_KEY = "your_openai_api_key_here"
     ```

### Usage

1. Run the Streamlit app:
   ```bash
   streamlit run QDA.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`.

3. Upload your documents (maximum of 7 files) using the file uploader.

4. Click on the "Analyze" button to start the analysis process.

5. View the results across different tabs including Coding, Themes, Word Frequency, Co-Occurrence, and Triangulation.

6. Download the complete analysis report in Excel format.

### Requirements

The following Python packages are required for this application:

- streamlit
- openai
- pandas
- numpy
- plotly
- networkx
- wordcloud
- PyPDF2
- python-docx
- matplotlib
- nltk

These packages are specified in the `requirements.txt` file.

### Contributing

Contributions are welcome! If you have suggestions or improvements, please fork the repository and create a pull request.

### License

This project is licensed under the MIT License - see the [LICENSE] file for details.

### Acknowledgements

- Special thanks to OpenAI for providing powerful language models.
- Thanks to Streamlit for creating an intuitive framework for building data applications.

### Contact

For any inquiries or issues related to this project, please contact smorande@gmail.com.

 