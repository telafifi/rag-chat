# RAG Chatbot with Anthropic Claude

A Retrieval-Augmented Generation (RAG) chatbot that allows you to chat with your PDF documents using Anthropic's Claude language model. The system processes PDF files, creates vector embeddings, and provides intelligent responses based on document content.

## Features

- **Multi-PDF Support**: Process and chat with multiple PDF documents simultaneously
- **Anthropic Claude Integration**: Uses Claude-3-Haiku for fast and cost-effective responses
- **Vector Search**: Leverages ChromaDB for efficient document retrieval
- **Source Attribution**: Shows which document(s) information comes from
- **Web Interface**: Clean Gradio-based chat interface
- **Local Processing**: All processing happens locally (except LLM calls)

## Architecture

- **Document Processing**: PyPDF for PDF text extraction
- **Text Chunking**: RecursiveCharacterTextSplitter for optimal chunk sizes
- **Embeddings**: HuggingFace sentence-transformers for local embeddings
- **Vector Store**: ChromaDB for similarity search
- **LLM**: Anthropic Claude for response generation
- **Interface**: Gradio for web-based chat

## Prerequisites

- Python 3.8+
- Anthropic API key
- Required Python packages (see Installation)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd rag-chat
   ```

2. **Install dependencies**:
   ```bash
   pip install langchain-anthropic langchain-community langchain-chroma sentence-transformers gradio python-dotenv
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```bash
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

4. **Prepare your documents**:
   - Place your PDF files in the `data/` folder
   - The system will automatically process all PDFs in this directory

## Usage

### 1. Populate the Database

First, ingest your PDF documents into the vector database:

```bash
python ingest_database.py
```

This will:
- Load all PDF files from the `data/` folder
- Split them into chunks (1000 characters with 200 overlap)
- Create embeddings using HuggingFace transformers
- Store everything in ChromaDB

### 2. Start the Chatbot

Run the chatbot using the provided script:

```bash
./run_chatbot.sh
```

Or directly with Python:

```bash
python chatbot.py
```

### 3. Access the Interface

Open your web browser and navigate to:
```
http://localhost:7860
```

## Adding New Documents

To add new PDF documents:

1. **Add PDFs to data folder**:
   ```bash
   cp /path/to/new/document.pdf data/
   ```

2. **Re-ingest the database**:
   ```bash
   rm -rf chroma_db
   python ingest_database.py
   ```

3. **Restart the chatbot**:
   ```bash
   ./run_chatbot.sh
   ```

## Configuration

### Document Processing
- **Chunk Size**: 1000 characters (configurable in `ingest_database.py`)
- **Chunk Overlap**: 200 characters
- **Retrieval**: Top 10 most similar chunks

### Model Settings
- **LLM**: Claude-3-Haiku (fast and cost-effective)
- **Temperature**: 0.5 (balanced creativity/consistency)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2

### Customization
You can modify these settings in the respective files:
- `chatbot.py`: LLM model, temperature, retrieval parameters
- `ingest_database.py`: Chunk size, overlap, embedding model

## File Structure

```
rag-chat/
├── README.md                 # This file
├── .env                     # Environment variables (not tracked)
├── .gitignore              # Git ignore rules
├── chatbot.py              # Main chatbot application
├── ingest_database.py       # Database ingestion script
├── run_chatbot.sh          # Convenience script to run chatbot
├── data/                   # PDF documents folder
│   └── *.pdf              # Your PDF files
└── chroma_db/             # Vector database (auto-generated)
    ├── chroma.sqlite3     # SQLite database
    └── ...                # Vector storage files
```

## Troubleshooting

### Common Issues

1. **"No module named 'langchain_anthropic'"**:
   ```bash
   pip install langchain-anthropic
   ```

2. **"No relevant docs were retrieved"**:
   - Check if PDFs are in the `data/` folder
   - Re-run `ingest_database.py`
   - Verify PDFs are not corrupted

3. **API Key Issues**:
   - Ensure `ANTHROPIC_API_KEY` is set in `.env`
   - Verify the API key is valid and has credits

4. **Port Already in Use**:
   - Kill existing processes: `pkill -f "python chatbot.py"`
   - Or change the port in `chatbot.py`

### Performance Tips

- **Large PDFs**: Consider splitting very large documents
- **Many Documents**: Monitor memory usage with large document collections
- **Response Speed**: Claude-3-Haiku provides the best speed/cost balance

## Development

### Adding Features

- **New LLM Models**: Modify the `ChatAnthropic` initialization in `chatbot.py`
- **Different Embeddings**: Change the `HuggingFaceEmbeddings` model
- **Custom Chunking**: Adjust `RecursiveCharacterTextSplitter` parameters
- **UI Improvements**: Modify the Gradio interface in `chatbot.py`

### Debugging

Enable debug output by uncommenting the print statements in `chatbot.py`:
```python
#print(f"Input: {message}. History: {history}\n")
```

## License

This project is open source. Please ensure you comply with Anthropic's usage policies when using their API.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section above
- Review the error messages in the terminal
- Ensure all dependencies are properly installed
