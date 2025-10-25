# RAG Chatbot with Anthropic Claude

A Retrieval-Augmented Generation (RAG) chatbot that allows you to chat with your PDF documents using Anthropic's Claude language model. The system processes PDF files, creates vector embeddings, and provides intelligent responses based on document content.

## Features

- **Multi-PDF Support**: Process and chat with multiple PDF documents simultaneously
- **Anthropic Claude Integration**: Uses Claude-3-Haiku for fast and cost-effective responses
- **Vector Search**: Leverages ChromaDB for efficient document retrieval
- **Source Attribution**: Shows which document(s) information comes from
- **Web Interface**: Clean Gradio-based chat interface
- **Local Processing**: All processing happens locally (except LLM calls)

## How It Works: RAG Concepts Explained

This application implements **Retrieval-Augmented Generation (RAG)**, a powerful AI technique that combines information retrieval with text generation. Here's how it works:

### 1. **Document Processing Pipeline**

```
PDF Documents → Text Extraction → Chunking → Embeddings → Vector Database
```

- **Text Extraction**: PDFs are converted to plain text using PyPDF
- **Chunking**: Large documents are split into smaller, manageable pieces (1000 characters each)
- **Embeddings**: Each chunk is converted to a numerical vector using HuggingFace's sentence transformers
- **Storage**: Vectors are stored in ChromaDB for fast similarity search

### 2. **Query Processing Flow**

```
User Question → Embedding → Similarity Search → Retrieved Chunks → LLM → Response
```

When you ask a question:

1. **Question Embedding**: Your question is converted to the same vector format as document chunks
2. **Similarity Search**: The system finds the most relevant document chunks using cosine similarity
3. **Context Assembly**: Retrieved chunks are combined with your question
4. **LLM Generation**: Claude processes the question + context to generate an accurate response

### 3. **Why RAG Works Better Than Plain LLMs**

- **Accuracy**: Responses are grounded in actual document content, not just training data
- **Up-to-date Information**: Can work with recent documents not in the LLM's training data
- **Source Attribution**: You know exactly which documents the information comes from
- **Cost Effective**: Only sends relevant context to the LLM, reducing API costs
- **Domain Specific**: Works with specialized documents (technical papers, manuals, etc.)

### 4. **Key Components Explained**

#### **Embeddings (Vector Representations)**
- Convert text into high-dimensional numerical vectors
- Similar content has similar vectors
- Enables mathematical similarity comparison
- Uses `sentence-transformers/all-MiniLM-L6-v2` model (384 dimensions)

#### **Vector Database (ChromaDB)**
- Stores embeddings with metadata (filename, chunk ID)
- Performs fast similarity search using approximate nearest neighbor algorithms
- Persists data locally for reuse
- Handles thousands of document chunks efficiently

#### **Text Chunking Strategy**
- **Chunk Size**: 1000 characters (balances context vs. precision)
- **Overlap**: 200 characters (prevents information loss at boundaries)
- **Recursive Splitting**: Respects natural text boundaries (paragraphs, sentences)

#### **Retrieval Strategy**
- **Top-K Retrieval**: Gets top 10 most similar chunks
- **Similarity Threshold**: Can filter out irrelevant results
- **Metadata Filtering**: Can search within specific documents

### 5. **The Complete RAG Workflow**

```
1. Document Ingestion:
   PDF → Text → Chunks → Embeddings → Vector DB

2. Query Processing:
   Question → Embedding → Search → Retrieve → Context + Question → LLM → Answer

3. Response Generation:
   Retrieved chunks + User question → Claude → Natural language response
```

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
1. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
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
python3 ingest_database.py
```

This will:
- Load all PDF files from the `data/` folder
- Split them into chunks (1000 characters with 200 overlap)
- Create embeddings using HuggingFace transformers
- Store everything in ChromaDB

**Note**: If you're using a virtual environment, make sure it's activated before running commands.

### 2. Start the Chatbot

Run the chatbot using the provided script:

```bash
./run_chatbot.sh
```

Or directly with Python:

```bash
python3 chatbot.py
```

**Alternative**: If you prefer to run without the shell script:

```bash
# Make sure your virtual environment is activated
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the chatbot
python3 chatbot.py
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

## Technical Implementation Details

### **Code Architecture**

The application is built with a modular design:

- **`ingest_database.py`**: Handles document processing and database population
- **`chatbot.py`**: Contains the RAG query processing and chat interface
- **`run_chatbot.sh`**: Convenience script for easy startup

### **Data Flow Implementation**

```python
# Document Processing (ingest_database.py)
loader = PyPDFDirectoryLoader(DATA_PATH)           # Load PDFs
text_splitter = RecursiveCharacterTextSplitter()  # Split into chunks
embeddings_model = HuggingFaceEmbeddings()        # Create embeddings
vector_store = Chroma()                           # Store in vector DB

# Query Processing (chatbot.py)
docs = retriever.invoke(message)                  # Retrieve similar chunks
knowledge = assemble_context(docs)                # Build context
response = llm.stream(rag_prompt)                 # Generate response
```

### **Key Algorithms Used**

1. **Text Chunking**: RecursiveCharacterTextSplitter
   - Respects sentence and paragraph boundaries
   - Maintains overlap to prevent information loss
   - Configurable chunk size and overlap

2. **Embedding Generation**: Sentence Transformers
   - Model: `all-MiniLM-L6-v2` (384-dimensional vectors)
   - Pre-trained on large text corpora
   - Optimized for semantic similarity

3. **Similarity Search**: ChromaDB's approximate nearest neighbor
   - Uses cosine similarity for vector comparison
   - Efficient indexing for fast retrieval
   - Supports metadata filtering

4. **Response Generation**: Anthropic Claude
   - Temperature-controlled generation
   - Context-aware prompting
   - Streaming responses for better UX

## Benefits and Use Cases

### **Why Use RAG Instead of Plain LLMs?**

| Traditional LLM | RAG System |
|----------------|------------|
| Limited to training data | Works with any documents |
| May hallucinate facts | Grounded in actual content |
| No source attribution | Shows which documents were used |
| Expensive for large contexts | Only sends relevant context |
| Static knowledge cutoff | Always up-to-date with your docs |

### **Perfect Use Cases**

- **📚 Academic Research**: Chat with research papers and technical documents
- **📋 Documentation**: Query user manuals, API docs, and technical specifications
- **📊 Business Intelligence**: Analyze reports, contracts, and business documents
- **🎓 Educational**: Create interactive learning experiences with textbooks
- **⚖️ Legal**: Search through case law, contracts, and legal documents
- **🔬 Scientific**: Process scientific papers and research findings

### **Performance Characteristics**

- **Speed**: Sub-second retrieval for most queries
- **Accuracy**: High precision due to document grounding
- **Scalability**: Handles thousands of document chunks efficiently
- **Cost**: Only pays for LLM tokens for relevant context
- **Privacy**: Documents stay local (only queries sent to API)

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
├── requirements.txt          # Python dependencies
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

2. **"No module named 'langchain_huggingface'"**:
   ```bash
   pip install langchain-huggingface
   ```

3. **"No relevant docs were retrieved"**:
   - Check if PDFs are in the `data/` folder
   - Re-run `ingest_database.py`
   - Verify PDFs are not corrupted

4. **API Key Issues**:
   - Ensure `ANTHROPIC_API_KEY` is set in `.env`
   - Verify the API key is valid and has credits

5. **Port Already in Use**:
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
