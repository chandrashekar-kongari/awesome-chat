# OpenAI Agents Chat API with LlamaIndex + Pinecone Integration

This is an OpenAI Agents-powered chat API with advanced document processing capabilities using LlamaIndex for RAG (Retrieval-Augmented Generation) and Pinecone for scalable vector storage.

## Features

### Core Capabilities

- **OpenAI Agents**: Streaming chat interface with ReAct pattern
- **Lifecycle Hooks**: Real-time agent execution monitoring
- **Function Tools**: Multiple utility tools for various tasks

### LlamaIndex + Pinecone Integration

- **Document Indexing**: Support for PDF, TXT, and other document formats
- **Vector Search**: Semantic search using OpenAI embeddings
- **RAG Pipeline**: Retrieval-Augmented Generation for document Q&A
- **Collection Management**: Organize documents into collections
- **Pinecone Storage**: Cloud-based vector storage with Pinecone
- **Scalable**: Handle millions of documents with sub-second search

## Installation

### Quick Setup (Recommended)

1. **Run the setup script**:

   ```bash
   chmod +x setup_pinecone.sh
   bash setup_pinecone.sh
   ```

2. **Set environment variables** (edit the generated `.env` file):

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_ENVIRONMENT=us-east-1-aws
   PINECONE_INDEX_NAME=llamaindex-documents
   PINECONE_NAMESPACE=default
   ```

3. **Get your API keys**:

   - **OpenAI**: https://platform.openai.com/api-keys
   - **Pinecone**: https://app.pinecone.io/

4. **Run the server**:
   ```bash
   python main.py
   ```

### Manual Installation

1. **Clean up conflicting packages**:

   ```bash
   pip uninstall -y langchain langchain-community langchain-core langchain-text-splitters langchain-openai chromadb
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables and run**:
   ```bash
   export OPENAI_API_KEY=your_openai_api_key_here
   python main.py
   ```

## Important Notes

### Architecture Benefits

This implementation uses **Pinecone** for enterprise-grade vector storage:

- **Pinecone Vector Store**: Cloud-based, highly scalable vector database
- **Namespace-based organization**: Collections organized by namespaces
- **Real-time indexing**: Immediate availability of indexed documents
- **High performance**: Sub-second similarity search at scale

### Storage Architecture

- Documents: `./documents/` directory (for uploads)
- Vector Indexes: **Pinecone Cloud** (organized by namespaces)
- Collections: `{namespace}_{collection_name}` format
- Embeddings: OpenAI text-embedding-3-small (1536 dimensions)

## Available Tools

### Basic Tools

- `how_many_jokes()` - Returns random number of jokes
- `get_random_fact()` - Returns interesting facts
- `calculate_math(expression)` - Evaluates math expressions
- `get_weather_info(city)` - Mock weather information
- `observe_result(observation)` - Records observations with AI analysis
- `reflect_on_progress(reflection)` - Strategic reflection with recommendations

### Document & RAG Tools (LlamaIndex)

- `index_document(file_path, collection_name)` - Index documents for search
- `query_documents(query, collection_name)` - Natural language document search
- `list_document_collections()` - List all document collections
- `delete_document_collection(collection_name)` - Remove document collections
- `create_document_summary(file_path, collection_name)` - Generate document summaries
- `semantic_search_documents(query, collection_name, top_k)` - Advanced semantic search

## Usage Examples

### Document Processing Workflow

1. **Upload a document** (via `/upload` endpoint or place in `./documents/` folder)

2. **Index the document**:

   ```
   User: "Index the document ./documents/my_document.pdf"
   ```

3. **Query the document**:

   ```
   User: "What are the main topics in this document?"
   User: "Find information about revenue growth"
   User: "Summarize the key findings"
   ```

4. **Advanced searches**:
   ```
   User: "Search for passages about machine learning with top 3 results"
   User: "Create a comprehensive summary of the document"
   ```

### Collection Management

```
User: "List all document collections"
User: "Delete the collection named 'old_documents'"
```

### ReAct Pattern Example

The agent follows a systematic ReAct pattern:

```
THOUGHT: User wants to analyze a document about financial reports
PLAN: 1) Index the document, 2) Create summary, 3) Answer specific questions
ACT: [Calls index_document("./documents/financial_report.pdf", "finance")]
OBSERVE: [Calls observe_result("Document indexed successfully with 150 chunks")]
REFLECT: [Calls reflect_on_progress("Document ready for analysis")]
DECISION: Create summary and ask for specific questions
```

## API Endpoints

### Chat Interface

- `POST /chat/stream` - Streaming chat with agent
- `OPTIONS /chat/stream` - CORS preflight

### File Management

- `POST /upload` - Upload documents for processing
- `GET /health` - Health check

### Usage with curl

**Upload a document**:

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@/path/to/document.pdf"
```

**Chat with the agent**:

```bash
curl -X POST "http://localhost:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"message": "Index the document ./documents/document.pdf"}'
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY` - Required for OpenAI API access
- `PINECONE_API_KEY` - Required for Pinecone vector storage
- `PINECONE_ENVIRONMENT` - Pinecone environment (e.g., us-east-1-aws)
- `PINECONE_INDEX_NAME` - Pinecone index name (default: llamaindex-documents)
- `PINECONE_NAMESPACE` - Pinecone namespace prefix (default: default)

### Storage

- Documents: `./documents/` directory (for uploads)
- Vector Indexes: **Pinecone Cloud** (organized by namespaces)
- Index: Auto-created if it doesn't exist
- Dimensions: 1536 (OpenAI text-embedding-3-small)

## Architecture

### Document Processing Pipeline

1. **Document Loading**: PyMuPDF for PDFs, SimpleDirectoryReader for text
2. **Text Splitting**: Sentence-based chunking for optimal retrieval
3. **Embeddings**: OpenAI text-embedding-3-small model (1536 dimensions)
4. **Vector Storage**: Pinecone cloud vector database
5. **Retrieval**: Semantic search with similarity scoring
6. **Namespace Organization**: Collections organized by Pinecone namespaces

### Agent Architecture

- **ReAct Pattern**: Systematic reasoning, planning, acting, observing, reflecting
- **Tool Integration**: Seamless integration of LlamaIndex tools
- **Streaming**: Real-time response streaming
- **Lifecycle Hooks**: Monitoring and debugging capabilities

## Troubleshooting

### Common Issues

1. **Langchain conflicts**: Run the setup script:

   ```bash
   bash setup_llamaindex.sh
   ```

2. **Import errors**: Ensure clean installation:

   ```bash
   pip uninstall -y langchain langchain-community langchain-core chromadb
   pip install -r requirements.txt
   ```

3. **File upload fails**: Ensure `python-multipart` is installed:

   ```bash
   pip install python-multipart
   ```

4. **Permission errors**: Make setup script executable:
   ```bash
   chmod +x setup_llamaindex.sh
   ```

### Performance Tips

- Use descriptive collection names for better organization
- Limit document size for better performance (< 50MB recommended)
- Use semantic search for precise queries
- Regular cleanup of unused collections
- Collections are automatically persisted and reloaded

### Why SimpleVectorStore?

- **No external dependencies**: Avoids ChromaDB/langchain conflicts
- **File persistence**: Automatically saves/loads indexes
- **Fast setup**: No database configuration required
- **Suitable for development**: Good for testing and small to medium datasets

## Development

### Adding New Tools

1. Create function with `@function_tool` decorator
2. Add to appropriate tools list
3. Update agent instructions
4. Test with the chat interface

### Extending Document Support

1. Add new reader in `DocumentManager.index_document()`
2. Handle new file types in conditional logic
3. Update documentation

### Scaling Considerations

For production use with large datasets, consider:

- Migrating to a dedicated vector database (Weaviate, Qdrant)
- Implementing proper authentication
- Adding rate limiting
- Using async document processing

## License

This project demonstrates integration between OpenAI Agents and LlamaIndex for advanced document processing and RAG capabilities without dependency conflicts.
