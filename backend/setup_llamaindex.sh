#!/bin/bash

echo "🧹 Cleaning up existing installations..."
pip uninstall -y langchain langchain-community langchain-core langchain-text-splitters langchain-openai chromadb

echo "📦 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "✅ Installation complete! LlamaIndex should now work without conflicts."
echo "📚 This setup uses SimpleVectorStore (in-memory with persistence)"
echo "🚀 Run 'python main.py' to start the server."

echo ""
echo "🧪 Testing LlamaIndex installation..."
python -c "
try:
    from llama_index.core import VectorStoreIndex
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.core.vector_stores import SimpleVectorStore
    print('✅ LlamaIndex installation test passed!')
except ImportError as e:
    print(f'❌ Installation test failed: {e}')
    exit(1)
except Exception as e:
    print(f'❌ Error during test: {e}')
    exit(1)
"

echo ""
echo "📖 Next steps:"
echo "1. Set your OpenAI API key: export OPENAI_API_KEY=your_key_here"
echo "2. Start the server: python main.py"
echo "3. Upload documents via /upload endpoint or place them in ./documents/"
echo "4. Use the chat interface to index and query documents" 