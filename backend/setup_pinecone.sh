#!/bin/bash

# Setup script for Pinecone + LlamaIndex integration
echo "🔧 Setting up Pinecone integration with LlamaIndex..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file template..."
    cat > .env << 'EOF'
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=llamaindex-documents

# Optional: Pinecone namespace for organizing documents
PINECONE_NAMESPACE=default
EOF
    echo "✅ .env file created! Please edit it with your API keys."
else
    echo "✅ .env file already exists."
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit the .env file with your API keys:"
echo "   - Get your OpenAI API key from: https://platform.openai.com/api-keys"
echo "   - Get your Pinecone API key from: https://app.pinecone.io/"
echo "   - Set your preferred Pinecone environment (e.g., us-east-1-aws)"
echo ""
echo "2. Run the application:"
echo "   python main.py"
echo ""
echo "3. The application will automatically:"
echo "   - Create a Pinecone index if it doesn't exist"
echo "   - Use Pinecone for all vector storage (required)"
echo "   - Fail if Pinecone is not properly configured"
echo ""
echo "🔍 Available Pinecone environments:"
echo "   - us-east-1-aws (recommended)"
echo "   - us-west-2-aws"
echo "   - eu-west-1-aws"
echo "   - asia-southeast-1-aws"
echo "   - Check Pinecone console for latest regions" 