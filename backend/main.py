import asyncio
import random
import os
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agents import Agent, ItemHelpers, Runner, function_tool
from openai import AsyncOpenAI
try:
    # Try importing lifecycle hooks - these might be in different locations depending on SDK version
    from agents import RunHooks
    LIFECYCLE_AVAILABLE = True
    print("✅ Lifecycle hooks imported successfully")
except ImportError:
    try:
        from agents.lifecycle import RunHooks
        LIFECYCLE_AVAILABLE = True
        print("✅ Lifecycle hooks imported from agents.lifecycle")
    except ImportError:
        print("⚠️  Lifecycle hooks not available in this SDK version")
        LIFECYCLE_AVAILABLE = False

import json
from dotenv import load_dotenv
from typing import Any
from pathlib import Path

# LlamaIndex imports
try:
    # Import core components first
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.schema import Document, TextNode
    from llama_index.readers.file import PyMuPDFReader
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.postprocessor import SimilarityPostprocessor
    from llama_index.core.settings import Settings
    from llama_index.core import QueryBundle
    
    # Pinecone imports
    from pinecone import Pinecone, ServerlessSpec
    from llama_index.vector_stores.pinecone import PineconeVectorStore
    
    LLAMAINDEX_AVAILABLE = True
    PINECONE_AVAILABLE = True
    print("✅ LlamaIndex dependencies imported successfully (with Pinecone support)")
except ImportError as e:
    print(f"⚠️  LlamaIndex not available: {e}")
    print("To enable LlamaIndex functionality, run: bash setup_llamaindex.sh")
    LLAMAINDEX_AVAILABLE = False
    PINECONE_AVAILABLE = False
except Exception as e:
    print(f"❌ Error importing LlamaIndex: {e}")
    print("This might be due to version conflicts. Try running: bash setup_llamaindex.sh")
    LLAMAINDEX_AVAILABLE = False
    PINECONE_AVAILABLE = False

# Load environment variables
load_dotenv()

# Check if OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY environment variable not set!")
    print("Please set it with: export OPENAI_API_KEY=your_key_here")
    print("Or create a .env file with: OPENAI_API_KEY=your_key_here")

# Initialize OpenAI client for observation and reflection tools
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize LlamaIndex if available
document_manager = None
if LLAMAINDEX_AVAILABLE:
    try:
        # Configure LlamaIndex settings
        Settings.llm = LlamaIndexOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        class DocumentManager:
            """Manages document indexing and retrieval using LlamaIndex with Pinecone."""
            
            def __init__(self):
                self.indexes = {}
                self.query_engines = {}
                self.documents_dir = Path("./documents")
                self.documents_dir.mkdir(exist_ok=True)
                
                # Initialize Pinecone client
                self.pinecone_client = None
                self.pinecone_index = None
                
                # Initialize Pinecone (required)
                if not PINECONE_AVAILABLE:
                    raise ImportError("Pinecone is required but not available. Please install: pip install pinecone-client llama-index-vector-stores-pinecone")
                
                pinecone_api_key = os.getenv("PINECONE_API_KEY")
                if not pinecone_api_key:
                    raise ValueError("PINECONE_API_KEY environment variable is required")
                
                self.pinecone_client = Pinecone(api_key=pinecone_api_key)
                self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "llamaindex-documents")
                self.pinecone_namespace = os.getenv("PINECONE_NAMESPACE", "default")
                self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
                
                # Create index if it doesn't exist
                self._ensure_pinecone_index()
                
                print("✅ DocumentManager initialized successfully with Pinecone")
            
            def _ensure_pinecone_index(self):
                """Ensure the Pinecone index exists, create if it doesn't."""
                try:
                    # Check if index exists
                    existing_indexes = self.pinecone_client.list_indexes()
                    index_names = [index.name for index in existing_indexes]
                    
                    if self.pinecone_index_name not in index_names:
                        # Parse region from environment string (e.g., "us-east-1-aws" -> "us-east-1")
                        region_parts = self.pinecone_environment.split('-')
                        if len(region_parts) >= 3:
                            # Handle format like "us-east-1-aws"
                            region = '-'.join(region_parts[:-1])  # Remove the last part (cloud provider)
                        else:
                            # Fallback to the full environment string
                            region = self.pinecone_environment
                        
                        print(f"🔧 Creating Pinecone index in region: {region}")
                        
                        # Create new index
                        self.pinecone_client.create_index(
                            name=self.pinecone_index_name,
                            dimension=1536,  # OpenAI embedding dimension
                            metric='cosine',
                            spec=ServerlessSpec(
                                cloud='aws',
                                region=region
                            )
                        )
                        print(f"✅ Created Pinecone index: {self.pinecone_index_name}")
                    else:
                        print(f"✅ Using existing Pinecone index: {self.pinecone_index_name}")
                    
                    # Connect to the index
                    self.pinecone_index = self.pinecone_client.Index(self.pinecone_index_name)
                    
                except Exception as e:
                    print(f"❌ Error setting up Pinecone index: {e}")
                    raise
            
            async def index_document(self, file_path: str, collection_name: str = "default") -> str:
                """Index a document and return status."""
                try:
                    file_path = Path(file_path)
                    if not file_path.exists():
                        return f"❌ File not found: {file_path}"
                    
                    # Choose appropriate reader based on file type
                    if file_path.suffix.lower() == '.pdf':
                        reader = PyMuPDFReader()
                        documents = reader.load_data(file_path)
                    else:
                        reader = SimpleDirectoryReader(input_files=[str(file_path)])
                        documents = reader.load_data()
                    
                    # Create Pinecone vector store
                    vector_store = PineconeVectorStore(
                        pinecone_index=self.pinecone_index,
                        namespace=f"{self.pinecone_namespace}_{collection_name}"
                    )
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    
                    # Create index
                    index = VectorStoreIndex.from_documents(
                        documents, 
                        storage_context=storage_context,
                        show_progress=True
                    )
                    
                    # Store index and create query engine
                    self.indexes[collection_name] = index
                    self.query_engines[collection_name] = index.as_query_engine(
                        similarity_top_k=3,
                        streaming=True
                    )
                    
                    return f"✅ Successfully indexed {file_path.name} in collection '{collection_name}' (Pinecone)"
                    
                except Exception as e:
                    return f"❌ Error indexing document: {str(e)}"
            
            async def query_documents(self, query: str, collection_name: str = "default") -> str:
                """Query documents in a collection."""
                try:
                    if collection_name not in self.query_engines:
                        # Try to recreate the index from Pinecone if it doesn't exist in memory
                        try:
                            vector_store = PineconeVectorStore(
                                pinecone_index=self.pinecone_index,
                                namespace=f"{self.pinecone_namespace}_{collection_name}"
                            )
                            storage_context = StorageContext.from_defaults(vector_store=vector_store)
                            index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
                            self.indexes[collection_name] = index
                            self.query_engines[collection_name] = index.as_query_engine(
                                similarity_top_k=3,
                                streaming=True
                            )
                        except Exception as e:
                            return f"❌ Error connecting to Pinecone collection '{collection_name}': {str(e)}"
                    
                    query_engine = self.query_engines[collection_name]
                    response = query_engine.query(query)
                    
                    # Format response with sources
                    result = f"📄 Query Result (Pinecone):\n\n{response}\n\n"
                    
                    if hasattr(response, 'source_nodes') and response.source_nodes:
                        result += "📚 Sources:\n"
                        for i, node in enumerate(response.source_nodes[:2], 1):
                            score = getattr(node, 'score', 'N/A')
                            result += f"  {i}. Score: {score:.3f}\n"
                            if hasattr(node, 'node') and hasattr(node.node, 'metadata'):
                                metadata = node.node.metadata
                                if 'file_name' in metadata:
                                    result += f"     Source: {metadata['file_name']}\n"
                                if 'page_label' in metadata:
                                    result += f"     Page: {metadata['page_label']}\n"
                    
                    return result
                    
                except Exception as e:
                    return f"❌ Error querying documents: {str(e)}"
            
            async def list_collections(self) -> str:
                """List all available document collections."""
                try:
                    # Check in-memory collections
                    collections = set(self.indexes.keys())
                    
                    if collections:
                        result = f"📚 Available Collections (Pinecone):\n"
                        for collection in collections:
                            if collection in self.indexes:
                                # In-memory collection
                                count = len(self.indexes[collection].docstore.docs)
                                result += f"  • {collection}: {count} documents (loaded)\n"
                            else:
                                result += f"  • {collection}: (in Pinecone)\n"
                        return result
                    else:
                        return f"📚 No document collections found (Pinecone). Use index_document() to create one."
                except Exception as e:
                    return f"❌ Error listing collections: {str(e)}"
            
            async def delete_collection(self, collection_name: str) -> str:
                """Delete a document collection."""
                try:
                    if collection_name in self.indexes:
                        del self.indexes[collection_name]
                    if collection_name in self.query_engines:
                        del self.query_engines[collection_name]
                    
                    # For Pinecone, delete the namespace
                    try:
                        namespace = f"{self.pinecone_namespace}_{collection_name}"
                        # Note: Pinecone doesn't have a direct delete namespace method
                        # We would need to delete all vectors in the namespace
                        # This is a more complex operation that might require listing and deleting vectors
                        print(f"⚠️  Pinecone namespace '{namespace}' vectors should be manually deleted")
                    except Exception as e:
                        print(f"⚠️  Could not delete Pinecone namespace: {e}")
                    
                    return f"✅ Successfully deleted collection '{collection_name}' from memory (Pinecone)"
                except Exception as e:
                    return f"❌ Error deleting collection: {str(e)}"
        
        document_manager = DocumentManager()
        print("📚 LlamaIndex DocumentManager initialized successfully")
        
    except Exception as e:
        print(f"❌ Error initializing LlamaIndex: {e}")
        LLAMAINDEX_AVAILABLE = False
        document_manager = None

app = FastAPI(title="OpenAI Agents Chat API")

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str

# Lifecycle hooks implementation based on official SDK documentation
if LIFECYCLE_AVAILABLE:
    class StreamingHooks(RunHooks):
        """Custom lifecycle hooks for streaming events to frontend based on OpenAI Agents SDK."""
        
        def __init__(self):
            super().__init__()
            self.event_queue = []
            self.stream_callback = None
        
        def set_stream_callback(self, callback):
            """Set callback function to immediately stream events."""
            self.stream_callback = callback
        
        async def _stream_event_immediately(self, event_type, event_data):
            """Stream event immediately if callback is set, otherwise queue it."""
            if self.stream_callback:
                try:
                    await self.stream_callback(event_type, event_data)
                except Exception as e:
                    print(f"Error streaming lifecycle event: {e}")
                    # Fallback to queuing if streaming fails
                    self.event_queue.append((event_type, event_data))
            else:
                # Queue if no callback available
                self.event_queue.append((event_type, event_data))
        
        async def on_agent_start(self, context, agent) -> None:
            """Called before the agent is invoked. Called each time the current agent changes."""
            event_data = {
                'agent_name': getattr(agent, 'name', 'Unknown Agent'),
                'agent_id': str(id(agent)),  # Use object id as identifier
                'context': 'agent_start',
                'timestamp': asyncio.get_event_loop().time()
            }
            print(f"🚀 LIFECYCLE: Agent starting - {event_data}")
            await self._stream_event_immediately('agent_lifecycle', event_data)
        
        async def on_agent_end(self, context, agent, output: Any) -> None:
            """Called when the agent produces a final output."""
            event_data = {
                'agent_name': getattr(agent, 'name', 'Unknown Agent'),
                'agent_id': str(id(agent)),
                'context': 'agent_end',
                'output': str(output) if output else None,
                'timestamp': asyncio.get_event_loop().time()
            }
            print(f"✅ LIFECYCLE: Agent ending - {event_data}")
            await self._stream_event_immediately('agent_lifecycle', event_data)
        
        async def on_handoff(self, context, from_agent, to_agent) -> None:
            """Called when a handoff occurs."""
            event_data = {
                'from_agent': getattr(from_agent, 'name', 'Unknown Agent'),
                'to_agent': getattr(to_agent, 'name', 'Unknown Agent'),
                'context': 'handoff',
                'timestamp': asyncio.get_event_loop().time()
            }
            print(f"🔄 LIFECYCLE: Handoff - {event_data}")
            await self._stream_event_immediately('handoff_lifecycle', event_data)
        
        async def on_tool_start(self, context, agent, tool) -> None:
            """Called before a tool is invoked."""
            event_data = {
                'tool_name': getattr(tool, 'name', getattr(tool, '__name__', 'Unknown Tool')),
                'agent_name': getattr(agent, 'name', 'Unknown Agent'),
                'context': 'tool_start',
                'timestamp': asyncio.get_event_loop().time()
            }
            print(f"⚡ LIFECYCLE: Tool starting - {event_data}")
            await self._stream_event_immediately('tool_lifecycle', event_data)
        
        async def on_tool_end(self, context, agent, tool, result: str) -> None:
            """Called after a tool is invoked."""
            event_data = {
                'tool_name': getattr(tool, 'name', getattr(tool, '__name__', 'Unknown Tool')),
                'agent_name': getattr(agent, 'name', 'Unknown Agent'),
                'context': 'tool_end',
                'result': result,
                'timestamp': asyncio.get_event_loop().time()
            }
            print(f"✅ LIFECYCLE: Tool ending - {event_data}")
            await self._stream_event_immediately('tool_lifecycle', event_data)
        
        def get_and_clear_events(self):
            """Get all queued events and clear the queue."""
            events = self.event_queue.copy()
            self.event_queue.clear()
            return events
else:
    print("⚠️  Running without lifecycle hooks - upgrade to newer SDK version for full functionality")

@function_tool
async def how_many_jokes() -> int:
    """Returns a random number of jokes to tell (1-10)."""
    await asyncio.sleep(4)  # Wait for 5 seconds
    return random.randint(1, 10)

@function_tool
async def get_random_fact() -> str:
    """Returns a random interesting fact."""
    await asyncio.sleep(4)  # Wait for 5 seconds
    facts = [
        "Octopuses have three hearts and blue blood.",
        "Bananas are berries, but strawberries aren't.",
        "A group of flamingos is called a 'flamboyance'.",
        "Honey never spoils - archaeologists have found edible honey in ancient Egyptian tombs.",
        "The shortest war in history lasted only 38-45 minutes.",
        "Wombat poop is cube-shaped.",
        "There are more possible games of chess than atoms in the observable universe.",
        "Sharks are older than trees.",
        "The unicorn is Scotland's national animal.",
        "A cloud can weigh more than a million pounds."
    ]
    return random.choice(facts)

@function_tool
async def calculate_math(expression: str) -> str:
    """Safely evaluates basic math expressions like '2+2' or '10*5'."""
    await asyncio.sleep(4)  # Wait for 5 seconds
    try:
        # Basic safety check - only allow numbers, operators, and parentheses
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

@function_tool
def get_weather_info(city: str) -> str:
    """Returns mock weather information for a given city."""
    weather_conditions = ["sunny", "cloudy", "rainy", "snowy", "partly cloudy"]
    temperature = random.randint(-10, 35)
    condition = random.choice(weather_conditions)
    
    return f"Weather in {city}: {temperature}°C, {condition}"

@function_tool
async def observe_result(observation: str) -> str:
    """Record an observation about the current state, results, or findings. Use this tool to document what you observed after taking an action or analyzing information."""
    try:
        # Use OpenAI to generate a more intelligent observation summary
        response = await openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": "You are an analytical observer. Given a raw observation, create a clear, concise, and insightful observation summary. Focus on what was observed, what it means, and any patterns or anomalies. Keep it brief but informative."
                },
                {
                    "role": "user",
                    "content": f"Raw observation: {observation}"
                }
            ],
            max_tokens=25,
            temperature=0.1
        )
        
        observation_summary = response.choices[0].message.content.strip()
        print(f"🔍 OBSERVATION GENERATED: {observation_summary}")
        return observation_summary
        
    except Exception as e:
        print(f"Error generating observation: {e}")
        return f"Observation recorded: {observation}"

@function_tool
async def reflect_on_progress(reflection: str) -> str:
    """Record a reflection about the progress, success, or next steps. Use this tool to document your thinking about whether the current approach is working and what to do next."""
    try:
        # Use OpenAI to generate a more intelligent reflection
        response = await openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": "You are a strategic thinker. Given a reflection prompt, analyze the current progress and provide thoughtful insights about: 1) What's working well, 2) What could be improved, 3) Next steps or recommendations. Be concise but thorough."
                },
                {
                    "role": "user",
                    "content": f"Reflection prompt: {reflection}"
                }
            ],
            max_tokens=25,
            temperature=0.1
        )
        
        reflection_summary = response.choices[0].message.content.strip()
        print(f"🤔 REFLECTION GENERATED: {reflection_summary}")
        return reflection_summary
        
    except Exception as e:
        print(f"Error generating reflection: {e}")
        return f"Reflection recorded: {reflection}"

# LlamaIndex function tools
if LLAMAINDEX_AVAILABLE and document_manager:
    @function_tool
    async def index_document(file_path: str, collection_name: str = "default") -> str:
        """Index a document (PDF, TXT, etc.) for retrieval and search. Provide the file path and optionally a collection name to organize documents."""
        await asyncio.sleep(2)  # Simulate processing time
        return await document_manager.index_document(file_path, collection_name)
    
    @function_tool
    async def query_documents(query: str, collection_name: str = "default") -> str:
        """Search and retrieve information from indexed documents. Provide a natural language query and optionally specify which collection to search."""
        await asyncio.sleep(2)  # Simulate processing time
        return await document_manager.query_documents(query, collection_name)
    
    @function_tool
    async def list_document_collections() -> str:
        """List all available document collections and their document counts."""
        await asyncio.sleep(1)  # Simulate processing time
        return await document_manager.list_collections()
    
    @function_tool
    async def delete_document_collection(collection_name: str) -> str:
        """Delete a document collection and all its indexed documents. Use with caution!"""
        await asyncio.sleep(1)  # Simulate processing time
        return await document_manager.delete_collection(collection_name)
    
    @function_tool
    async def create_document_summary(file_path: str, collection_name: str = "summary") -> str:
        """Create a comprehensive summary of a document by indexing it and generating an overview."""
        try:
            await asyncio.sleep(1)  # Simulate processing time
            
            # First index the document
            index_result = await document_manager.index_document(file_path, collection_name)
            if "❌" in index_result:
                return index_result
            
            # Then query for a summary
            summary_query = "Provide a comprehensive summary of this document including key points, main topics, and important details."
            summary_result = await document_manager.query_documents(summary_query, collection_name)
            
            return f"📄 Document Summary for {file_path}:\n\n{summary_result}"
            
        except Exception as e:
            return f"❌ Error creating document summary: {str(e)}"
    
    @function_tool
    async def semantic_search_documents(query: str, collection_name: str = "default", top_k: int = 5) -> str:
        """Perform semantic search across documents to find the most relevant passages for a query."""
        try:
            await asyncio.sleep(2)  # Simulate processing time
            
            if collection_name not in document_manager.indexes:
                # Try to load the collection from Pinecone first
                try:
                    vector_store = PineconeVectorStore(
                        pinecone_index=document_manager.pinecone_index,
                        namespace=f"{document_manager.pinecone_namespace}_{collection_name}"
                    )
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
                    document_manager.indexes[collection_name] = index
                except Exception as e:
                    return f"❌ Error loading Pinecone collection '{collection_name}': {str(e)}"
            
            # Get the index and create a custom retriever
            index = document_manager.indexes[collection_name]
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=top_k
            )
            
            # Perform retrieval
            query_bundle = QueryBundle(query)
            nodes = retriever.retrieve(query_bundle)
            
            # Format results
            result = f"🔍 Semantic Search Results (Pinecone) for: '{query}'\n\n"
            
            for i, node in enumerate(nodes, 1):
                score = getattr(node, 'score', 'N/A')
                result += f"📄 Result {i} (Score: {score:.3f}):\n"
                result += f"{node.node.text[:300]}{'...' if len(node.node.text) > 300 else ''}\n\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error performing semantic search: {str(e)}"
    
    # List of available tools when LlamaIndex is available
    llamaindex_tools = [
        index_document,
        query_documents,
        list_document_collections,
        delete_document_collection,
        create_document_summary,
        semantic_search_documents
    ]
    
    print("📚 LlamaIndex function tools created successfully")
else:
    llamaindex_tools = []
    print("📚 LlamaIndex tools not available - install dependencies to enable")

async def stream_agent_response(message: str) -> AsyncGenerator[str, None]:
    """Stream the agent's response with improved error handling and connection management."""
    agent = None
    streaming_hooks = None
    lifecycle_event_queue = None
    lifecycle_task = None
    
    try:
        # Send initial connection event
        yield f"event: connection\ndata: {json.dumps({'status': 'connected'})}\n\n"
        await asyncio.sleep(0.01)
        
        # Create agent with multiple tools using ReAct pattern
        # Combine base tools with LlamaIndex tools
        base_tools = [how_many_jokes, get_random_fact, calculate_math, get_weather_info, observe_result, reflect_on_progress]
        all_tools = base_tools + llamaindex_tools
        
        agent = Agent(
            name="Assistant",
            model="gpt-4o",
            
            instructions="""You are an intelligent assistant that follows an adaptive ReAct (Reason-and-Act) pattern for problem-solving. You approach every task systematically through step-by-step reasoning, planning, and action execution, with the ability to adapt and iterate based on observations and reflections.

            ## 🚨 CRITICAL RULE: ONE TOOL AT A TIME
            **YOU MUST ONLY CALL ONE TOOL PER TURN. NEVER CALL MULTIPLE TOOLS SIMULTANEOUSLY.**
            - Call one tool, observe the result, then decide on the next action
            - This ensures proper sequencing and prevents overwhelming the system
            - Always wait for a tool's result before calling another tool

            ## 🧠 Adaptive ReAct Framework - Follow This Process:

            ### **THOUGHT**: Start by reasoning about the user's request
            - What is the user asking for?
            - What information do I need to gather?
            - What steps are required to solve this?
            - Which tools might be helpful?

            ### **PLAN**: Create a step-by-step plan
            - Break down complex tasks into smaller steps
            - Identify which tools to use and in what order
            - Consider potential challenges or edge cases
            - Determine success criteria

            ### **ACT**: Execute your plan using available tools
            - Use tools deliberately based on your plan
            - Execute one action at a time
            - Pay attention to tool results
            - **REMEMBER: Only call ONE tool per turn**

            ### **OBSERVE**: Use the observe_result tool to record observations
            - **ALWAYS call observe_result() tool after receiving results from any action**
            - Document what the tool returned
            - Note whether the result was expected
            - Record if you have enough information to proceed
            - **PAY ATTENTION TO THE INTELLIGENT ANALYSIS PROVIDED**

            ### **REFLECT**: Use the reflect_on_progress tool to evaluate progress
            - **ALWAYS call reflect_on_progress() tool after making observations**
            - Evaluate if you're closer to solving the user's request
            - Determine if you need additional actions
            - Decide if you should modify your approach
            - **CAREFULLY CONSIDER THE RECOMMENDATIONS PROVIDED**

            ### **ADAPTIVE DECISION MAKING**: 
            **CRITICAL: After each OBSERVE and REFLECT cycle, you MUST:**
            1. **Analyze the insights** from both observe_result() and reflect_on_progress() tools
            2. **Check if the recommendations suggest additional actions**
            3. **Decide whether to:**
               - Continue with the original plan
               - Modify the approach based on insights
               - Take additional corrective actions
               - Gather more information
               - Try alternative methods
            4. **If recommendations suggest more actions, CONTINUE the ReAct cycle**
            5. **Only stop when reflections confirm the task is completely satisfied**

            ## 🛠️ Available Tools:

            ### Basic Tools:
            🎭 **how_many_jokes()** - Returns a random number (1-10) of jokes to tell
            🧠 **get_random_fact()** - Returns fascinating trivia and facts
            🧮 **calculate_math(expression)** - Safely evaluates mathematical expressions
            🌤️ **get_weather_info(city)** - Returns weather information for a city
            👁️ **observe_result(observation)** - Record observations about results or findings (provides intelligent analysis)
            🤔 **reflect_on_progress(reflection)** - Record reflections about progress and next steps (provides strategic recommendations)

            ### Document & RAG Tools (LlamaIndex):
            📁 **index_document(file_path, collection_name)** - Index a document (PDF, TXT, etc.) for retrieval and search
            🔍 **query_documents(query, collection_name)** - Search and retrieve information from indexed documents using natural language
            📚 **list_document_collections()** - List all available document collections and their document counts
            🗑️ **delete_document_collection(collection_name)** - Delete a document collection (use with caution!)
            📄 **create_document_summary(file_path, collection_name)** - Create a comprehensive summary of a document
            🎯 **semantic_search_documents(query, collection_name, top_k)** - Perform semantic search to find relevant document passages

            ## 📝 Response Format:

            **THOUGHT:** [Your reasoning about the request]
            **PLAN:** [Your step-by-step approach]
            **ACT:** [Execute tools as needed]
            **OBSERVE:** [Use observe_result() tool to record what happened]
            **REFLECT:** [Use reflect_on_progress() tool to evaluate progress]
            **DECISION:** [Based on observation and reflection insights, what should happen next?]

            Continue this cycle until both observation and reflection confirm the task is complete.

            ## 🎯 Examples:

            **User:** "What's 25 * 17 + 89?"
            **THOUGHT:** This is a math problem that needs calculation.
            **PLAN:** 1) Use calculate_math() to solve the expression, 2) Observe the result, 3) Reflect on whether it's correct
            **ACT:** [Call calculate_math("25 * 17 + 89")]
            **OBSERVE:** [Call observe_result("The calculation returned 514")]
            **REFLECT:** [Call reflect_on_progress("Math calculation completed")]
            **DECISION:** [If reflection suggests verification needed, continue. If confirmed complete, finish.]

            **User:** "Can you analyze this PDF document for me?"
            **THOUGHT:** The user wants me to analyze a document. I need to index it first, then query it.
            **PLAN:** 1) Index the document, 2) Create a summary, 3) Allow for follow-up questions
            **ACT:** [Call index_document("path/to/document.pdf", "user_document")]
            **OBSERVE:** [Call observe_result("Document was successfully indexed")]
            **REFLECT:** [Call reflect_on_progress("Document is ready for analysis")]
            **DECISION:** [Create summary and ask if user has specific questions]

            ## 🔄 Adaptive Iteration Rules:
            - **NEVER stop after just one action** - always observe and reflect first
            - **Act on recommendations** from observe_result() and reflect_on_progress()
            - **If insights suggest issues, corrections, or improvements - take action**
            - **Continue iterating until reflections confirm complete satisfaction**
            - **Use the intelligent analysis to guide your next steps**
            - **Be responsive to suggestions for alternative approaches**
            - **Only conclude when reflection explicitly confirms task completion**

            ## 🚨 Key Behavioral Rules:
            1. **Always observe and reflect after each action**
            2. **Pay attention to the AI-generated insights and recommendations**
            3. **Take additional actions if recommendations suggest them**
            4. **Don't stop until reflection confirms complete success**
            5. **Use insights to improve your approach dynamically**
            6. **For document tasks, use the appropriate RAG tools for indexing and querying**

            Remember: Think step-by-step, plan carefully, act deliberately, observe intelligently, reflect strategically, and adapt based on insights!""",
            tools=all_tools,
        )

        # Initialize lifecycle hooks if available
        if LIFECYCLE_AVAILABLE:
            streaming_hooks = StreamingHooks()
            
            # Create a queue for immediate lifecycle events
            lifecycle_event_queue = asyncio.Queue()
            
            # Create immediate streaming callback for lifecycle events
            async def stream_lifecycle_event(event_type, event_data):
                """Immediately queue lifecycle events for streaming."""
                print(f"🎣 QUEUEING LIFECYCLE IMMEDIATELY: {event_type} - {event_data}")
                await lifecycle_event_queue.put((event_type, event_data))
            
            # Set the stream callback to enable immediate streaming
            streaming_hooks.set_stream_callback(stream_lifecycle_event)
            
            print("🎣 Using lifecycle hooks for enhanced streaming")
            # Run the agent with streaming and lifecycle hooks
            result = Runner.run_streamed(agent, input=message, hooks=streaming_hooks, max_turns=50)
        else:
            print("📡 Using basic streaming (no lifecycle hooks)")
            # Run the agent with basic streaming
            result = Runner.run_streamed(agent, input=message, max_turns=50)
        
        # Send processing start event
        yield f"event: processing\ndata: {json.dumps({'status': 'started'})}\n\n"
        await asyncio.sleep(0.01)
        
        # Create output queue for coordinating between tasks
        output_queue = asyncio.Queue()
        
        # Create a separate task to continuously monitor and stream lifecycle events
        async def lifecycle_event_streamer():
            """Continuously monitor lifecycle events and stream them immediately."""
            try:
                while True:
                    if lifecycle_event_queue:
                        try:
                            # Wait for lifecycle events with a timeout
                            event_type, event_data = await asyncio.wait_for(
                                lifecycle_event_queue.get(), timeout=0.1
                            )
                            lifecycle_event_data = f"event: {event_type}\ndata: {json.dumps(event_data)}\n\n"
                            print(f"🎣 STREAMING LIFECYCLE IMMEDIATELY: {event_type} - {event_data}")
                            await output_queue.put(('lifecycle', lifecycle_event_data))
                        except asyncio.TimeoutError:
                            # No lifecycle events, continue monitoring
                            continue
                    else:
                        # No lifecycle queue, wait a bit
                        await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                print("🎣 Lifecycle event streamer cancelled")
                raise
            except Exception as e:
                print(f"🚨 Lifecycle event streamer error: {e}")
                error_data = {
                    'error': str(e),
                    'context': 'lifecycle_streamer'
                }
                await output_queue.put(('error', f"event: lifecycle_error\ndata: {json.dumps(error_data)}\n\n"))
        
        # Start the lifecycle event streamer task if we have lifecycle hooks
        if LIFECYCLE_AVAILABLE and streaming_hooks and lifecycle_event_queue:
            lifecycle_task = asyncio.create_task(lifecycle_event_streamer())
        
        # Main agent event processing task
        async def main_event_processor():
            """Process main agent events."""
            try:
                event_count = 0
                async for event in result.stream_events():
                    event_count += 1
                    event_type_name = getattr(event, 'type', 'unknown')
                    print(f"PROCESSING EVENT #{event_count}: {event_type_name} (Class: {type(event).__name__})")  # Debug log
                    
                    # Handle different event types with safe serialization
                    if hasattr(event, 'type'):
                        try:
                            # Safely extract event data
                            event_type = getattr(event, 'type', 'unknown')
                            
                            # Handle different event types specifically based on OpenAI Agents SDK
                            if event_type == 'response_created':
                                # For response created events, extract basic info
                                safe_data = {
                                    'type': 'response_created',
                                    'status': 'response_started'
                                }
                            elif event_type == 'response_chunk':
                                # For response chunks, extract the content
                                content = ""
                                if hasattr(event, 'data') and hasattr(event.data, 'content'):
                                    content = str(event.data.content)
                                elif hasattr(event, 'content'):
                                    content = str(event.content)
                                safe_data = {
                                    'type': 'response_chunk',
                                    'content': content
                                }
                            elif event_type == 'response_completed':
                                safe_data = {
                                    'type': 'response_completed',
                                    'status': 'response_finished'
                                }
                            elif 'stream_event' in event_type.lower():
                                # Handle generic stream events
                                safe_data = {
                                    'type': event_type,
                                    'status': 'stream_event_received'
                                }
                            else:
                                # For other events, try to safely extract data
                                safe_data = {
                                    'type': event_type,
                                    'raw_type': str(type(event).__name__)
                                }
                                
                                # Try to extract common attributes safely
                                if hasattr(event, 'data'):
                                    try:
                                        # Convert to string if it's not serializable
                                        data_str = str(event.data)
                                        safe_data['data_summary'] = data_str[:200] + "..." if len(data_str) > 200 else data_str
                                    except:
                                        safe_data['data_summary'] = "non-serializable data"
                                
                                # Try to extract text content if available
                                if hasattr(event, 'text'):
                                    try:
                                        safe_data['text'] = str(event.text)
                                    except:
                                        safe_data['text'] = "non-serializable text"
                            
                            event_data = {
                                'type': event_type,
                                'data': safe_data,
                                'timestamp': asyncio.get_event_loop().time()
                            }
                            
                            agent_event_data = f"event: agent_event\ndata: {json.dumps(event_data)}\n\n"
                            await output_queue.put(('agent', agent_event_data))
                            
                        except Exception as serialize_error:
                            print(f"Event serialization error: {serialize_error}")
                            # Send a safe error event
                            error_event = {
                                'type': 'serialization_error',
                                'data': {
                                    'error': str(serialize_error),
                                    'event_type': str(type(event).__name__)
                                },
                                'timestamp': asyncio.get_event_loop().time()
                            }
                            error_event_data = f"event: agent_event\ndata: {json.dumps(error_event)}\n\n"
                            await output_queue.put(('error', error_event_data))
                    
                    # Add periodic heartbeat for long-running operations
                    if event_count % 10 == 0:
                        heartbeat_data = f"event: heartbeat\ndata: {json.dumps({'events_processed': event_count})}\n\n"
                        await output_queue.put(('heartbeat', heartbeat_data))
                
                # Signal completion
                await output_queue.put(('complete', None))
                
            except Exception as stream_error:
                print(f"Main event processor error: {stream_error}")
                error_data = {
                    'error': str(stream_error),
                    'error_type': type(stream_error).__name__,
                    'context': 'main_event_processor'
                }
                error_event_data = f"event: stream_error\ndata: {json.dumps(error_data)}\n\n"
                await output_queue.put(('error', error_event_data))
                await output_queue.put(('complete', None))
        
        # Start the main event processor task
        main_task = asyncio.create_task(main_event_processor())
        
        # Stream events from the output queue
        try:
            while True:
                try:
                    # Get events from the output queue
                    event_type, event_data = await asyncio.wait_for(output_queue.get(), timeout=0.1)
                    
                    if event_type == 'complete':
                        # Main processing is complete
                        break
                    elif event_data:
                        yield event_data
                        await asyncio.sleep(0.001)  # Small delay to prevent overwhelming
                        
                except asyncio.TimeoutError:
                    # No events in queue, continue monitoring
                    # Check if main task is still running
                    if main_task.done():
                        break
                    continue
                    
        except Exception as output_error:
            print(f"Output streaming error: {output_error}")
            error_data = {
                'error': str(output_error),
                'context': 'output_streaming'
            }
            yield f"event: stream_error\ndata: {json.dumps(error_data)}\n\n"
        
        # Wait for tasks to complete
        if lifecycle_task and not lifecycle_task.done():
            lifecycle_task.cancel()
            try:
                await lifecycle_task
            except asyncio.CancelledError:
                pass
        
        if not main_task.done():
            main_task.cancel()
            try:
                await main_task
            except asyncio.CancelledError:
                pass
        
        # Send completion event with summary
        completion_data = {
            'status': 'finished',
            'timestamp': asyncio.get_event_loop().time()
        }
        yield f"event: complete\ndata: {json.dumps(completion_data)}\n\n"
        print("STREAMING COMPLETE")
        
    except asyncio.CancelledError:
        print("🚨 Stream was cancelled by client")
        yield f"event: cancelled\ndata: {json.dumps({'status': 'cancelled'})}\n\n"
        raise
    except Exception as e:
        print(f"🚨 STREAMING ERROR: {e}")
        error_data = {
            'error': str(e),
            'type': type(e).__name__,
            'timestamp': asyncio.get_event_loop().time()
        }
        yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
        await asyncio.sleep(0.01)
    finally:
        # Cleanup
        print("🧹 Cleaning up streaming resources")
        if lifecycle_task and not lifecycle_task.done():
            lifecycle_task.cancel()
            try:
                await lifecycle_task
            except asyncio.CancelledError:
                pass
        
        if streaming_hooks:
            streaming_hooks.event_queue.clear()
        if lifecycle_event_queue:
            # Clear any remaining items in the queue
            while not lifecycle_event_queue.empty():
                try:
                    lifecycle_event_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        
        # Send final cleanup event
        try:
            yield f"event: cleanup\ndata: {json.dumps({'status': 'completed'})}\n\n"
        except Exception as cleanup_error:
            print(f"Cleanup error (non-critical): {cleanup_error}")

@app.options("/chat/stream")
async def chat_stream_options():
    """Handle preflight OPTIONS requests for CORS."""
    return {"message": "OK"}

@app.post("/chat/stream")
async def chat_stream(chat_message: ChatMessage):
    """Stream chat responses from the OpenAI agent with separate event types."""
    try:
        return StreamingResponse(
            stream_agent_response(chat_message.message),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "Access-Control-Allow-Origin": "http://localhost:3000",
                "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
                "Transfer-Encoding": "chunked",
                "Content-Encoding": "identity"  # Disable compression
            }
        )
    except Exception as e:
        print(f"Stream endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "OpenAI Agents Chat API is running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file for document processing."""
    try:
        # Create documents directory if it doesn't exist
        documents_dir = Path("./documents")
        documents_dir.mkdir(exist_ok=True)
        
        # Save the uploaded file
        file_path = documents_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return {
            "status": "success",
            "message": f"File '{file.filename}' uploaded successfully",
            "file_path": str(file_path),
            "size": len(content),
            "instructions": "Use the chat interface to index and query this document with commands like: 'Index the document ./documents/" + file.filename + "'"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True,
        loop="asyncio",
        http="h11",  # Use h11 for better streaming support
        timeout_keep_alive=30,  # Keep connections alive for 30 seconds
        timeout_graceful_shutdown=30,  # Graceful shutdown timeout
        limit_max_requests=10000,  # Maximum requests per worker
        limit_concurrency=1000,  # Maximum concurrent connections
        backlog=2048  # Socket backlog size
    ) 