# OpenAI Agents FastAPI Backend

A FastAPI backend using OpenAI Agents SDK with streaming responses and multiple tools.

## Features

- **Multiple Tools**: Joke generator, random facts, math calculator, and weather info
- **Streaming Responses**: Real-time streaming compatible with the ChatBox frontend
- **CORS Support**: Configured for frontend integration
- **Tool Visualization**: Shows tool calls and outputs during conversations

## Setup

1. **Navigate to the backend directory**:

   ```bash
   cd backend
   ```

2. **Activate the virtual environment**:

   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up OpenAI API key**:

   ```bash
   export OPENAI_API_KEY=your_openai_api_key_here
   ```

   Or create a `.env` file in the backend directory:

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Running the Server

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server will start on `http://localhost:8000`.

## API Endpoints

### POST /chat/stream

Stream chat responses from the OpenAI agent.

**Request Body**:

```json
{
  "message": "Tell me a joke"
}
```

**Response**: Server-sent events with streaming text.

### GET /health

Health check endpoint.

## Available Tools

The agent has access to these tools:

1. **how_many_jokes()**: Returns a random number of jokes to tell
2. **get_random_fact()**: Returns interesting random facts
3. **calculate_math(expression)**: Evaluates basic math expressions
4. **get_weather_info(city)**: Returns mock weather information

## Example Usage

- "Tell me some jokes" - Uses the joke tool
- "What's 15 \* 7?" - Uses the calculator tool
- "Give me a random fact" - Uses the fact tool
- "What's the weather in Tokyo?" - Uses the weather tool

## Testing

You can test the API using curl:

```bash
curl -X POST "http://localhost:8000/chat/stream" \
     -H "Content-Type: application/json" \
     -d '{"message": "Tell me a joke"}'
```

## Integration with Frontend

This backend is designed to work with the ChatBox.tsx component in the frontend. The streaming response format matches what the frontend expects:

- Each chunk is prefixed with `data: `
- Tool calls and outputs are clearly marked
- Word-by-word streaming for better user experience
