import asyncio
import random
import os
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agents import Agent, ItemHelpers, Runner, function_tool
from openai import AsyncOpenAI

import json
from dotenv import load_dotenv
from typing import Any

# Load environment variables
load_dotenv()

# Check if OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    pass

# Initialize OpenAI client for observation and reflection tools
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

# Using modern OpenAI Agents SDK streaming - no custom hooks needed!

@function_tool
async def how_many_jokes() -> int:
    """Returns a random number of jokes to tell (1-10)."""
    await asyncio.sleep(10)  # Wait for 5 seconds
    return random.randint(1, 10)

@function_tool
async def get_random_fact() -> str:
    """Returns a random interesting fact."""
    await asyncio.sleep(10)  # Wait for 5 seconds
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
    await asyncio.sleep(10)  # Wait for 5 seconds
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
        return observation_summary
        
    except Exception as e:
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
        return reflection_summary
        
    except Exception as e:
        return f"Reflection recorded: {reflection}"

@function_tool
async def thought_tool(thought: str) -> str:
    """Record a thought or reasoning step about the user's request or the current situation. Use this tool to document your internal reasoning or hypotheses before taking action."""
    # You can add custom logic here, or just echo the thought
    return f"Thought recorded: {thought}"

@function_tool
async def plan_tool(plan: str) -> str:
    """Record a plan or step-by-step approach for solving the user's request. Use this tool to outline your intended actions or strategies."""
    return f"Plan recorded: {plan}"

@function_tool
async def act_tool(action: str) -> str:
    """Record an action or execution step taken to address the user's request. Use this tool to document what action you are performing or have performed."""
    return f"Action recorded: {action}"

@function_tool
async def decision_tool(decision: str) -> str:
    """Record a decision or conclusion made after reasoning, planning, and acting. Use this tool to document your final choice or next step based on all available information."""
    return f"Decision recorded: {decision}"

async def stream_agent_response(message: str) -> AsyncGenerator[str, None]:
    """Stream the agent's response using modern OpenAI Agents SDK streaming."""
    
    try:
        # Send initial connection event
        yield f"event: connection\ndata: {json.dumps({'status': 'connected'})}\n\n"
        await asyncio.sleep(0.01)
        
        # Create agent with tools - much simpler configuration
        agent = Agent(
            name="Assistant",
            model="gpt-4o",
            instructions="""
You are an intelligent assistant. Internally, you may use step-by-step reasoning, planning, and tool use (ReAct), but your reply to the user should only include the final answer or message, not your internal thoughts, plans, or tool calls. Only output the final user-facing message.
""",
            tools=[how_many_jokes, get_random_fact, calculate_math, get_weather_info, observe_result, reflect_on_progress, thought_tool, plan_tool, act_tool, decision_tool],
        )
        
        # Send processing start event
        yield f"event: processing\ndata: {json.dumps({'status': 'started'})}\n\n"
        await asyncio.sleep(0.01)
        
        # Use the modern streaming API - much simpler!
        result = Runner.run_streamed(agent, input=message, max_turns=50)

       
        
        # Stream events directly from the SDK
        async for event in result.stream_events():
            try:
                # print("\n\n event: ", event)
                # Pass through the event directly
                def to_dict(obj):
                    if isinstance(obj, dict):
                        return {k: to_dict(v) for k, v in obj.items()}
                    elif hasattr(obj, "__dict__"):
                        return {k: to_dict(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
                    elif isinstance(obj, (list, tuple)):
                        return [to_dict(i) for i in obj]
                    else:
                        return obj

                event_dict = to_dict(event)
     
                yield f"event: {event.type}\ndata: {json.dumps(event_dict)}\n\n"
                await asyncio.sleep(0.001)  # Small delay to prevent overwhelming
                
            except Exception as event_error:
                # Handle individual event errors gracefully
                error_data = {
                    'type': 'event_error',
                    'error': str(event_error),
                    'event_type': getattr(event, 'type', 'unknown'),
                    'timestamp': asyncio.get_event_loop().time()
                }
                yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
        
        # Send completion event
        completion_data = {
            'status': 'finished',
            'timestamp': asyncio.get_event_loop().time()
        }
        yield f"event: complete\ndata: {json.dumps(completion_data)}\n\n"
        
    except asyncio.CancelledError:
        # Handle cancellation gracefully
        yield f"event: cancelled\ndata: {json.dumps({'status': 'cancelled'})}\n\n"
        raise
    except Exception as e:
        # Handle any other errors
        error_data = {
            'error': str(e),
            'type': type(e).__name__,
            'timestamp': asyncio.get_event_loop().time()
        }
        yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
        await asyncio.sleep(0.01)
    finally:
        # Send final cleanup event
        try:
            yield f"event: cleanup\ndata: {json.dumps({'status': 'completed'})}\n\n"
        except Exception:
            pass

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
        raise HTTPException(status_code=500, detail=str(e))


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