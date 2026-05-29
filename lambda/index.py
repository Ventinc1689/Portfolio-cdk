import json
import boto3
import os
import logfire
from pydantic import TypeAdapter
from botocore.config import Config
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

# Set up logfire
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

# ============================================
# AWS clients
# ============================================
dynamodb = boto3.resource('dynamodb')
TABLE_NAME = os.environ['TABLE_NAME']
table = dynamodb.Table(TABLE_NAME)

config = Config(
    retries={
        'max_attempts': 5,
        'mode': 'adaptive' # Rate limiting
    }
)

bedrock_agent_client = boto3.client(
    'bedrock-agent-runtime', 
    region_name='us-east-1', 
    config=config
)

bedrock = BedrockConverseModel(
    'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
)

# Agent responsible for answering questions about the resume using retrieved context from the knowledge base
resume_agent = Agent(
    model = bedrock,
    system_prompt=(
        "You are a portfolio assistant for Vincent Zhu. Your job is to answer questions about Vincent's resume, projects, skills, experience, education. Always call search_resume first to get accurate info from Vincent's resume."
    )
)

# Function to search the resume knowledge base using Bedrock Agent Runtime
@resume_agent.tool
def search_resume(ctx: RunContext, query: str) -> str:
    """Search Vincent's resume knowledge base"""

    kb_id = os.environ.get('KNOWLEDGE_BASE_ID', '')
    if not kb_id:
        return "Knowledge base not configured"
    
    try:
        response = bedrock_agent_client.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={'text': query},
            retrievalConfiguration={
                'vectorSearchConfiguration': {'numberOfResults': 3}
            }
        )

        results = response.get('retrievalResults', [])

        if not results:
            return "NOT_FOUND"
        
        chunks = [r['content']['text'] for r in results if 'content' in r]
        return "\n\n---\n\n".join(chunks)
    except Exception as e:
        logfire.error(f"KB retrieval failed: {e}")
        return f"Error retrieving from knowledge base: {e}"

# Agent responsible for searching the web
web_agent = Agent(
    model = bedrock,
    tools=[duckduckgo_search_tool()],
    system_prompt=(
        'Your job is to answer general knowledge questions, current events, or resume-unrelated information by searching the web. Provide concise, factual answers.'
    )
)

# The main agent that routes between resume_agent and web_agent based on the user's question. It uses a system prompt to determine which tool to call, and is responsible for providing the final answer to the user.
agent = Agent(
    model = bedrock,
    system_prompt=(
        "Your job is to determine which tool/agent to call based on the user's question. If the question is about Vincent's resume, experience, projects, skills, or education, call the ask_resume_agent. For general knowledge questions, current events, or anything not related to Vincent's resume, call the ask_web_agent. Write a friendly, concise final answer for the user. Do not preface with 'Based on Vincent's resume' or anything of that sort"
    )
)

# Calls resume agent
@agent.tool
async def ask_resume_agent(ctx: RunContext, query: str) -> str:
    """Search Vincent's resume knowledge base"""

    logfire.info(f"Invoking resume search for query: {query}")
    result = await resume_agent.run(query)
    return str(result.output)

# Calls web agent
@agent.tool
async def ask_web_agent(ctx: RunContext, query: str) -> str:
    """Search the web for general knowledge questions or current events"""
    
    logfire.info(f"Invoking web search for query: {query}")
    result = await web_agent.run(query)
    return str(result.output)

@agent.tool
def get_current_location(ctx: RunContext):
    """Example of a simple tool that returns Vincent's current location. In a real implementation, this could call an API or database to get dynamic information."""
    return "Pittsburgh, PA"

@agent.tool
def get_weather(ctx: RunContext, location: str):    
    """Example of a simple tool that returns the current weather. In a real implementation, this could call a weather API to get dynamic information."""
    return "Partly cloudy, 72°F"

# Helper function to get chat history from DynamoDB
def get_chat(session_id):
    try:
        response = table.get_item(Key={'sessionId': session_id})
        if 'Item' in response and 'history' in response['Item']:
            return response['Item']['history']
    except Exception as e:
        print(f"Error reading DynamoDB: {e}")
    return "[]"

# Helper function to extract display messages for the frontend, filtering out tool calls and other non-display content
def extract_display_messages(history_string):
    try:
        history = json.loads(history_string)
        messages = []
        for entry in history:
            if entry.get('kind') == 'request':
                for part in entry.get('parts', []):
                    if part.get('part_kind') == 'user-prompt':
                        messages.append({'role': 'user', 'content': part['content']})
            elif entry.get('kind') == 'response':
                for part in entry.get('parts', []):
                    if part.get('part_kind') == 'text':
                        messages.append({'role': 'assistant', 'content': part['content']})
        return messages
    except Exception as e:
        print(f"Error extracting display messages: {e}")
        return []

# Set up a reusable Pydantic TypeAdapter to handle history list parsing
messages_adapter = TypeAdapter(list[ModelMessage])

# The main Lambda handler
def lambda_handler(event, context):
    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'POST, OPTIONS'
    }
    
    try:
        try:
            body = json.loads(event.get('body', '{}'))
        except:
            body = {}
        
        # Fetch session ID 
        session_id = body.get('sessionId', 'default-session')

        # Page refresh
        if body.get('init') == True:
            history_string = get_chat(session_id)
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({'history': extract_display_messages(history_string)})
            }

        # Clear/Delete chat history
        if body.get('clear') == True:
            try:
                table.delete_item(Key={'sessionId': session_id})
                return {
                    'statusCode': 200,
                    'headers': headers,
                    'body': json.dumps({'message': 'History deleted successfully'})
                }
            except Exception as db_err:
                print(f"Error deleting row from DynamoDB: {db_err}")
                return {'statusCode': 500, 'headers': headers, 'body': json.dumps({'error': str(db_err)})}

        # Standard Chat Turn
        question = body.get('question', '')
        if not question:
            return {'statusCode': 400, 'headers': headers, 'body': json.dumps({'error': 'No question'})}
        
        history_string = get_chat(session_id)
        
        try:
            history = messages_adapter.validate_json(history_string)
        except Exception as e:
            print(f"Schema mismatch detected, starting fresh: {e}")
            history = []

        with logfire.span('agent_run', session_id=session_id, question=question):
            result = agent.run_sync(
                user_prompt = question,
                message_history = history
            )
        logfire.force_flush()

        full_history = result.all_messages_json().decode('utf-8')

        try:
            table.put_item(Item={
                'sessionId': session_id,
                'history': full_history
            })
        except Exception as e:
            print(f"DB Save error: {e}")

        used_kb = False
        used_search = False
        for msg in result.new_messages():
            for p in getattr(msg, 'parts', []):
                tool_name = (getattr(p, 'tool_name', '') or '').lower()
                if getattr(p, 'part_kind', None) == 'tool-call':
                    if 'resume' in tool_name:
                        used_kb = True
                    elif 'web' in tool_name:
                        used_search = True

        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps({
                'answer': str(result.output),
                'used_kb': used_kb,
                'used_search': used_search,
            })
        }
        
    except Exception as e:
        print(f"Lambda crash: {str(e)}")
        logfire.error(f"Lambda crash: {str(e)}")
        logfire.force_flush()
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({'error': str(e)})
        }