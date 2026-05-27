import json
import boto3
import os
import re
import logfire
from pydantic import TypeAdapter
from botocore.config import Config
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

dynamodb = boto3.resource('dynamodb')
TABLE_NAME = os.environ['TABLE_NAME']
table = dynamodb.Table(TABLE_NAME)

config = Config(
    retries={
        'max_attempts': 5,
        'mode': 'adaptive' # Rate limiting
    }
)

bedrock_agent_client = boto3.client('bedrock-agent-runtime', region_name='us-east-1', config=config)

bedrock = BedrockConverseModel(
    'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
)

agent = Agent(
    model=bedrock,
    tools=[duckduckgo_search_tool()],
    system_prompt=(
        "You are a portfolio assistant for Vincent Zhu. Answer user questions based on <resume_context> when dealing with resume information. Use available tools when information is outside of resume context or questions you do not have answer to. Be friendly and concise. Do not preface answers with 'Based on resume' or anything of that sort."
    )
)

def get_chat(session_id):
    try:
        response = table.get_item(Key={'sessionId': session_id})
        if 'Item' in response and 'history' in response['Item']:
            return response['Item']['history']
    except Exception as e:
        print(f"Error reading DynamoDB: {e}")
    return "[]"

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

def query_knowledge_base(query_text: str) -> str:
    KNOWLEDGE_BASE_ID = os.environ.get('KNOWLEDGE_BASE_ID', '')

    if not KNOWLEDGE_BASE_ID:
        return "No resume context available."
    try:
        response = bedrock_agent_client.retrieve(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            retrievalQuery={'text': query_text},
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
        print(f"Error querying Bedrock KB: {e}")
        return "Error fetching resume context."

# Set up a reusable Pydantic TypeAdapter to handle history list parsing
messages_adapter = TypeAdapter(list[ModelMessage])

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
        except Exception as parse_err:
            print(f"Schema mismatch detected, starting fresh: {parse_err}")
            history = []
            
        relevant_chunks = query_knowledge_base(question)

        if relevant_chunks and relevant_chunks not in ("NOT_FOUND", "No resume context available.") and not relevant_chunks.startswith("Error"):
            augmented_prompt = (
                f"<resume_context>\n{relevant_chunks}\n</resume_context>\n\n"
                f"User question: {question}"
            )
        else:
            augmented_prompt = (
                f"<resume_context>\nNo new context retrieved. Check chat history for follow-up context.\n</resume_context>\n\n"
                f"User question: {question}"
            )
        
        # Invoke the Agent
        result = agent.run_sync(
            user_prompt=augmented_prompt,
            message_history=history,
            deps=relevant_chunks,
        )

        raw_messages = json.loads(result.all_messages_json().decode('utf-8'))
        cleaned_messages = []
        
        KEEP_PARTS = {'user-prompt', 'text'}

        for msg in raw_messages:
            if 'parts' in msg:
                filtered_parts = [p for p in msg['parts'] if p.get('part_kind') in KEEP_PARTS]
                if len(filtered_parts) > 0:
                    msg['parts'] = filtered_parts
                    cleaned_messages.append(msg)
            else:
                cleaned_messages.append(msg)

        # Strip resume_context blocks from user prompts before saving
        # This keeps DynamoDB history lean — context is re-retrieved fresh each turn
        RESUME_CONTEXT_PATTERN = re.compile(
            r'<resume_context>.*?</resume_context>\s*\n*User question:\s*',
            flags=re.DOTALL
        )

        for msg in cleaned_messages:
            if msg.get('kind') == 'request':
                for part in msg.get('parts', []):
                    if part.get('part_kind') == 'user-prompt':
                        part['content'] = RESUME_CONTEXT_PATTERN.sub('', part['content'])
                
        # Save the schema-safe cleaned array
        updated_history = json.dumps(cleaned_messages)

        try:
            table.put_item(Item={
                'sessionId': session_id,
                'history': updated_history
            })
        except Exception as db_save_err:
            print(f"DB Save error: {db_save_err}")

        used_search = any(
            getattr(p, 'part_kind', None) == 'tool-call' and 'duckduckgo' in getattr(p, 'tool_name', '').lower()
            for msg in result.new_messages()
            for p in getattr(msg, 'parts', [])
        )

        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps({
                'answer': str(result.output), 
                'used_search': used_search, 
                'debug_kb_chunks': relevant_chunks
            })
        }
        
    except Exception as e:
        print(f"Lambda crash: {str(e)}")
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({'error': str(e)})
        }