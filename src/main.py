import streamlit as st
from openai import OpenAI
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from datetime import datetime
from template import PROMPT_TEMPLATE
import os
from langchain.schema import LLMResult
from uuid import uuid4

# Get API keys from Streamlit secrets
api_key = st.secrets["OPENAI_API_KEY"]
langchain_api_key = st.secrets.get("LANGCHAIN_API_KEY")

# Set up LangChain API key
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "UCLA_TESTING"

# Initialize callback manager with tracer
tracer = LangChainTracer(
    project_name="ucla-post-op-care"
)
callback_manager = CallbackManager([tracer])

# Instantiate OpenAI client
client = OpenAI(api_key=api_key)

# Configure Streamlit page
st.set_page_config(
    page_title="UCLA Post-Op Care Assistant",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state for message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to get chatbot response
def get_chatbot_response(user_input):
    # Generate run_id outside try block
    run_id = str(uuid4())
    
    tracer.on_llm_start(
        serialized={},
        prompts=[user_input],
        run_id=run_id
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": PROMPT_TEMPLATE},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        assistant_response = response.choices[0].message.content
        
        # Log successful completion
        tracer.on_llm_end(
            response=LLMResult(
                generations=[[{"text": assistant_response}]],
                llm_output={"model": "gpt-4"}
            ),
            run_id=run_id
        )
        
        return assistant_response
            
    except Exception as e:
        # Log error if it occurs
        tracer.on_llm_error(
            error=str(e),
            run_id=run_id
        )
        return f"Error: {str(e)}"

# Streamlit UI
st.title("🏥 UCLA Post-Op Care Assistant")

# Sidebar with information
with st.sidebar:
    st.header("Important Information")
    st.warning("For Emergencies Call 911")
    st.info("""
    **Urgent Issues (Off hours/weekends)**
    Call: (310) 206-6766
    Ask for: Plastic Surgery Resident

    **Weekday Issues (8AM-5PM)**
    Call: (310) 794-7616
    Email: estayton@mednet.ucla.edu
    """)

# Main chat interface
user_input = st.chat_input("Type your question here...")

if user_input:
    with st.spinner('Getting response...'):
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get assistant response
        assistant_response = get_chatbot_response(user_input)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

# Display message history
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).write(content)

# Footer
st.markdown("---")
st.markdown("*This is an AI assistant. For medical emergencies, please call 911 or contact the clinic directly.*")