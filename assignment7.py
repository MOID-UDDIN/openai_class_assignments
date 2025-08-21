from dotenv import load_dotenv
import os
from agents import Agent,Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, function_tool
from tavily import TavilyClient



load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

tavily_client = TavilyClient(api_key=tavily_api_key)

# Check if the API key is present; if not, raise an error
if not gemini_api_key or not tavily_api_key:
    raise ValueError("GEMINI_API_KEY or TAVILY_API_KEY is not set. Please ensure it is defined in your .env file.")

#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

@function_tool
def web_search(query:str)-> str:
    """Web Search Provider"""
    print("web_search tool fire...")
    response = tavily_client.search(query)
    return response


agent = Agent(
    name = "General Assistant",
    instructions = "You are a general assistant.",
    tools=[web_search]
)

msg = input("Enter your query : ")
result = Runner.run_sync(
    agent,
    input = msg,
    run_config = config
    )

print(result.final_output)