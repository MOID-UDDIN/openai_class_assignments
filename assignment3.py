from agents import Agent, OpenAIChatCompletionsModel, Runner, RunConfig, function_tool
from openai import AsyncOpenAI
import chainlit as cl
import os
from dotenv import load_dotenv

load_dotenv()

@function_tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@function_tool
def subtract(a: int, b: int) -> int:
    """Add two numbers."""
    return a - b

@function_tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
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

agent: Agent = Agent(
    name="Math Agent",
    instructions="You are a smart math agent. Use the provided tools to solve math questions when needed.",
    model=model,
    tools=[add, subtract, multiply],
)

@cl.on_chat_start
async def start():
    await cl.Message(content="Hi! I'm a Math Bot. You can ask me math-related questions.").send()

@cl.on_message
async def handle(message: cl.Message):
    thinking = cl.Message(content="Let me calculate that...")
    await thinking.send()

    result = Runner.run_sync(
        agent,
        input=message.content,
        run_config=config
    )

    thinking.content = result.final_output
    await thinking.update()