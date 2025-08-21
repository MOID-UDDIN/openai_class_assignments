from agents import Runner, Agent, OpenAIChatCompletionsModel, RunConfig
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import chainlit as cl


load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
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
    name="FAQ bot",
    instructions='''You are a helpful FAQ bot. 
    You must answer frequently asked questions such as: 
    "What is your name?", "What can you do?", etc. 
    Respond clearly and helpfully.''', 
    model=model)

@cl.on_chat_start
async def handle_start_chat():
    await cl.Message(content="Welcome to my FAQ bot").send()

@cl.on_message
async def handle_message(message: cl.Message):
    

    msg = cl.Message(content="let me think")
    await msg.send()


    result = Runner.run_sync(
        agent,
        input=message.content,
        run_config=config
    )

    msg.content = result.final_output
    await msg.update()