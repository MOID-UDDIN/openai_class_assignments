from agents import Agent, OpenAIChatCompletionsModel, Runner, RunConfig, function_tool
from openai import AsyncOpenAI
import chainlit as cl
import os
from dotenv import load_dotenv
import requests

load_dotenv()

weather_key=os.getenv("WEATHER_API_KEY")
@function_tool
def get_weather(city:str):
    """Get weather of any city"""
    response = requests.get(f"http://api.weatherapi.com/v1/current.json?key={weather_key}&q={city}")
    data = response.json()
    return f"The Weather in {city} is {data['current']['temp_c']}Celsius and {data['current']['condition']['text']}."

@function_tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
    


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
    name="multi Agent",
    instructions="If the user asks for weather, call the `get_weather` function with the city name." 
    "If the user asks to add two numbers, call the `add` function.",
    model=model,
    tools=[get_weather,add],
)

@cl.on_chat_start
async def start():
    await cl.Message(content="Hi! I'm a multi talented Bot. You can ask me about a city's weather or to add two numbers.").send()

@cl.on_message
async def handle(message: cl.Message):
    thinking = cl.Message(content="wait...")
    await thinking.send()

    result = Runner.run_sync(
        agent,
        input=message.content,
        run_config=config
    )

    thinking.content = result.final_output
    await thinking.update()