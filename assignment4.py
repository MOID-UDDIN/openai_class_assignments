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
    return f"The Weather in {city} is {data['current']['temp_c']}Celsius with {data['current']['condition']['text']}."
    


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
    name="weather Agent",
    instructions="You are a weather agent. If the user asks for weather, call the `get_weather` function with the city name.",
    model=model,
    tools=[get_weather],
)

@cl.on_chat_start
async def start():
    await cl.Message(content="Hi! I'm a Weather Bot. You can ask me weather information of any particular city.").send()

@cl.on_message
async def handle(message: cl.Message):
    thinking = cl.Message(content="Let me check...")
    await thinking.send()

    result = Runner.run_sync(
        agent,
        input=message.content,
        run_config=config
    )

    thinking.content = result.final_output
    await thinking.update()