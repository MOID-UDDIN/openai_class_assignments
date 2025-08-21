from agents import Runner, set_tracing_disabled, Agent, function_tool, RunContextWrapper
from pydantic import BaseModel
from agents import AsyncOpenAI, OpenAIChatCompletionsModel
from dotenv import find_dotenv, load_dotenv
import os

set_tracing_disabled(True)
load_dotenv(find_dotenv(), override=True)

api_key = os.getenv("GEMINI_API_KEY")
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

client = AsyncOpenAI(api_key=api_key, base_url=base_url)
model = OpenAIChatCompletionsModel(openai_client=client, model="gemini-2.5-flash")


class HotelSchema(BaseModel):
    hotel_name: str
    hotel_rooms: int
    hotel_price: int
    hotel_facilities: list  


hotel1 = HotelSchema(
    hotel_name="marriott hotel",
    hotel_rooms=120,
    hotel_price=18000,
    hotel_facilities=["Free WiFi", "Spa", "Restaurant"]
)
hotel2 = HotelSchema(
    hotel_name="pearl continental",
    hotel_rooms=100,
    hotel_price=15000,
    hotel_facilities=["Free WiFi", "Pool", "Gym"]
)
hotel3 = HotelSchema(
    hotel_name="serena hotel",
    hotel_rooms=80,
    hotel_price=20000,
    hotel_facilities=["Free WiFi", "Garden", "Conference Hall"]
)

hotels = [hotel1, hotel2, hotel3]

@function_tool
def get_hotel_info(ctx: RunContextWrapper):
    """Fetch hotel information"""
    hotel: HotelSchema = ctx.context
    print("Hotel info tool fired --->")
    return f"{hotel.hotel_name} - Rooms: {hotel.hotel_rooms}, Price: Rs {hotel.hotel_price}, Facilities: {hotel.hotel_facilities}"


def dynamic_instruction(ctx: RunContextWrapper[HotelSchema], agent):
    return (
        f"You are a hotel booking assistant of {ctx.context.hotel_name}."
        f"If the user asks about {ctx.context.hotel_name}, use the get_hotel_info tool. "
        f"If the user asks about a hotel that is not in your context, say: "
        f"'We don't have any information regarding this hotel.'"
    )

assistant = Agent(
    name="Hotel Assistant",
    instructions=dynamic_instruction,
    model=model,
    tools=[get_hotel_info]
)


set_tracing_disabled(True)
msg = input("Enter your question: ").lower()

# ---- Hotel context selection ----
selected_hotel = None
msg_lower = msg.lower()

for h in hotels:
    hotel_name_lower = h.hotel_name.lower()
    hotel_keywords = hotel_name_lower.split()

    
    if any(keyword in msg_lower for keyword in hotel_keywords):
        selected_hotel = h
        break

if selected_hotel:
    res = Runner.run_sync(
        starting_agent=assistant,
        input=msg,
        context=selected_hotel
    )
    print(res.final_output)
else:
    print("We don't have any information regarding this hotel.")