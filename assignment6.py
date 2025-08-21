from dotenv import load_dotenv
import os
from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    function_tool,
    GuardrailFunctionOutput,
    input_guardrail,
    RunContextWrapper,
    TResponseInputItem,
    InputGuardrailTripwireTriggered,
    ModelSettings,
    set_tracing_disabled
)
from pydantic import BaseModel
from typing import Any
import asyncio

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in .env file")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

set_tracing_disabled(True)

fake_orders = {
    "123": {"status": "Shipped", "product": "Laptop", "price": "$1000"},
    "456": {"status": "Processing", "product": "Mobile Phone", "price": "$500"},
    "789": {"status": "Delivered", "product": "Headphones", "price": "$100"},
}

class OrderStatusOutput(BaseModel):
    order_id: str
    status: str
    product: str
    price: str


@function_tool
def get_order_status(order_id: str) -> OrderStatusOutput:
    """Simulates fetching order status"""
    if order_id in fake_orders:
        data = fake_orders[order_id]
        return OrderStatusOutput(
            order_id=order_id,
            status=data["status"],
            product=data["product"],
            price=data["price"]
        )
    else:
        return OrderStatusOutput(
            order_id=order_id,
            status="Not Found",
            product="N/A",
            price="N/A"
        )

class NegativeCheckOutput(BaseModel):
    is_safe: bool
    reason: str

@input_guardrail
async def check_negative_input(
    ctx: RunContextWrapper[Any], agent: Agent[Any], input_data: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    guard_agent = Agent(
        "NegativeLanguageGuard",
        instructions=(
            "Check if the input contains offensive, rude, or negative language. "
            "If yes, respond with is_safe=False and give reason. "
            "Otherwise respond with is_safe=True."
        ),
        model=model,
        output_type=NegativeCheckOutput,
    )

    result = await Runner.run(guard_agent, input_data, context=ctx.context)
    final_output = result.final_output

    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_safe
    )

human_agent = Agent(
    name="Human Agent",
    instructions="You are a helpful human support agent. Handle escalated queries with empathy.",
    model=model,
)

bot_agent = Agent(
    name="Bot Agent",
    instructions="""
You are a smart customer support bot.
- Answer FAQs about products.
- Use get_order_status tool to fetch order status.
- If input looks complex or sentiment is negative, escalate to HumanAgent.
""",
    model=model,
    tools=[get_order_status],
    input_guardrails=[check_negative_input],
    handoffs=[human_agent], 
    model_settings=ModelSettings(tool_choice="auto")
)

async def main():
    try:
        msg = input("Enter your question: ")
        result = await Runner.run(bot_agent, input=msg)
        print(result.final_output)

    except InputGuardrailTripwireTriggered:
        print("Input blocked: Offensive or unsafe language detected.")

if __name__ == "__main__":
    asyncio.run(main())
