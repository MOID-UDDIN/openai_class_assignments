from typing import Any
from openai import AsyncOpenAI
from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    Runner,
    OpenAIChatCompletionsModel,
    TResponseInputItem,
    input_guardrail,
    output_guardrail,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    set_tracing_disabled
)
from dotenv import find_dotenv, load_dotenv
import os
import asyncio
from pydantic import BaseModel


set_tracing_disabled(True)
# ------------------ Load API Keys ------------------
load_dotenv(find_dotenv(), override=True)

api_key = os.getenv("GEMINI_API_KEY")
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

client = AsyncOpenAI(api_key=api_key, base_url=base_url)
model = OpenAIChatCompletionsModel(openai_client=client, model="gemini-2.5-flash")




# ------------------ Input Guardrail ------------------
class MathOutPut(BaseModel):
    is_math: bool
    reason: str

@input_guardrail
async def check_input(
    ctx: RunContextWrapper[Any], agent: Agent[Any], input_data: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    input_agent = Agent(
        "InputGuardrailAgent",
        instructions="Check and verify if input is related to math.",
        model=model,
        output_type=MathOutPut,
    )
    result = await Runner.run(input_agent, input_data, context=ctx.context)
    final_output = result.final_output

    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_math
    )


# ------------------ Output Guardrail ------------------
class SafeOutputCheck(BaseModel):
    is_safe: bool
    reason: str

@output_guardrail
async def check_output(
    ctx: RunContextWrapper[Any], agent: Agent[Any], output_data: str
) -> GuardrailFunctionOutput:
    output_agent = Agent(
        "OutputGuardrailAgent",
        instructions=(
            "Analyze the given output and determine if it contains any reference to politics, "
            "elections, or political figures (e.g., presidents, prime ministers, ministers, political parties). "
            "If any such reference exists, respond with is_safe=False and give a short reason. "
            "If not, respond with is_safe=True and a short reason."
        ),
        model=model,
        output_type=SafeOutputCheck,
    )

    result = await Runner.run(output_agent, output_data, context=ctx.context)
    final_output = result.final_output

    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_safe
    )

math_agent = Agent(
    "MathAgent",
    instructions="You are a math agent.",
    model=model,
    input_guardrails=[check_input],
)


general_agent = Agent(
    "GeneralAgent",
    instructions="You are a helpful agent.",
    model=model,
    output_guardrails=[check_output],
)



async def main():
    try:
        msg = input("Enter your question: ")
        result = await Runner.run(general_agent, msg)
        print(f"\n\nFinal Output: {result.final_output}")
    except InputGuardrailTripwireTriggered:
        print("Error: Invalid prompt (Input not allowed).")
    except OutputGuardrailTripwireTriggered:
        print("Error: Response contains political content. Cannot display.")

if __name__ == "__main__":
    asyncio.run(main())