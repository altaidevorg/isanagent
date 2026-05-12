import asyncio
import os
from typing import Literal
from pydantic import BaseModel, Field
from afterimage import (
    ConversationGenerator,
    ToolCallingInstructionGeneratorCallback,
)

# Tool Schemas
class SearchFlightsArgs(BaseModel):
    origin: str = Field(description="3-letter airport code for departure.")
    destination: str = Field(description="3-letter airport code for arrival.")
    date: str = Field(description="Date of travel in YYYY-MM-DD format.")

class SearchFlights(BaseModel):
    """Search for available flights between two cities on a specific date."""
    name: Literal["search_flights"] = "search_flights"
    arguments: SearchFlightsArgs

class BookFlightArgs(BaseModel):
    flight_number: str = Field(description="The flight number to book.")
    passenger_name: str = Field(description="Full name of the passenger.")

class BookFlight(BaseModel):
    """Book a specific flight for a passenger."""
    name: Literal["book_flight"] = "book_flight"
    arguments: BookFlightArgs

def tool_model_to_openai_schema(tool_model):
    name = tool_model.model_fields["name"].default
    args_schema = tool_model.model_fields["arguments"].annotation.model_json_schema()
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": tool_model.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": args_schema.get("properties", {}),
                "required": args_schema.get("required", []),
            },
        },
    }

from typing import List, Union
from afterimage import AsyncStructuredGenerator

class AnyToolCall(BaseModel):
    function: Union[SearchFlights, BookFlight]

class ToolInvocation(BaseModel):
    reasoning: str = Field(description="Reasoning for selecting the tool.")
    response: str = Field(description="The final response to the user.")
    tool_calls: List[AnyToolCall] = Field(description="A list of tool calls to execute.")

async def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY")
        return

    tools = [SearchFlights, BookFlight]

    from afterimage import InMemoryDocumentProvider
    docs = InMemoryDocumentProvider(["Flight booking system documentation."])

    instruction_cb = ToolCallingInstructionGeneratorCallback(
        api_key=api_key,
        documents=docs,
        model_name="gemini-2.5-flash",
        tools=tools,
        n_instructions=3,
    )

    respondent_prompt = "You are a helpful flight booking assistant. Use the provided tools to search for and book flights based on user requests."

    gen = AsyncStructuredGenerator(
        output_schema=ToolInvocation,
        respondent_prompt=respondent_prompt,
        api_key=api_key,
        model_name="gemini-2.5-flash",
        instruction_generator_callback=instruction_cb,
    )
    
    await gen.generate(num_samples=3)

if __name__ == "__main__":
    asyncio.run(main())
