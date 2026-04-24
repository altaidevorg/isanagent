import asyncio
import os
from typing import List, Literal, Type, Union
from pydantic import BaseModel, Field

from afterimage import (
    StructuredGenerator,
    PersonaGenerator,
    ToolCallingInstructionGeneratorCallback,
    InMemoryDocumentProvider,
)


def tool_model_to_openai_schema(tool_model: Type[BaseModel]):
    """
    Convert `<Tool>.arguments` Pydantic model into
    the OpenAI function schema format.
    """
    name = tool_model.model_fields["name"].default
    args_model = tool_model.model_fields["arguments"].annotation

    args_schema = args_model.model_json_schema()

    params = {
        "type": "object",
        "properties": args_schema.get("properties", {}),
        "required": args_schema.get("required", []),
    }

    # normalize types to UPPERCASE dialect
    for k, v in params["properties"].items():
        if "title" in v:
            del v["title"]

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": tool_model.__doc__ or "",
            "parameters": params,
        },
    }


def tools_to_function_schemas(models: List[Type[BaseModel]]):
    return [tool_model_to_openai_schema(m) for m in models]


# --- Schema Definitions ---

# We define specific arguments for each tool to ensure strict structured output.


class Color(BaseModel):
    r: int = Field(255, description="Red component (0-255).")
    g: int = Field(255, description="Green component (0-255).")
    b: int = Field(255, description="Blue component (0-255).")


class TurnOnLightArgs(BaseModel):
    room: Literal["kitchen", "bedroom", "living_room", "kids_room", "bathroom"] = Field(
        description="The name of the room."
    )
    brightness: int = Field(80, description="Brightness percentage (0-100).")
    color: str = Field("white", description="Color of the light.")


class TurnOnLight(BaseModel):
    """Turn on a light in a specific room."""

    name: Literal["turn_on_light"] = "turn_on_light"
    arguments: TurnOnLightArgs


class TurnOffLightArgs(BaseModel):
    room: Literal["kitchen", "bedroom", "living_room", "kids_room", "bathroom"] = Field(
        description="The name of the room."
    )


class TurnOffLight(BaseModel):
    """Turn off a light in a specific room."""

    name: Literal["turn_off_light"] = "turn_off_light"
    arguments: TurnOffLightArgs


class SetThermostatArgs(BaseModel):
    temperature: float = Field(description="Target temperature in Celsius.")
    mode: Literal["cool", "heat", "auto"] = Field(
        "auto", description="Thermostat mode."
    )


class SetThermostat(BaseModel):
    """Set the thermostat temperature and mode."""

    name: Literal["set_thermostat"] = "set_thermostat"
    arguments: SetThermostatArgs


class PlayMusicArgs(BaseModel):
    genre: str = Field(description="Music genre.")
    volume: int = Field(50, description="Volume level (0-100).")


class PlayMusic(BaseModel):
    """Play music in a specific genre."""

    name: Literal["play_music"] = "play_music"
    arguments: PlayMusicArgs


class LockDoorArgs(BaseModel):
    door: Literal["front_door", "back_door", "garage"] = Field(
        description="Which door to lock."
    )


class LockDoor(BaseModel):
    """Lock a specific door."""

    name: Literal["lock_door"] = "lock_door"
    arguments: LockDoorArgs


class CheckWeatherArgs(BaseModel):
    location: str = Field(description="City name.")


class CheckWeather(BaseModel):
    """Check the weather for a specific location."""

    name: Literal["check_weather"] = "check_weather"
    arguments: CheckWeatherArgs


# Define the Union of all possible tool calls
class AnyToolCall(BaseModel):
    function: Union[
        TurnOnLight, TurnOffLight, SetThermostat, PlayMusic, LockDoor, CheckWeather
    ]


class ToolInvocation(BaseModel):
    reasoning: str = Field(
        description="Chain-of-thought reasoning for selecting the specific tool(s) and arguments."
    )
    response: str = Field(
        description="The final response to the user in natural language."
    )
    # We use a list to support multiple actions in one go
    tool_calls: List[AnyToolCall] = Field(
        description="A list of tool calls to execute."
    )


# --- Mock "Tool Registry" (Context) ---
# Even though we have the schema, providing textual context helps the model understand *behavior*.

SMART_HOME_CONTEXT = [
    """
    # Smart Home User Manual
    
    You have a smart home system with the following capabilities:
    
    - **Lights**: Control brightness and color for any room (living_room, kitchen, bedroom, kids_room, bathroom etc.).
    - **Climate**: Set the thermostat temperature and mode (cool/heat/auto).
    - **Music**: Play music by genre throughout the house.
    - **Security**: Lock doors (front/back/garage) and check weather.
    
    Users may ask for things in natural language, often implying multiple steps or inferring parameters.
    """
]


async def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable.")
        return

    # 1. Setup Document Provider
    docs = InMemoryDocumentProvider(SMART_HOME_CONTEXT)

    # 2. Setup Persona Generator
    persona_gen = PersonaGenerator(api_key=api_key)
    await persona_gen.generate_from_documents(docs, n_iterations=0)
    print("Need some sleep")
    await asyncio.sleep(30)

    # 3. Setup Instruction Generator
    # We use ToolCallingInstructionGeneratorCallback to ensure instructions
    # specifically target the available tools.
    tools = [
        TurnOnLight,
        TurnOffLight,
        SetThermostat,
        PlayMusic,
        LockDoor,
        CheckWeather,
    ]

    instruction_callback = ToolCallingInstructionGeneratorCallback(
        api_key=api_key,
        tools=tools,
        documents=docs,
        num_random_contexts=1,
    )

    # 4. Setup Structured Generator
    respondent_prompt = """
    You are the central logic unit for a Smart Home Assistant.
    Map the user's natural language request to the structured tool calls.
    
    - Select the correct tool(s) from the schema.
    - Infer arguments where possible.
    - If no tool matches, return an empty list.
    """

    generator = StructuredGenerator(
        output_schema=ToolInvocation,
        respondent_prompt=respondent_prompt,
        api_key=api_key,
        model_name="gemini-2.0-flash",
        instruction_generator_callback=instruction_callback,
    )

    print("Starting generation of synthetic tool-calling dataset...")
    # Generate 10 samples
    await generator.generate(num_samples=10, max_concurrency=4)
    print("Generation complete. Check the output JSONL file.")


if __name__ == "__main__":
    asyncio.run(main())
