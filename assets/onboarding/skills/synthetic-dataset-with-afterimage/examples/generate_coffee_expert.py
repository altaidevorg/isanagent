import os
import asyncio
from afterimage import (
    ConversationGenerator,
    PersonaInstructionGeneratorCallback,
    PersonaGenerator,
    InMemoryDocumentProvider,
    WithContextRespondentPromptModifier,
)

# Get API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Set the GEMINI_API_KEY environment variable!")

# Define the respondent prompt
respondent_prompt = """You are a knowledgeable coffee expert.
Your task is to answer questions about coffee brewing, beans, and history.
Provide detailed, accurate, and enthusiastic responses.
"""

# Prepare contextual documents
# In a real scenario, you might load these from files using DirectoryDocumentProvider or JSONLDocumentProvider
texts = [
    """
    Pour-over coffee is a method of brewing coffee by pouring hot water over ground coffee beans through a filter. 
    The water drains through the coffee and filter into a carafe or mug. 
    Pour-over brewing allows for intricate flavor extraction. 
    Common devices include the V60, Chemex, and Kalita Wave.
    Key variables are grind size, water temperature, and pouring technique.
    """,
    """
    Espresso is a concentrated coffee beverage brewed by forcing hot water under high pressure (9-10 bars) through finely-ground coffee beans.
    It is the base for many drinks like lattes, cappuccinos, and macchiatos.
    A good espresso has a layer of crema on top.
    """,
]
documents = InMemoryDocumentProvider(texts)


async def main():
    print("1. Generating Personas from documents...")
    # Initialize PersonaGenerator
    persona_gen = PersonaGenerator(api_key=api_key)

    # Generate personas for the documents
    # This will populate the .personas attribute of each Document in the provider
    await persona_gen.generate_from_documents(documents)

    # Inspect generated personas
    for i, doc in enumerate(documents.get_all()):
        print(f"\nDocument {i + 1} Personas:")
        for p_entry in doc.personas:
            for p in p_entry.descriptions:
                print(f"- {p}")

    print("\n2. Setting up Conversation Generator...")

    # Set up the persona instruction generator callback
    # This callback will select a random persona from the document's personas
    # and instruct the LLM to roleplay that persona when asking questions.
    instruction_generator_callback = PersonaInstructionGeneratorCallback(
        api_key=api_key,
        documents=documents,
        num_random_contexts=1,
        n_instructions=2,
    )

    # Set up the respondent prompt modifier (optional, but good for RAG/Context usage)
    respondent_prompt_modifier = WithContextRespondentPromptModifier()

    # Initialize the ConversationGenerator
    conv_gen = ConversationGenerator(
        respondent_prompt=respondent_prompt,
        api_key=api_key,
        model_name="gemini-2.0-flash",
        instruction_generator_callback=instruction_generator_callback,
        respondent_prompt_modifier=respondent_prompt_modifier,
    )

    print("3. Generating Conversations...")
    # Generate conversations
    # The 'persona' field in the output will indicate which persona was used.
    await conv_gen.generate(
        num_dialogs=4,
        max_turns=1,
        max_concurrency=2,
    )

    print("\n4. Load generated conversations...")
    conversations = conv_gen.load_conversations()

    print(f"\nGenerated {len(conversations)} conversations.")

    # Display a sample
    if conversations:
        conv = conversations[0]
        print("\nSample Conversation:")
        print(f"Persona: {conv.persona}")
        print(f"Context: {conv.instruction_context[:100]}...")
        for turn in conv.conversations:
            print(f"{turn.role}: {turn.content[:100]} ...")


if __name__ == "__main__":
    asyncio.run(main())
