import asyncio
import os
from afterimage import (
    ConversationGenerator,
    PersonaGenerator,
    PersonaInstructionGeneratorCallback,
    InMemoryDocumentProvider,
)

SYMPTOM_DOCS = [
    "Common cold symptoms include runny nose, sore throat, and mild cough. Usually resolves in 7-10 days.",
    "Influenza (flu) presents with sudden high fever, muscle aches, fatigue, and dry cough.",
    "COVID-19 symptoms can include loss of taste/smell, shortness of breath, fever, and cough.",
    "Appendicitis typically starts with dull pain near the navel that shifts to the lower right abdomen, accompanied by nausea."
]

async def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY")
        return

    docs = InMemoryDocumentProvider(SYMPTOM_DOCS)
    
    # Generate patient personas based on the symptom documents
    persona_gen = PersonaGenerator(api_key=api_key, model_name="gemini-2.5-flash")
    await persona_gen.generate_from_documents(docs)

    instruction_cb = PersonaInstructionGeneratorCallback(
        api_key=api_key,
        documents=docs,
        model_name="gemini-2.5-flash",
        num_random_contexts=1,
        n_instructions=2,
    )

    respondent_prompt = (
        "You are a helpful AI medical triage assistant. "
        "Ask clarifying questions one at a time to understand the user's symptoms. "
        "Do not provide a final diagnosis immediately. "
        "After gathering enough information, advise whether they should rest at home, see a doctor, or go to the ER."
    )

    gen = ConversationGenerator(
        respondent_prompt=respondent_prompt,
        api_key=api_key,
        model_name="gemini-2.5-flash",
        instruction_generator_callback=instruction_cb,
    )
    
    # max_turns=3 allows for back-and-forth questioning
    await gen.generate(num_dialogs=5, max_turns=3, max_concurrency=2)

if __name__ == "__main__":
    asyncio.run(main())
