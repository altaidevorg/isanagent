import asyncio
import os
from afterimage import (
    ConversationGenerator,
    ConversationJudge,
    LLMFactory
)
from afterimage.preference.types import PreferenceConfig

async def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY")
        return

    from afterimage import PersonaInstructionGeneratorCallback, InMemoryDocumentProvider
    docs = InMemoryDocumentProvider(["Common everyday tasks."])
    instruction_cb = PersonaInstructionGeneratorCallback(
        api_key=api_key,
        documents=docs,
        model_name="gemini-2.5-flash",
        num_random_contexts=1,
        n_instructions=1,
    )

    # Base generator for the initial conversation
    gen = ConversationGenerator(
        respondent_prompt="You are a helpful assistant.",
        api_key=api_key,
        model_name="gemini-2.5-flash",
        instruction_generator_callback=instruction_cb,
    )

    # Custom judge prompt to enforce tone alignment
    judge_prompt = """
    You are evaluating two responses from an AI assistant.
    The ideal response should be extremely concise, polite, and direct.
    It should avoid unnecessary pleasantries or overly verbose explanations.
    Choose the response that best fits this concise and polite persona.
    """

    from afterimage import SmartKeyPool, LLMFactory, EmbeddingProviderFactory
    pool = SmartKeyPool.from_single_key(api_key)
    llm = LLMFactory.create(
        provider="gemini", model_name="gemini-2.5-flash", api_key=pool
    )
    embedding = EmbeddingProviderFactory.create(
        {"type": "gemini", "model": "gemini-embedding-001"},
        key_pool=pool,
    )

    # Create the judge
    judge = ConversationJudge(
        llm=llm,
        embedding_provider=embedding,
    )

    pref = gen.to_preference_generator(
        judge=judge,
        config=PreferenceConfig(num_pairs=5, output_path="./tone_dpo.jsonl"),
    )
    
    pairs, analytics = await pref.generate()
    pref.save_pairs(pairs, analytics)
    await judge.aclose()

if __name__ == "__main__":
    asyncio.run(main())
