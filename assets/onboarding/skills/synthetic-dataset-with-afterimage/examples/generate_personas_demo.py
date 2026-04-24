import asyncio
import os
from afterimage import (
    GenerationMonitor,
    JSONLStorage,
    InMemoryDocumentProvider,
    PersonaGenerator,
)


async def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set the GEMINI_API_KEY environment variable!")

    # Sample texts
    texts = [
        "The new quantum computing chip allows for unprecedented calculations, breaking current encryption standards. This technological leap promises to revolutionize various fields, from medicine to finance, by solving problems currently intractable for classical computers. However, it also poses significant challenges to cybersecurity, necessitating the development of new, quantum-resistant cryptographic algorithms to protect sensitive data in the future.",
        "A guide to baking sourdough bread at home, focusing on starter maintenance and baking techniques. This ancient craft involves cultivating a wild yeast starter, which imparts a unique tangy flavor and chewy texture to the bread. Mastering sourdough requires patience and attention to detail, from understanding the nuances of different flour types to perfecting the fermentation process and achieving that coveted crispy crust and open crumb structure.",
        "Analysis of the latest trends in sustainable fashion, including biodegradable fabrics and circular economy models. The fashion industry is a major contributor to environmental pollution, from water consumption and chemical use in production to textile waste. Sustainable fashion aims to mitigate these impacts by promoting ethical sourcing, eco-friendly materials, and production processes that minimize waste and maximize resource efficiency. This shift requires innovation across the supply chain, from design to consumer behavior, to create a truly circular and responsible industry.",
    ]

    # Initialize monitor and storage
    monitor = GenerationMonitor()
    storage = JSONLStorage(documents_path="docs_with_generated_personas.jsonl")

    # Initialize PersonaGenerator
    persona_gen = PersonaGenerator(
        api_key=api_key,
        storage=storage,
        monitor=monitor,
        max_concurrency=2,
    )

    # 1. Generate for a single text
    print("--- Generating for a single document ---")
    single_text = "A deep dive into the philosophy of Stoicism and its applications in modern life. Stoicism, an ancient Greek philosophy, emphasizes virtue, reason, and living in harmony with nature. Its core tenets include the dichotomy of control, the importance of inner tranquility, and the acceptance of what cannot be changed. In today's fast-paced world, Stoic principles offer practical tools for managing stress, cultivating resilience, and finding meaning amidst adversity, guiding individuals towards a more fulfilling and purposeful existence."
    personas = await persona_gen.agenerate_from_text(single_text)
    print(f"Source: {single_text}")
    for p in personas:
        print(f"- {p}")
    print("\n")

    # 2. Generate for a list of documents in batch
    print("--- Generating for multiple documents in batch ---")
    doc_provider = InMemoryDocumentProvider(texts=texts)
    await persona_gen.generate_from_documents(doc_provider, n_iterations=1)
    print(f"Batch generation complete. Personas saved to {storage.documents_path}")
    print(f"Monitoring logs saved to {monitor.log_dir}")

    # Shutdown monitor
    monitor.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
