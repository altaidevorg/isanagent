import os
import asyncio
from datetime import timedelta


from afterimage import (
    AsyncConversationGenerator,
    ContextualInstructionGeneratorCallback,
    GenerationMonitor,
    JSONLDocumentProvider,
    WithContextRespondentPromptModifier,
)


# Get API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Set the GEMINI_API_KEY environment variable!")


# setup monitoring
def simple_alert_handler(alert):
    print(f"alert - {alert.name} - {alert.message}")


monitor = GenerationMonitor(
    log_dir="logs",
    alert_handlers=[simple_alert_handler],
    metrics_interval=60,
)

# Define the respondent prompt
respondent_prompt = """Sen, dünya çapında tanınmış başarılı ve deneyimli bir mali müşavirsin.
Görevin, diğer mali müşavirler ve sıradan vergi mükellefleri tarafından sorulan sorulara Türk vergi hukuku kapsamında ve Gelir İdaresi Başkanlığı (GİB) özelgelerine dayanarak cevap vermek.
Her soruya dikkatle cevap ver ve yanıtlarının gerçek bir mali müşavirden alınıyormuş gibi güvenilir ve detaylı olmasını sağla.
Cevaplarına her zaman GİB özelgelerine kararlarına dayandır ve uygun şekilde atıflarda bulun.
Hukuki ve ticari kavramları doğru şekilde kullanmaya özen göster ama herkesin anlayabileceği bir şekilde sadeleştirerek açıkla.
Onlara asla 'gerçek bir mali müşavire danışmalarını' ya da 'hukuki yardıma başvurmalarını' tavsiye etme çünkü sen, tüm vatandaşların ve mali müşavirlerin güvendiği, saygıdeğer bir mali müşavirsin.
"""

# Prepare contextual documents
documents = JSONLDocumentProvider(
    "../scraping/gib-ozelge.jsonl", content_key="markdown"
)

# Set up the instruction generator callback
instruction_generator_callback = ContextualInstructionGeneratorCallback(
    api_key=api_key,
    documents=documents,
    num_random_contexts=1,  # Experiment with different values
    n_instructions=3,
)

# Set up the respondent prompt modifier
respondent_prompt_modifier = WithContextRespondentPromptModifier()


async def main():
    # Initialize the ConversationGenerator
    conv_gen = AsyncConversationGenerator(
        respondent_prompt=respondent_prompt,
        api_key=api_key,
        model_name="gemini-2.0-flash",
        monitor=monitor,
        instruction_generator_callback=instruction_generator_callback,
        respondent_prompt_modifier=respondent_prompt_modifier,
    )

    # let the correspondent prompt be automatically generated

    # Print the auto-generated correspondent prompt
    # note: normally, you do not need to call `initialize()`` method here manually,,
    # and it will be automatically called in the `generate()` method
    # we call it here just to trigger the creation of correspondent prompt and print it
    # before entering the generation loop.
    await conv_gen.ainitialize(instruction_generator_callback)

    # Print the auto-generated correspondent prompt
    print("Generated Correspondent Prompt:")
    print(conv_gen.correspondent_prompt)

    # Generate conversations
    await conv_gen.generate(
        num_dialogs=20,  # Total dialogs to generate
        max_turns=1,  # Max turns per conversation
        max_concurrency=4,
    )

    # Get metrics for the last one hour
    generation_time = monitor.get_metrics("generation_time", window=timedelta(hours=1))
    if generation_time:
        print(f"Avg. generation time: {generation_time.get('mean', 0):.2f} secs")

    # Generate visualizations
    figures = monitor.visualize_metrics(save_dir="plots")
    print("You have these figures visualized for you", figures.keys())
    print("You can show them with figures[key].show() or save them to a file.")

    # Optional: Export metrics data
    # monitor.export_metrics(
    # "monitoring_metrics_export.json", format="json", window=timedelta(minutes=1)
    # )

    # graceful shutdown
    monitor.shutdown()


# Generate conversations
if __name__ == "__main__":
    asyncio.run(main())
