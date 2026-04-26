import asyncio

from app.config.settings import TranscriptionSettings
from app.services.transcript_pipeline import TranscriptPipeline
from app.transcription.faster_whisper_provider import FasterWhisperProvider


def test_stub_provider_emits_segments():
    """The stub provider should drive segments through the pipeline to listeners."""
    settings = TranscriptionSettings(provider="faster_whisper", emit_interval_ms=50)
    provider = FasterWhisperProvider(settings)
    pipeline = TranscriptPipeline(provider)

    captured: list[str] = []
    pipeline.add_listener(lambda segment: captured.append(segment.text))

    async def _drive() -> None:
        task = pipeline.start()
        # Let the scripted phrases flow through.
        await asyncio.sleep(0.6)
        await pipeline.stop()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_drive())

    assert captured, "Pipeline should have produced at least one segment"
    assert all(isinstance(s, str) and s for s in captured)
