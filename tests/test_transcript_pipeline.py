import asyncio

from app.config.settings import load_settings
from app.services.transcript_pipeline import TranscriptPipeline
from app.transcription.factory import create_transcription_provider


def test_stub_provider_emits_segments():
    """The stub provider should drive segments through the pipeline to listeners."""
    settings = load_settings(
        bootstrap=False,
        overrides={
            "transcription": {"provider": "faster_whisper", "chunk_seconds": 1},
        },
    )
    provider = create_transcription_provider(settings)
    pipeline = TranscriptPipeline(provider)

    captured: list[str] = []
    pipeline.add_listener(lambda segment: captured.append(segment.text))

    async def _drive() -> None:
        task = pipeline.start()
        # Let the scripted phrases flow through.
        await asyncio.sleep(1.5)
        await pipeline.stop()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_drive())

    assert captured, "Pipeline should have produced at least one segment"
    assert all(isinstance(s, str) and s for s in captured)
