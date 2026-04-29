from rag2.config import Settings


def test_default_openai_models_are_current() -> None:
    settings = Settings()
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.chat_model == "gpt-4o-mini"
