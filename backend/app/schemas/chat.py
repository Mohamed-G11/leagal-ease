from pydantic import BaseModel, ConfigDict, Field

from app.core.schemas import Answer, Citation

MAX_MESSAGE_LENGTH = 4000


class ChatIn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH, description="The user's question.")
    conversation_id: int | None = Field(None, description="Omit to start a new conversation.")


class NumberedCitation(Citation):
    number: int | None = Field(
        None, description="The [n] that refers to this source in the answer text, when it is known."
    )


class ChatOut(Answer):
    """The pipeline's Answer, plus where it was saved."""

    citations: list[NumberedCitation] = Field(default_factory=list)
    conversation_id: int
    message_id: int
