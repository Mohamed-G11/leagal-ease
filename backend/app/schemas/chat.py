from pydantic import BaseModel, ConfigDict, Field

from app.core.schemas import Answer

MAX_MESSAGE_LENGTH = 4000


class ChatIn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH, description="The user's question.")
    conversation_id: int | None = Field(None, description="Omit to start a new conversation.")


class ChatOut(Answer):
    """The pipeline's Answer, plus where it was saved."""

    conversation_id: int
    message_id: int
