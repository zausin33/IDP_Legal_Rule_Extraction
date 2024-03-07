from enum import Enum

from pydantic import BaseModel


def multi_line_user_input(message) -> str:
    print(message, "When done, type 'END' on a new line and press enter.")
    user_edit_lines = []
    while True:
        line = input()
        if line.strip().upper() == 'END':
            break
        user_edit_lines.append(line)

    user_edit = '\n'.join(user_edit_lines)
    return user_edit


class ChatModels(Enum):
    GPT_35_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_4_PREVIEW = "gpt-4-1106-preview"


class CompletionModels(Enum):
    TEXT_DAVINCI_002 = "text-davinci-002"
    CODE_DAVINCI_002 = "code-davinci-002"
    TEXT_DAVINCI_003 = "text-davinci-003"
    GPT_35_TURBO_INSTRUCT = "gpt-3.5-turbo-instruct"


class OpenAIModelName(Enum):
    Chat = ChatModels
    Completion = CompletionModels


class Config(BaseModel):
    open_ai_model_name: str
    use_examples_for_translation: bool = False
    with_human_editing: bool = False
    number_retries_different_errors: int = 5
    number_retries_same_error: int = 2
    extract_rules_first: bool = True,
    use_goz_commentary:bool = False

