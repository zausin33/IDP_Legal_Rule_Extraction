from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from src.langchain_modules import prompts

from src.langchain_modules.output_parser import ScoreOutputParser

score_output_parser = ScoreOutputParser()
chat = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")
chat_prompt = ChatPromptTemplate.from_messages([
    prompts.CONTEXT_SCORER_SYSTEM_PROMPT,
    prompts.CONTEXT_SCORER_HUMAN_EXAMPLE_1,
    prompts.CONTEXT_SCORER_AI_EXAMPLE_1,
    prompts.CONTEXT_SCORER_HUMAN_EXAMPLE_2,
    prompts.CONTEXT_SCORER_AI_EXAMPLE_2,
    prompts.CONTEXT_SCORER_HUMAN_PROMPT,
])
chain = LLMChain(llm=chat, prompt=chat_prompt, verbose=True)


def score_context(cause: str, effect: str, context: str) -> int:
    result = chain.run(cause=cause, effect=effect, context=context)
    print(f"Result: {result}")
    score = score_output_parser.parse(result)
    return score

