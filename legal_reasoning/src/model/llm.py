from typing import List, Any

import langchain
from langchain import OpenAI
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationTokenBufferMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, \
    SystemMessagePromptTemplate
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.schema.language_model import BaseLanguageModel

from legal_reasoning.src.code_execution import CodeExecutor, PrologParser, PrologRetryParser, PrologMultipleRuleParser
from legal_reasoning.src.model import prompts
from legal_reasoning.src.model.correction_module import CorrectionModule, CompilationCorrectionModule, CorrectionModuleChain
from legal_reasoning.src.model.llm_pipeline import Pipeline, PrepareAndExecuteTranslationChain, ParseAndCorrectCode, \
    PrologRegexCorrection, \
    GozTextSectionHandlerDependencyTree, Split, GozServiceSectionHandler, GozServiceRulesHandler
from legal_reasoning.src.utils import Config, ChatModels

langchain.verbose = True

system_prompt = SystemMessagePromptTemplate.from_template(prompts.SYSTEM_PROMPT)

prolog_translation_prompt = PromptTemplate.from_template(prompts.PROLOG_TRANSLATION_PROMPT)


class LLMPrompter:
    config: Config
    llm: BaseLanguageModel
    translation_chain: LLMChain
    memory: BaseChatMemory
    output_parser: BaseOutputParser
    code_executor: CodeExecutor
    llm_correction_modules: CorrectionModuleChain
    human_correction_modules: List[CorrectionModule]

    def __init__(
            self,
            config: Config,
            code_executor: CodeExecutor,
            ):
        self.config = config

        if config.open_ai_model_name in [x.value for x in ChatModels]:
            self.llm = ChatOpenAI(temperature=0, model_name=config.open_ai_model_name, model_kwargs={"top_p": 1, "seed": 2023})
            prompt = ChatPromptTemplate(
                messages=[
                    MessagesPlaceholder(variable_name="chat_history"),  # Placeholder for chat history
                    system_prompt,
                    HumanMessagePromptTemplate.from_template("{question}"),  # Placeholder for input
                ]
            )
        else:
            self.llm = OpenAI(temperature=0, model_name=config.open_ai_model_name, max_tokens=2000)
            prompt = PromptTemplate.from_template("{question}")
            self.nl_rule_translation_prompt = ChatPromptTemplate.from_template("")

        self.memory = ConversationTokenBufferMemory(memory_key="chat_history", return_messages=True, llm=self.llm,
                                                    max_token_limit=2000)
        self.code_executor = code_executor

        self.translation_chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=True,
            memory=self.memory,
        )

        self.output_parser = PrologRetryParser.from_llm(
            chain=self.translation_chain,
            parser=PrologParser(),
        )

        self.llm_correction_modules = CorrectionModuleChain(
            config,
            [CompilationCorrectionModule(self.config, self.translation_chain, self.output_parser, code_executor),]
             # LoopCorrectionModule(self.config, self.translation_chain, self.output_parser)]
        )
        self.human_correction_modules = []

        self.translation_prompt = lambda text, user_edit: prolog_translation_prompt.format(
            text=text,
            rule_count=len(text.split("\n---\n")) - 1,
        )

    def prompt_llm_for_translation(self, context) -> Any:
        inner_pipeline = Pipeline([
            PrepareAndExecuteTranslationChain(self.config, self.translation_chain, self.memory),
            # SplitLLMResponseIntoCodeAndQuery(),
            Split(parser=PrologRetryParser.from_llm(chain=self.translation_chain, parser=PrologMultipleRuleParser())),
            PrologRegexCorrection(),
            ParseAndCorrectCode(self.output_parser, self.llm_correction_modules),
        ])

        pipeline = Pipeline([GozTextSectionHandlerDependencyTree(self.config, text_section_pipeline=inner_pipeline)])
        context = pipeline.run(context)

        pipeline = Pipeline([GozServiceSectionHandler(self.config, text_section_pipeline=inner_pipeline)])
        context = pipeline.run(context)

        pipeline = Pipeline([GozServiceRulesHandler(self.config, text_section_pipeline=inner_pipeline)])
        context = pipeline.run(context)

        return context["text_df"], context["service_df"], context["section_df"]


