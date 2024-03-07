from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate

SYSTEM_PROMPT_ARCTIC_SEA = SystemMessagePromptTemplate.from_template(
    "You are a helpful assistant for causal reasoning in the domain of weather and climate in the Arctic Sea."
)

YES_NO_PROMPT = HumanMessagePromptTemplate.from_template(
    "Does {X} directly affect {Y}?\n\n"
    "Let’s work this out in a step by step way to be sure that we have the right answer. Then provide your final answer"
    " within the tags <Answer>yes/no</Answer>."
)

YES_NO_PROMPT_WITH_CONTEXT = HumanMessagePromptTemplate.from_template(
    "Does {X} directly affect {Y}?\n\n"
    "In addition to your own knowledge, you can use the following pieces of context to answer this.\n"
    "Context: \n\"\"\"\n{context}\n\"\"\"\n\n"
    "Let’s work this out in a step by step way to be sure that we have the right answer. Then provide your final answer"
    " within the tags <Answer>yes/no</Answer>."
)

YES_NO_PROMPT_WITH_CONTEXT_NOT_DIRECTLY = HumanMessagePromptTemplate.from_template(
    "Does {X} affect {Y}?\n\n"
    "In addition to your own knowledge, you can use the following pieces of context to answer this.\n"
    "Context: \n\"\"\"\n{context}\n\"\"\"\n\n"
    "Let’s work this out in a step by step way to be sure that we have the right answer. Then provide your final answer"
    " within the tags <Answer>yes/no</Answer>."
)

MULTIPLE_CHOICE_PROMPT_ONLY_ONE_DIRECTIONAL = HumanMessagePromptTemplate.from_template(
    "Which cause-and-effect relationship is more likely?\n"
    "A. {X} is a direct cause of {Y}.\n"
    "B. {Y} is a direct cause of {X}.\n"
    "C. No causal relationship exists.\n\n"
    "Let’s work this out in a step by step way to be sure that we have the right answer. Then provide your final answer"
    " within the tags <Answer>A/B/C</Answer>."
)

MULTIPLE_CHOICE_PROMPT_BIDIRECTIONAL = HumanMessagePromptTemplate.from_template(
    "Which cause-and-effect relationship is more likely?\n"
    "A. {X} is a direct cause of {Y}.\n"
    "B. {Y} is a direct cause of {X}.\n"
    "C. Both cause each other.\n"
    "D. No causal relationship exists.\n\n"
    "Let’s work this out in a step by step way to be sure that we have the right answer. Then provide your final answer"
    " within the tags <Answer>A/B/C/D</Answer>."
)


########################
# New prompts
########################
yes_no_template_with_context_new = """Given the following question and context, answer the question thinking step by step. If the question is not answered with the context, use your own knowledge to answer the question. Provide your final answer at the end within the tags <Answer>yes/no</Answer>.

Remember, solve the question by thinking step by step.

> Question: Is {X} a direct cause of {Y}?
> Context:
>>>
{context}
>>>"""
YES_NO_PROMPT_WITH_CONTEXT_NEW = HumanMessagePromptTemplate.from_template(yes_no_template_with_context_new)


yes_no_template_without_context_new = """Given the following question, answer the question thinking step by step. Provide your final answer at the end within the tags <Answer>yes/no</Answer>.

Remember, solve the question by thinking step by step.

> Question: Is {X} a direct cause of {Y}?
"""
YES_NO_PROMPT_WITHOUT_CONTEXT_NEW = HumanMessagePromptTemplate.from_template(yes_no_template_without_context_new)


########################
# prompts for flare
########################
flare_prompt = """Is {X} a direct cause of {Y}?

Let’s work this out in a step by step way to be sure that we have the right answer. Then provide your final answer within the tags <Answer>yes/no</Answer>.
Let's think step by step."""
FLARE_PROMPT = HumanMessagePromptTemplate.from_template(flare_prompt)



########################
# Context Scorer Prompts
########################


CONTEXT_SCORER_SYSTEM_PROMPT = SystemMessagePromptTemplate.from_template(
    "As ChatGPT, you are required to evaluate the usefulness of provided context information in identifying a "
    "cause-effect relationship between two variables. The user will supply two variables and specify which one is the "
    "Cause and which one is the Effect. Additionally, the user will provide context information. You must assess the "
    "usefulness of this context information in determining the cause-effect relationship on a scale from 0 to 10, "
    "where 0 means 'not useful at all' and 10 means 'extremely useful'."
)

CONTEXT_SCORER_HUMAN_EXAMPLE_1 = HumanMessagePromptTemplate.from_template(
"""Variable 1: Rainfall
Variable 2: Soil Moisture
Cause: Rainfall
Effect: Soil Moisture
Context Information: \"In an agricultural field, when it rains the soil absorbs water. The amount of moisture in the soil depends on the quantity of rainfall.\""""
)

CONTEXT_SCORER_AI_EXAMPLE_1 = AIMessagePromptTemplate.from_template(
"""Based on the provided context information, which explains how rainfall affects soil moisture in an agricultural field, I would rate the usefulness of this information in determining the cause-effect relationship between rainfall and soil moisture as 9 out of 10. The context information is highly useful as it clearly explains the relation between the two variables.
<Score>9</Score>"""
)

CONTEXT_SCORER_HUMAN_EXAMPLE_2 = HumanMessagePromptTemplate.from_template(
"""Variable 1: Studying
Variable 2: Exam Grades
Cause: Studying
Effect: Exam Grades
Context Information: \"Studying is an activity that students do. Exam grades are letters or numbers that represent how well a student performed on an exam.\""""
)

CONTEXT_SCORER_AI_EXAMPLE_2 = AIMessagePromptTemplate.from_template(
"""Based on the provided context information, which does not explain the relationship between studying and exam grades, I would rate the usefulness of this information in determining the cause-effect relationship between studying and exam grades as 2 out of 10. The context information is not very useful because it simply defines the two variables without providing any insight into how they are related.
<Score>2</Score>"""
)

CONTEXT_SCORER_HUMAN_PROMPT = HumanMessagePromptTemplate.from_template(
"""Variable 1: {cause}
Variable 2: {effect}
Cause: {cause}
Effect: {effect}
Context Information: \"{context}\""""
)