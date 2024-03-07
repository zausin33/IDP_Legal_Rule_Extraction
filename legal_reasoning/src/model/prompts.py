SYSTEM_PROMPT = """Your are an expert in the logical programming language Prolog and of the german GOZ (Gebührenordnung für Zahnärtzte). The user will give you tasks where you have to write prolog code related to the GOZ. 
Please wrapp code in the following way: 
```prolog 
<your code>
```"""

PROLOG_TRANSLATION_PROMPT = """Given this rules from the german GOZ:
---
{text}

And this invoice schema:
```prolog
invoice(
    % A Service is defined trough its Service Number (Service), the date / session when it was rendered, its multiplier, its point score, the teeth it is applied to, ...
    [service(Service, date(Year, Month, Day), Multiplier, PointScore, Charge, Description, Justification, [tooth(Area, Postion), ...]), ... ],
    % Billed Material Costs are defined trough the Section of the GOZ which allows its billing, the date / session when it was rendered, its multiplier, ...
    [material_cost(GOZ_Section, date(Year, Month, Day), Multiplier, Count, Charge, Description, Justification, [tooth(Area, Postion), ...]), ... ],
    invoice_date(date(Year, Month, Day)),
    invoice_amount(InvoiceAmount)
).
% Charge = (round(PointValue in Euros * Multiplier * PointScore * 100)) / 100
% A Tooth on which a Service is applied is uniquely defined by the combination of its area and position in this area: tooth(Area, Postion) nummerated after the FDI-Zahnschema
% Area = 1 -> Right upper jaw
% Area = 2 -> Left upper jaw
% Area = 3 -> Left lower jaw
% Area = 4 -> Right lower jaw
% Position is exact position of this tooth in its area (from 1 to 8)
% Example invoice
invoice(
    [
        service(11111, date(2023, 10, 01), 1.0, 100, 5.62, 'Teeth Cleaning', '', [tooth(1,7)]),
        service(11111, date(2023, 10, 01), 1.0, 100, 5.62, 'Teeth Cleaning', '', [tooth(1,6)]),
        service(987654, date(2023, 10, 01), 3.5, 200, 39.37, 'Tooth Filling', 'Very time consuming', [tooth(1,7), tooth(1,6)]),
        service(123456, date(2023, 10, 02), 2.3, 150, 19.40, 'Consultation', '', [])
    ],
    [
        material_cost('§ 4 (1)', date(2023, 10, 02), 1.0, 1, 40.00, 'Fill Material', 'Auslagen', [tooth(1,7), tooth(1,6)])
    ]
    invoice_date(date(2023, 10, 04),
    invoice_amount(110.01)
).
```
The {rule_count} rules indicate conditions and restrictions stated in the GOZ that should be adhered to when creating an invoice.
We want to mark invoices as invalid if they do not adhere to those rules.
Which of those rules can you translate into prolog code given following restriction:
Only translate the rules as they are, do not include any examples.
Be as accurate as possible.
You must be able to exactly define every new predicate based on the given ones.
Do not make any assumptions which are not given.
Translate the rules where a translation given the restrictions is possible.
If you can not translate the ful rule, translate as much as possible, but do not invent something. Maybe you can translate a part of the rule.
Keep the code as simple as possible.
To determine if an invoice is correct, use the predicate 
```prolog
is_invoice_invalid(Invoice) : -
    Invoice = invoice(Services, Material_Costs, invoice_date(date(Year, Month, Day), invoice_amount(InvoiceAmount)),
    % The corresponding rule (only one rule for each predicate)
    % Please double check that the prolog code you write in here is semantically equivalent to the rule in the text.
    % Keep the code as simple as possible.
    print(...), % When rule is not fulfilled and therefore the invoice is invalid, print the reason for the invalidity of the invoice
    true. % Then the invoice is invalid.
```
Use this predicate to check each rule individually. Before you translate a rule, think about it.
Expected format:
---
Rule 1:
Thoughts: ...
Translation:
```prolog
% code here (is_invoice_invalid/1 predicate with potential helper predicates for this rule)
```
---
Rule 2:
Thoughts: ...
Translation:
```prolog
% code here (is_invoice_invalid/1 predicate with potential helper predicates for this rule)
```
---
... 

When you can not translate leaf the code section empty like this:
---
Rule X:
Thoughts: ...
Translation:
```prolog
```
---

So you should have a block for each of the {rule_count} given rules at the end.
For translated rules include print/1 statements, so that the user can see the reason for the invalidity of the invoice.
When you calculate with currencies, always round the result to two digits with round(X*100)/100 and always calculate in Euros (even after summation).
I only want to have exactly defined rules.
First, think step by step how you would translate a rule, then translate it. Explain your thoughts step by step."""


PROLOG_TRANSLATION_PROMPT_WITH_COMMENTARY = PROLOG_TRANSLATION_PROMPT + """

In the following I will give you some more context text. This text is only intended to help you understand the context in which you are translating the rule.
Context:
>>>
{commentary}
>>>"""


REFERENCE_PROMPT = """

The rules references some already translated section.
As references for you, here are the translations of the referenced sections:
{references}
"""

PROLOG_TRANSLATION_PROMPT_WITH_REFERENCE = PROLOG_TRANSLATION_PROMPT + REFERENCE_PROMPT

PROLOG_TRANSLATION_PROMPT_WITH_REFERENCE_AND_COMMENTARY = PROLOG_TRANSLATION_PROMPT_WITH_COMMENTARY + REFERENCE_PROMPT

CONSTRUCT_ONE_ROOT_PROMPT = """Given this translation of a legal text into prolog code:
```prolog
{code}
```

Please construct a new rule is_whole_invoice_correct with the predicates
{predicates}

New Rule:"""
