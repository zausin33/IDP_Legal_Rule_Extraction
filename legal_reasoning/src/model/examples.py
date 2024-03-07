translation_examples = [
    {
        "input": """Der Zahnarzt kann Gebühren nur für selbstständige zahnärztliche Leistungen berechnen, 
die er selbst erbracht hat oder die unter seiner Aufsicht nach fachlicher Weisung erbracht wurden (eigene Leistungen).
Für eine Leistung, die Bestandteil oder eine besondere Ausführung einer anderen Leistung nach dem Gebührenverzeichnis ist, kann der Zahnarzt eine Gebühr nicht berechnen, wenn er für die andere Leistung eine Gebühr berechnet. 
Dies gilt auch für die zur Erbringung der im Gebührenverzeichnis aufgeführten operativen Leistungen methodisch notwendigen operativen Einzelschritte. Eine Leistung ist methodisch notwendiger Bestandteil einer anderen Leistung, wenn sie inhaltlich von der Leistungsbeschreibung der anderen Leistung(Zielleistung) umfasst und auch in deren Bewertung berücksichtigt worden ist.""",
        "output": """```prolog        
performed_or_supervised(Dentist, Service) :- performed(Dentist, Service); supervised(Dentist, Service).
part_or_special_performance(Service, OtherService) :- part_of(Service, OtherService); special_performance(Service, OtherService).
surgical_step(Service, SurgicalService) :- necessary_for(Service, SurgicalService).
necessary_component(Service, OtherService) :- covered_by(Service, OtherService), taken_into_account(Service, OtherService).

is_invoice_correct(Dentist, Invoice) :-
    forall(member(Service, Invoice),
           (performed_or_supervised(Dentist, Service),
            \+ (charged(Dentist, OtherService),
                (part_or_special_performance(Service, OtherService);
                 surgical_step(Service, OtherService);
                 necessary_component(Service, OtherService))))).
```"""
    },
    {
        "input": """Mit den Gebühren sind die Praxiskosten einschließlich der Kosten für Füllungsmaterial, für den Sprechstundenbedarf, für die Anwendung von Instrumenten und Apparaten sowie für Lagerhaltung abgegolten, soweit nicht im Gebührenverzeichnis etwas anderes bestimmt ist.
Hat der Zahnarzt zahnärztliche Leistungen unter Inanspruchnahme Dritter, die nach dieser Verordnung selbst nicht liquidationsberechtigt sind, erbracht, so sind die hierdurch entstandenen Kosten ebenfalls mit der Gebühr abgegolten.
Kosten, die nach Absatz 3 mit den Gebühren abgegolten sind, dürfen nicht gesondert berechnet werden. Eine Abtretung des Vergütungsanspruchs in Höhe solcher Kosten ist gegenüber dem Zahlungspflichtigen unwirksam.
Sollen Leistungen durch Dritte erbracht werden, die diese dem Zahlungspflichtigen unmittelbar berechnen, so hat der Zahnarzt ihn darüber zu unterrichten.""",
    "output": """```prolog
covered_by_fee(Fee, ThirdPartyCost) :- provided_by_third_party(ThirdPartyCost), not(entitled_to_payment(ThirdPartyCost)).
cannot_charge_separately(Dentist, Cost) :- covered_by_fee(Fee, Cost).
invalid_claim(Dentist, Cost) :- covered_by_fee(Fee, Cost).

must_inform(Dentist, Payer, ThirdPartyService) :-
    rendered_by_third_party(ThirdPartyService),
    charged_directly(ThirdPartyService, Payer),
    informed(Dentist, Payer, ThirdPartyService).

is_invoice_correct(Dentist, Invoice, Payer) :-
    forall(member(Item, Invoice),
           (covered_by_fee(Fee, Item),
            \+ cannot_charge_separately(Dentist, Item),
            \+ invalid_claim(Dentist, Item),
            (rendered_by_third_party(Item), charged_directly(Item, Payer) -> must_inform(Dentist, Payer, Item); true)
           )
          ).
```"""
    },
]

