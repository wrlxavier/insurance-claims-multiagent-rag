"""Use cases that orchestrate domain rules through ports.

M5-02: ``submit_claim``, ``get_assessment``, ``submit_human_decision`` and
``list_assessments`` -- the assessment lifecycle. These are frozen-dataclass
interactors (ports injected as fields, one ``__call__``), unlike the earlier
pure-function pipeline use cases, because they carry dependencies.
"""
