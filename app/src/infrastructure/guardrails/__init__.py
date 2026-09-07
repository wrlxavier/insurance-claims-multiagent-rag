"""Optional runtime defense-in-depth guardrails -- [M5-08 Appendix].

Currently one member: ``local_prompt_injection_classifier``, the real
implementation of ``infrastructure.graph.context.InjectionClassifierPort``.
Off by default (``PROMPT_INJECTION_CLASSIFIER_ENABLED=false``); needs the
optional ``embed`` uv group. See ``docs/PROMPT_INJECTION_CLASSIFIER.md``.
"""
