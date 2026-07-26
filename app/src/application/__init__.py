"""Application layer.

Use cases and ports (interfaces) that orchestrate domain rules. Depends
only on domain. Has no knowledge of infrastructure details (LangGraph,
database, HTTP) — only the ports that represent them.
"""
