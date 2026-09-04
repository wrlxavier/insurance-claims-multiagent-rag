"""Exceptions shared across graph nodes ([M5-08])."""


class SchemaValidationError(RuntimeError):
    """A structured LLM output failed schema validation and was rejected.

    Raised instead of coercing a malformed response into something the node
    can proceed with -- the M5-08 rule that a structured output is validated
    against its schema and rejected, never patched, on failure.
    """
