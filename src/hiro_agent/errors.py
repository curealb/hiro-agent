"""Errors with deliberately credential-free, user-facing messages."""


class HiroError(Exception):
    pass


class ConfigurationError(HiroError):
    pass


class PersistenceError(HiroError):
    pass


class ModelError(HiroError):
    pass


class ContextLimitError(ModelError):
    pass


class BudgetExceeded(HiroError):
    pass


class RunCancelled(HiroError):
    pass


class ToolError(HiroError):
    pass
