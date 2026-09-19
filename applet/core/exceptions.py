"""
Edge Error Taxonomy for Pulsar SpaceHack 2026 Applet.
Onboard software must never ungracefully crash on bad sensor inputs.
"""

class AppletBaseException(Exception):
    """Base exception for all satellite applet errors."""
    pass


class CorruptedDataError(AppletBaseException):
    """Raised when an image or metadata file is malformed or unreadable."""
    pass


class InvalidManifestError(AppletBaseException):
    """Raised when the input bundle manifest is missing or invalid."""
    pass


class ResourceLimitExceededError(AppletBaseException):
    """Raised when execution crosses allowed RAM, CPU, or pass duration ceilings."""
    pass


class BandwidthBudgetExceededError(AppletBaseException):
    """Raised when the generated downlink bundle exceeds orbital pass bandwidth."""
    pass


class PipelineProcessingError(AppletBaseException):
    """Raised when an individual processing stage fails."""
    pass
