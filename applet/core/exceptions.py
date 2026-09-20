"""
Edge Error Taxonomy for Pulsar SpaceHack 2026 Applet.
Onboard software must never ungracefully crash on bad sensor inputs.
"""

class AppletBaseException(Exception):
    """Base exception for all satellite applet errors."""
    pass


class InvalidManifestError(AppletBaseException):
    """Raised when the input bundle manifest is missing or invalid."""
    pass



