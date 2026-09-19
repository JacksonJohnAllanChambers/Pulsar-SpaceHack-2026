"""
Pipeline Registry and Dispatcher for Hackathon Tracks.
Enables switching between Track 1 (Tactical ISR), Track 3 (Prioritization), and Track 4 (Screening).
"""

from typing import List
from applet.config import AppletConfig
from applet.core.base import BasePipeline
from applet.pipelines.quality_screener import ImageQualityScreener
from applet.pipelines.vessel_detector import VesselDetector
from applet.pipelines.chip_verifier import ChipVerifier
from applet.pipelines.ais_correlator import AISKinematicCorrelator


class PipelineDispatcher:
    """
    Constructs and sequences pipeline stages based on requested track.
    """

    @staticmethod
    def get_pipeline(track: str, config: AppletConfig) -> List[BasePipeline]:
        track = track.lower()

        if track in ["track4", "screening", "quality"]:
            return [ImageQualityScreener(config)]

        # Track 1 (tactical ISR) and Track 3 (prioritisation) share the full cascade:
        # the downlink packager ranks by the priorities the correlator assigns.
        return [
            ImageQualityScreener(config),
            VesselDetector(config),
            ChipVerifier(config),
            AISKinematicCorrelator(config),
        ]
