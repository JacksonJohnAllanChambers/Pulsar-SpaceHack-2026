import numpy as np

from applet.config import AppletConfig
from applet.pipelines.arctic_classifier import ArcticClassifier


def _detection(wake_length_m=0.0):
    return {"wake_length_m": wake_length_m}


def test_bright_smooth_no_wake_chip_is_iceberg_candidate():
    config = AppletConfig()
    config.arctic.enabled = True
    detection = _detection()
    chip = np.full((64, 64, 4), 0.02, dtype=np.float32)
    chip[16:48, 16:48] = 0.42

    ArcticClassifier(config).classify(detection, chip, ship_probability=0.05)

    assert detection["arctic_classification"] == "ICEBERG"
    assert detection["iceberg_probability"] > detection["ship_probability"]


def test_long_wake_and_ship_probability_prefer_ship():
    config = AppletConfig()
    config.arctic.enabled = True
    detection = _detection(wake_length_m=240.0)
    chip = np.full((64, 64, 4), 0.02, dtype=np.float32)
    chip[16:48, 16:48] = 0.30

    ArcticClassifier(config).classify(detection, chip, ship_probability=0.95)

    assert detection["arctic_classification"] == "SHIP"


def test_close_evidence_stays_uncertain():
    config = AppletConfig()
    config.arctic.enabled = True
    detection = _detection()
    chip = np.full((64, 64, 4), 0.02, dtype=np.float32)
    chip[16:48, 16:48] = 0.22

    ArcticClassifier(config).classify(detection, chip, ship_probability=0.50)

    assert detection["arctic_classification"] == "UNCERTAIN"