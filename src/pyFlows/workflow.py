import threading
from pathlib import Path

if __package__:
    from .processing_to_detection import run as processing_to_detection
    from .raw_to_processing import run as raw_to_processing
    from .timed_downlink import run as timed_downlink
else:
    from processing_to_detection import run as processing_to_detection
    from raw_to_processing import run as raw_to_processing
    from timed_downlink import run as timed_downlink


def main():
    base_dir = Path(__file__).resolve().parent.parent
    pipelines = (
        ("raw-to-processing", raw_to_processing),
        ("processing-to-detection", processing_to_detection),
        ("timed-downlink", timed_downlink),
    )
    threads = [
        threading.Thread(name=name, target=pipeline, args=(base_dir,))
        for name, pipeline in pipelines
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()