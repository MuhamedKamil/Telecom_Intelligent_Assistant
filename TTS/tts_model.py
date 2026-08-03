import time
from pathlib import Path
from silma_tts.api import SilmaTTS
from typing import Optional, Dict, List, Any, Union


class TTS:
    """
    Wrapper around SilmaTTS for voice cloning.
    """

    def __init__(
        self,
   
        TTS_config: Optional[Dict] 

    ):
      
        self.model = SilmaTTS()
        self.reference_audio = TTS_config["reference_audio"]
        self.reference_text  = TTS_config["reference_text"]

    def generate(
        self,
        text: str,
        output_file: str = "generated_audio.wav",
        speed: float     = 1.0,
        seed: int        = None,
    ):
        """
        Generate speech from text.

        Args:
            text: Text to synthesize.
            output_file: Output wav filename.
            speed: Speech speed.
            seed: Random seed.

        Returns:
            dict containing waveform, sample rate, spectrogram,
            output path, and inference time.
        """
        start = time.time()

        wav, sr, spec = self.model.infer(
            ref_file=self.reference_audio,
            ref_text=self.reference_text,
            gen_text=text.strip(),
            file_wave=output_file,
            seed=seed,
            speed=speed,
        )

        elapsed = time.time() - start

        return {
            "wav": wav,
            "sample_rate": sr,
            "spectrogram": spec,
            "output_file": output_file,
            "elapsed_time": elapsed,
        }


    