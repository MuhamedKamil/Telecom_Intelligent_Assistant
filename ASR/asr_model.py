from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import load_audio


class ASR:
    def __init__(
        self,
        model_name: str    = "base",
        device: str        = "cuda",
        compute_type: str  = "int8",
       

    ):
        self.processor = AutoProcessor.from_pretrained("namaa-space/cohere-transcribe-arabic-07-2026-int4"),
        self.asr_model = CohereAsrForConditionalGeneration.from_pretrained("namaa-space/cohere-transcribe-arabic-07-2026-int4", device_map="auto")

 

        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available. Falling back to CPU.")
            device = "cpu"

        self.device = device
        self.compute_type = compute_type

        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )

        print(f"Loaded Faster-Whisper model: {model_name}")
        print(f"Device: {device}")
        print(f"Compute Type: {compute_type}")

        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    def detect_language(self,audio_path):
        audio = decode_audio(audio_path,sampling_rate=self.model.feature_extractor.sampling_rate,)
        language, language_probability, all_language_probs = self.model.detect_language(audio)
        lang_probs = dict(all_language_probs)
        en_prob = lang_probs.get("en", 0.0)
        ar_prob = lang_probs.get("ar", 0.0)
        if ar_prob >= en_prob:
            selected_lang = "ar"
            selected_prob = ar_prob
        else:
            selected_lang = "en"
            selected_prob = en_prob
       
        return selected_lang

    def transcribe(
        self,
        audio_path: str,
        beam_size: int = 5,
    ):
        language = self.detect_language(audio_path)

        inputs = self.processor(audio_path, sampling_rate=16000, language="en", return_tensors="pt").to(
            device=self.asr_model.device, 
            dtype=self.asr_model.dtype)

        outputs = self.asr_model.generate(**inputs, max_new_tokens=256)


        return {
            "language": language,
            "language_probability": 100,
            "segments": outputs,
        }

  
test_asr = ASR()
out = ASR.transcribe("")