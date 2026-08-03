import torch
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import load_audio


class ASR:
    """
    Detects the spoken language, then converts speech into text for the RAG/LLM pipeline    
    """

    def __init__(
        self,
        language_detector_name: str = "base",
        device: str                 = "cuda",
        compute_type: str           = "int8",
        asr_model:str               = "namaa-space/cohere-transcribe-arabic-07-2026-int4",
    ):
 
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available. Falling back to CPU.")
            device = "cpu"

        self.language_detector = WhisperModel(language_detector_name,device=device,compute_type=compute_type,)
        self.processor         = AutoProcessor.from_pretrained(asr_model),
        self.asr_model         = CohereAsrForConditionalGeneration.from_pretrained(asr_model, device_map=device)

        print(f"Loaded Faster-Whisper Language Detector: {language_detector_name}")
        print(f"Loaded ASR MODEL : {asr_model}")


    def detect_language(self,audio_path):
        """
        Detect language in audio
        Args:
            audio_path: path to audio.
            Returns:
                selected_lang(str): string represent the language.
                selected_prob(float): number represent the prob.
        """
        audio = decode_audio(audio_path,sampling_rate=self.language_detector.feature_extractor.sampling_rate,)
        language, language_probability, all_language_probs = self.language_detector.detect_language(audio)
        lang_probs = dict(all_language_probs)
        en_prob = lang_probs.get("en", 0.0)
        ar_prob = lang_probs.get("ar", 0.0)
        if ar_prob >= en_prob:
            selected_lang = "ar"
            selected_prob = ar_prob
        else:
            selected_lang = "en"
            selected_prob = en_prob
       
        return selected_lang, selected_prob

    def transcribe(
        self,
        audio_path     :str,
        sampling_rate  :int = 16000,
        max_new_tokens :int = 256
    ):
        """
        Transcribe speech from an audio file into text using the ASR (Automatic Speech Recognition) model.
        Args:
            audio_path (str): 
                Path to the audio file to be transcribed.
                
            sampling_rate (int): 
                Target sampling rate in Hz. Default: 16000.
                Controls audio quality and processing speed.
                
            max_new_tokens (int): 
                Maximum length of generated transcription. Default: 256.
                Controls how long the output text can be.
        """
        language, prob = self.detect_language(audio_path)
        inputs = self.processor(audio_path, sampling_rate = sampling_rate, language= language, return_tensors="pt").to(
            device = self.asr_model.device, 
            dtype  = self.asr_model.dtype) 

        outputs = self.asr_model.generate(**inputs, max_new_tokens= max_new_tokens)

        return {
            "language"             : language,
            "language_probability" : prob,
            "segments"             : outputs,
        }
