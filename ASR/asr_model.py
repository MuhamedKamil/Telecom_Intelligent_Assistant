import torch
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import load_audio
from typing import Optional, Dict, List, Any, Union


class ASR:
    """
    Detects the spoken language, then converts speech into text for the RAG/LLM pipeline    
    """

    def __init__(
        self,
        ASR_config: Optional[Dict] 

    ):
        
        self.language_detector_name: str = ASR_config["language_detector_name"]
        self.device: str                 = ASR_config["device"]
        self.compute_type: str           = ASR_config["compute_type"]
        self.asr_model:str               = ASR_config["asr_model_name"]
        self.sampling_rate  :int         = ASR_config["sampling_rate"]
        self.max_new_tokens :int         = ASR_config["max_new_tokens"]
 
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available. Falling back to CPU.")
            self.device = "cpu"

        self.language_detector = WhisperModel(self.language_detector_name,device=self.device,compute_type=self.compute_type)
        self.processor         = AutoProcessor.from_pretrained(self.asr_model)
        self.asr_model         = CohereAsrForConditionalGeneration.from_pretrained(self.asr_model, device_map=self.device)

        print(f"Loaded Faster-Whisper Language Detector: {self.language_detector_name}")
        print(f"Loaded ASR MODEL : {self.asr_model}")


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
        audio_path :str,
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
        inputs = self.processor(audio_path, sampling_rate = self.sampling_rate, language = language, return_tensors="pt").to(
            device = self.asr_model.device, 
            dtype  = self.asr_model.dtype) 

        outputs = self.asr_model.generate(**inputs, max_new_tokens= self.max_new_tokens)

        return {
            "language"             : language,
            "language_probability" : prob,
            "segments"             : outputs,
        }
