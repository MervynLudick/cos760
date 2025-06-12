# -*- coding: utf-8 -*-
"""Speech Recognition Comparison: Whisper vs SeamlessM4T-v2 vs OpenAI API"""

from datasets import load_dataset, Audio
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, SeamlessM4Tv2ForSpeechToText
from huggingface_hub import login
import evaluate
import openai
import tempfile
import soundfile as sf
import os

# Configuration
DATASET_NAME = "mozilla-foundation/common_voice_17_0"
LANGUAGE = "af"
WHISPER_MODELS = ["openai/whisper-large-v3", "openai/whisper-small"]
SEAMLESS_MODEL = "facebook/seamless-m4t-v2-large"

def setup_device_and_dtype():
    """Setup device and data type for models."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return device, torch_dtype

def load_dataset_with_audio():
    """Load and prepare the Common Voice dataset."""
    cv_17 = load_dataset(DATASET_NAME, LANGUAGE, split="train")
    cv_17 = cv_17.cast_column("audio", Audio(sampling_rate=16000))
    return cv_17

def evaluate_model_predictions(predictions, references, model_name):
    """Evaluate and print WER results for a model."""
    wer_metric = evaluate.load("wer")
    wer_score = wer_metric.compute(predictions=predictions, references=references)
    print(f"WER for {model_name}: {wer_score:.4f}")
    return wer_score

def transcribe_with_whisper(dataset, model_id, device, torch_dtype):
    """Transcribe audio using Whisper models."""
    print(f"\nTesting {model_id}...")
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={"language": LANGUAGE}
    )
    
    predictions, references = [], []
    for sample in dataset:
        result = pipe(sample["audio"])
        predictions.append(result["text"])
        references.append(sample["sentence"])
        print(f"Actual: {sample['sentence']}")
        print(f"Prediction: {result['text']}\n")
    
    return predictions, references

def transcribe_with_seamless(dataset, device):
    """Transcribe audio using SeamlessM4T-v2."""
    print(f"\nTesting {SEAMLESS_MODEL}...")
    
    processor = AutoProcessor.from_pretrained(SEAMLESS_MODEL)
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(SEAMLESS_MODEL)
    model.to(device)
    
    predictions, references = [], []
    for sample in dataset:
        audio_array = sample["audio"]["array"]
        sampling_rate = sample["audio"]["sampling_rate"]
        
        inputs = processor(
            audios=audio_array, sampling_rate=sampling_rate,
            return_tensors="pt", src_lang="afr"
        ).to(device)
        
        output_tokens = model.generate(**inputs, tgt_lang="afr")
        transcription = processor.batch_decode(output_tokens, skip_special_tokens=True)[0]
        
        predictions.append(transcription)
        references.append(sample["sentence"])
        print(f"Actual: {sample['sentence']}")
        print(f"Prediction: {transcription}\n")
    
    return predictions, references

def transcribe_with_openai_api(dataset):
    """Transcribe audio using OpenAI API."""
    print("\nTesting OpenAI API...")
    
    api_key = os.getenv("OPENAI_API_KEY", "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")
    openai.api_key = api_key
    
    predictions, references = [], []
    for sample in dataset:
        audio_array = sample["audio"]["array"]
        sample_rate = sample["audio"]["sampling_rate"]
        
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
            sf.write(temp_file.name, audio_array, samplerate=sample_rate)
            temp_file.flush()
            
            with open(temp_file.name, "rb") as f:
                response = openai.audio.transcriptions.create(
                    model="gpt-4o-transcribe", file=f, 
                    response_format="text", language=LANGUAGE
                )
        
        transcription = response.strip()
        predictions.append(transcription)
        references.append(sample["sentence"])
        print(f"Actual: {sample['sentence']}")
        print(f"Prediction: {transcription}\n")
    
    return predictions, references

def main():
    """Main execution function."""
    print("=== Speech Recognition Model Comparison ===")
    
    # Hugging Face authentication
    login()
    
    # Setup
    device, torch_dtype = setup_device_and_dtype()
    dataset = load_dataset_with_audio()
    results = {}
    
    # Test all models
    for model_id in WHISPER_MODELS:
        predictions, references = transcribe_with_whisper(dataset, model_id, device, torch_dtype)
        results[model_id] = evaluate_model_predictions(predictions, references, model_id)
    
    predictions, references = transcribe_with_seamless(dataset, device)
    results[SEAMLESS_MODEL] = evaluate_model_predictions(predictions, references, SEAMLESS_MODEL)
    
    predictions, references = transcribe_with_openai_api(dataset)
    results["OpenAI API"] = evaluate_model_predictions(predictions, references, "OpenAI API")
    
    # Final comparison
    print(f"\n=== FINAL RESULTS ===")
    for model, wer in results.items():
        print(f"{model}: WER = {wer:.4f}")
    
    best_model = min(results, key=results.get)
    print(f"\nBest performing model: {best_model} (WER: {results[best_model]:.4f})")

if __name__ == "__main__":
    main()
