# -*- coding: utf-8 -*-
"""
Speech Recognition Comparison: Whisper vs SeamlessM4T-v2 vs OpenAI API
"""

from datasets import load_dataset, Audio
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, SeamlessM4Tv2ForSpeechToText
import evaluate
import openai
import tempfile
import soundfile as sf

# Load dataset once
cv_17 = load_dataset("mozilla-foundation/common_voice_17_0", "af", split="train")
cv_17 = cv_17.cast_column("audio", Audio(sampling_rate=16000))

# Setup device once
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def test_whisper_model(model_id):
    """Test a Whisper model and return WER."""
    print(f"\n=== Testing {model_id} ===")
    
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
        generate_kwargs={"language": "af"}
    )
    
    predictions, references = [], []
    for x in cv_17:
        result = pipe(x["audio"])
        predictions.append(result["text"])
        references.append(x["sentence"])
        print(f"Actual: {x['sentence']}")
        print(f"Prediction: {result['text']}\n")
    
    wer = evaluate.load("wer").compute(predictions=predictions, references=references)
    print(f"WER for {model_id}: {wer:.4f}")
    return wer

# Test both Whisper models
whisper_large_wer = test_whisper_model("openai/whisper-large-v3")
whisper_small_wer = test_whisper_model("openai/whisper-small")

# Test SeamlessM4T-v2
print(f"\n=== Testing SeamlessM4T-v2 ===")
processor_m4tv2 = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model_m4tv2 = SeamlessM4Tv2ForSpeechToText.from_pretrained("facebook/seamless-m4t-v2-large")
model_m4tv2.to(device)

predictions_m4tv2, references_m4tv2 = [], []
for x in cv_17:
    audio_array = x["audio"]["array"]
    sampling_rate = x["audio"]["sampling_rate"]
    
    inputs = processor_m4tv2(
        audios=audio_array, sampling_rate=sampling_rate, 
        return_tensors="pt", src_lang="afr"
    ).to(device)
    
    output_tokens = model_m4tv2.generate(**inputs, tgt_lang="afr")
    transcription = processor_m4tv2.batch_decode(output_tokens, skip_special_tokens=True)[0]
    
    predictions_m4tv2.append(transcription)
    references_m4tv2.append(x["sentence"])
    print(f"Actual: {x['sentence']}")
    print(f"Prediction: {transcription}\n")

seamless_wer = evaluate.load("wer").compute(predictions=predictions_m4tv2, references=references_m4tv2)
print(f"WER for SeamlessM4T-v2: {seamless_wer:.4f}")

# Test OpenAI API
print(f"\n=== Testing OpenAI API ===")
openai.api_key = "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"

predictions_openai, references_openai = [], []
for sample in cv_17:
    audio_array = sample["audio"]["array"]
    sample_rate = sample["audio"]["sampling_rate"]
    
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
        sf.write(temp_file.name, audio_array, samplerate=sample_rate)
        temp_file.flush()
        
        with open(temp_file.name, "rb") as f:
            response = openai.audio.transcriptions.create(
                model="gpt-4o-transcribe", file=f, 
                response_format="text", language="af"
            )
    
    transcription = response.strip()
    predictions_openai.append(transcription)
    references_openai.append(sample["sentence"])
    print(f"Actual: {sample['sentence']}")
    print(f"Prediction: {transcription}\n")

openai_wer = evaluate.load("wer").compute(predictions=predictions_openai, references=references_openai)
print(f"WER for OpenAI API: {openai_wer:.4f}")

# Final comparison
print(f"\n=== FINAL RESULTS ===")
print(f"Whisper Large-v3: {whisper_large_wer:.4f}")
print(f"Whisper Small: {whisper_small_wer:.4f}")
print(f"SeamlessM4T-v2: {seamless_wer:.4f}")
print(f"OpenAI API: {openai_wer:.4f}")