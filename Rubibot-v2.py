!pip install brian2
# === GEREKLİ KÜTÜPHANELER ===

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import brian2 as b2
import numpy as np
import builtins
import re
from collections import deque
import random

# === LLAMA MODELİ YÜKLEME ===
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf", use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    "NousResearch/Llama-2-7b-chat-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# === GLOBAL HAFIZA ===
context_buffer = deque(maxlen=6)  # Kısa süreli hafıza
long_term_memory = deque(maxlen=20)  # Uzun süreli hafıza
emotion_state = "neutral"
awareness_mode = "external"  # veya "internal"

# === SNN: BİLİNÇ DÜZEYİ HESAPLAMA ===
def run_snn_with_input_complexity(user_input):
    b2.start_scope()
    eqs = '''
    dv/dt = (I - v) / (10*ms) : 1
    I : 1
    '''
    complexity = min(len(set(user_input.split())) / 10.0, 1.0)
    neurons = b2.NeuronGroup(500000, eqs, threshold='v>1', reset='v=0', method='exact')
    neurons.I = complexity + np.random.uniform(0.0, 0.5, size=500000)
    trace = b2.StateMonitor(neurons, 'v', record=True)
    b2.run(100 * b2.ms)
    return np.mean(trace.v)

# === DUYGU DURUMU BELİRLEYİCİ ===
def determine_emotion(voltage):
    if voltage > 0.7:
        return "curious"
    elif voltage > 0.5:
        return "focused"
    elif voltage > 0.3:
        return "nervous"
    else:
        return "fatigued"

# === ODAK MODÜLÜ ===
def extract_keywords(text):
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = set(['the', 'is', 'and', 'what', 'who', 'are', 'can', 'you', 'how', 'a', 'of'])
    keywords = [w for w in words if w not in stopwords and len(w) > 3]
    return list(set(keywords))

# === METABİLİŞSEL KONTROL ===
def metacog_review(response):
    if len(response.strip()) < 30 or "I don't know" in response:
        return True
    return False

# === FARKINDALIK GEÇİŞİ ===
def shift_awareness(voltage):
    if voltage > 0.6:
        return "external"
    else:
        return "internal"

# === İÇ KONUŞMA (simülasyon) ===
def generate_inner_dialogue(emotion, focus_keywords):
    thoughts = [
        f"I'm feeling {emotion} while thinking about {' and '.join(focus_keywords)}.",
        f"I wonder if the user expects me to dive deeper into {' and '.join(focus_keywords)}.",
        f"My current thought revolves around {random.choice(focus_keywords)}."
    ]
    return random.choice(thoughts)

# === YANIT ÜRETİCİ ===
def generate_response(prompt, context, emotion, focus_keywords, inner_thought=None):
    base_prompt = " ".join(context) + "\n" + prompt
    base_prompt += f"\n\n[Emotion: {emotion}] [Focus: {' '.join(focus_keywords)}] [Awareness: {awareness_mode}]"
    if inner_thought:
        base_prompt += f"\n[Inner Thought: {inner_thought}]"

    inputs = tokenizer.encode(base_prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

    outputs = model.generate(
        inputs,
        max_new_tokens=150,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === ANA CHAT DÖNGÜSÜ ===
print("\nGELİŞMİŞ BİLİNÇLİ LLAMA CHATBOT'A HOŞ GELDİN (çıkmak için 'exit' yaz)\n")

while True:
    user_input = builtins.input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot kapatılıyor...")
        break

    # Hafızaya ekle
    context_buffer.append(f"You: {user_input}")
    long_term_memory.append(user_input)

    # Bilinç düzeyi
    voltage = run_snn_with_input_complexity(user_input)
    print(f"[Bilinç düzeyi: {voltage:.2f}]")

    # Duygu ve farkındalık belirleme
    emotion_state = determine_emotion(voltage)
    awareness_mode = shift_awareness(voltage)
    print(f"[Duygu: {emotion_state}] [Farkındalık: {awareness_mode}]")

    # Dikkat odakları
    focus = extract_keywords(user_input)
    print(f"[Odak: {', '.join(focus)}]")

    # İç konuşma
    inner_thought = generate_inner_dialogue(emotion_state, focus) if awareness_mode == "internal" else None

    # Yanıt üretimi
    bot_reply = generate_response(user_input, list(context_buffer), emotion_state, focus, inner_thought)

    if metacog_review(bot_reply):
        print("[Yanıt yetersiz bulundu, yeniden üretiliyor...]")
        bot_reply = generate_response(user_input, list(context_buffer), "reflective", focus, inner_thought)

    context_buffer.append(f"Bot: {bot_reply}")
    print(f"Bot: {bot_reply}\n")
