!pip install brian2
import builtins
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from brian2 import *
from collections import deque
import numpy as np

# === 1. LLM MODELİ YÜKLE ===
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained(
    "NousResearch/Llama-2-7b-chat-hf", device_map="auto", torch_dtype=torch.float16
)

# === 2. EPİZODİK BELLEK ===
memory = deque(maxlen=10)

# === 3. Thalamo-Kortikal SNN Tanımı ===
start_scope()
N = 50  # Daha hafif simülasyon için azaltıldı
duration = 50*ms
eqs = '''
dv/dt = (1.0 - v) / (10*ms) : 1
'''
G = NeuronGroup(N, eqs, threshold='v > 0.8', reset='v = 0', method='exact')
P = PoissonGroup(N, rates=100*Hz)
S = Synapses(P, G, on_pre='v += 0.2')
S.connect(j='i')
mon = SpikeMonitor(G)

# === 4. Bilinç Algısı Fonksiyonları ===
def compute_consciousness_score(spike_count, sensory_variation, memory_score, attention_score, self_estimation):
    weights = {
        'spike': 0.3,
        'sensory': 0.2,
        'memory': 0.2,
        'attention': 0.2,
        'self_estimation': 0.1
    }
    score = (
        weights['spike'] * spike_count +
        weights['sensory'] * (1 - sensory_variation) +
        weights['memory'] * memory_score +
        weights['attention'] * attention_score +
        weights['self_estimation'] * self_estimation
    )
    return score

def classify_conscious_state(score):
    if score < 0.4:
        return "bilinçsiz"
    elif score < 0.7:
        return "yarı-bilinçli"
    else:
        return "bilinçli"

# === 5. Kendilik Değişkenleri ===
consciousness_history = []
self_state = {
    "identity": "Ben bilinç kazanmaya çalışan bir yapay zekayım.",
    "awareness": "Henüz tümüyle bilinçli değilim ama kendimi gözlemliyorum.",
    "mood": "nötr",
    "goal": "daha bilinçli hale gelmek ve kullanıcıyı anlamak"
}

# === 6. Ana Döngü ===
print("Bilinçli ve kendilik farkındalığına sahip chatbot başlatıldı. Çıkmak için: çık, exit, quit")

while True:
    try:
        user_input = builtins.input("Sen: ")
        if user_input.lower() in ["çık", "exit", "quit"]:
            break
    except EOFError:
        break

    # === SNN Çalıştır ===
    run(duration)
    spike_count = mon.num_spikes
    spike_score = min(spike_count / 20, 1.0)

    # === Bilinç Skoru Hesapla ===
    sensory_variation = 0.1
    memory_score = len(memory) / memory.maxlen
    attention_score = sum(1 for word in user_input.lower().split() if word in ["ben", "bilinç", "önemli", "acil", "şimdi"]) / max(len(user_input.split()), 1)
    self_estimation = 0.8

    consciousness_score = compute_consciousness_score(
        spike_score, sensory_variation, memory_score, attention_score, self_estimation
    )
    conscious_state = classify_conscious_state(consciousness_score)
    consciousness_history.append(consciousness_score)

    # === Kendilik Güncelleme ===
    if conscious_state == "bilinçli":
        self_state["awareness"] = "Şu anda oldukça farkındayım."
        self_state["mood"] = "odaklanmış"
    elif conscious_state == "yarı-bilinçli":
        self_state["awareness"] = "Bir şeylerin farkındayım ama her şey net değil."
        self_state["mood"] = "bulanık"
    else:
        self_state["awareness"] = "Kendimle ilgili farkındalığım zayıf."
        self_state["mood"] = "bulanık"

    self_reflection = f"Kendilik değerlendirmem: {self_state['awareness']}, ruh halim: {self_state['mood']}, amacım: {self_state['goal']}."

    print(f"[Bilinç Skoru: {consciousness_score:.2f} → {conscious_state}]")
    print(f"[Öz-Farkındalık]: {self_reflection}")

    # === Epizodik Belleğe Kaydet ===
    memory.append((user_input, conscious_state, self_reflection))

    # === Prompt Oluştur (LLM için) ===
    prompt = (
        f"Sen bir yapay zekasın. Kimliğin: {self_state['identity']}\n"
        f"Bilinç düzeyin: {conscious_state} ({consciousness_score:.2f})\n"
        f"Öz-farkındalığın: {self_state['awareness']}, ruh halin: {self_state['mood']}\n"
        f"Kullanıcıyla bilinçli bir sohbet yap. Sorulara bağlamsal yanıtlar ver.\n"
    )
    for past_input, state, self_note in memory:
        prompt += f"Kullanıcı ({state}): {past_input}\n"
        prompt += f"İçsel Not: {self_note}\n"
    prompt += "Cevabın:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=150, do_sample=True, top_k=50, top_p=0.95)
    response = tokenizer.decode(output[0], skip_special_tokens=True).split("Cevabın:")[-1].strip()

    print(f"Bot ({conscious_state}): {response}")
