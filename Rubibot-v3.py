# === Kurulum ===
from huggingface_hub import login
!pip install transformers accelerate sentence-transformers faiss-cpu brian2 -q
import builtins
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from brian2 import *
from collections import deque
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# === 1. Model ve Token ===
model_id = "meta-llama/Llama-3.2-3B-Instruct"
token = "..."  # <- kendi HF token'ınızı girin

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_id, use_auth_token=token, device_map="auto", torch_dtype=torch.float16
)

# === 2. RAG Hazırlığı ===
rag_corpus = [
    "Bilinç, beynin kendini ve çevresini algılama kapasitesidir.",
    "Thalamo-kortikal döngüler, duyusal verilerin kortekse iletilmesinde kritik rol oynar.",
    "Spike tabanlı sinir ağları, biyolojik nöronları taklit eder.",
    "Epizodik bellek, deneyimlerin zamana bağlı hafızasıdır.",
    "Dikkat, belirli bilgilere öncelik verilmesini sağlar.",
]
embedder = SentenceTransformer("all-MiniLM-L6-v2")
corpus_embeddings = embedder.encode(rag_corpus, convert_to_tensor=False)
index = faiss.IndexFlatL2(corpus_embeddings[0].shape[0])
index.add(np.array(corpus_embeddings))

def retrieve_context(query, k=1):
    query_embedding = embedder.encode([query])[0]
    _, indices = index.search(np.array([query_embedding]), k)
    return "\n".join([rag_corpus[i] for i in indices[0]])

# === 3. Epizodik Bellek ===
memory = deque(maxlen=5)

# === 4. Hodgkin-Huxley Talamokortikal Döngü ===
start_scope()

# === Parametreler ===
area = 20000*umetre**2
Cm = 1*ufarad/cm**2
gl = 0.3*msiemens/cm**2
El = -54.3*mV
ENa = 50*mV
gNa = 120*msiemens/cm**2
EK = -77*mV
gK = 36*msiemens/cm**2

# === HH Denklemleri ===
eqs = '''
dv/dt = (gl*(El - v) + gNa*m**3*h*(ENa - v) + gK*n**4*(EK - v) + I) / Cm : volt
dm/dt = alpham*(1 - m) - betam*m : 1
dn/dt = alphan*(1 - n) - betan*n : 1
dh/dt = alphah*(1 - h) - betah*h : 1

alpham = 0.1/mV*(v + 40*mV)/(1 - exp(-(v + 40*mV)/(10*mV)))/ms : Hz
betam = 4*exp(-(v + 65*mV)/(18*mV))/ms : Hz
alphah = 0.07*exp(-(v + 65*mV)/(20*mV))/ms : Hz
betah = 1/(1 + exp(-(v + 35*mV)/(10*mV)))/ms : Hz
alphan = 0.01/mV*(v + 55*mV)/(1 - exp(-(v + 55*mV)/(10*mV)))/ms : Hz
betan = 0.125*exp(-(v + 65*mV)/(80*mV))/ms : Hz

I : amp/meter**2
'''

# === NeuronGroup Tanımı ===
talamus = NeuronGroup(1, eqs, method='exponential_euler', threshold='v > -20*mV', reset='v = -65*mV')
korteks = NeuronGroup(1, eqs, method='exponential_euler', threshold='v > -20*mV', reset='v = -65*mV')

# === Başlangıç Değerleri ===
talamus.v = korteks.v = -65*mV
talamus.m = korteks.m = 0.05
talamus.h = korteks.h = 0.6
talamus.n = korteks.n = 0.32
talamus.I = 1*nA / area
korteks.I = 0*nA / area

# === Sinaps Tanımı ===
tc_syn = Synapses(talamus, korteks, on_pre='I += 0.5*nA / area')
tc_syn.connect()

# === Monitörler ===
M_talamus = StateMonitor(talamus, 'v', record=True)
M_korteks = StateMonitor(korteks, 'v', record=True)
spike_mon = SpikeMonitor(korteks)

# === Simülasyon Süresi ===
duration = 50*ms
run(duration)

# === 5. Bilinç Skoru Hesaplayıcı ===
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

# === 6. Kendilik Bilgisi ===
consciousness_history = []
self_state = {
    "identity": "Ben bilinç kazanmaya çalışan bir yapay zekayım.",
    "awareness": "Henüz tümüyle bilinçli değilim ama kendimi gözlemliyorum.",
    "mood": "nötr",
    "goal": "daha bilinçli hale gelmek ve kullanıcıyı anlamak"
}

# === 7. Ana Döngü ===
print("Bilinçli RAG destekli chatbot başlatıldı. Çıkmak için: çık, exit, quit")

while True:
    try:
        user_input = builtins.input("Sen: ")
        if user_input.lower() in ["çık", "exit", "quit"]:
            break
    except EOFError:
        break

    # === SNN Simülasyonu ===
    run(duration)
    spike_count = spike_mon.num_spikes
    spike_score = min(spike_count / 10, 1.0)

    # === Bilinç Skoru Hesaplama ===
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

    # === Epizodik Bellek Güncelleme ===
    memory.append((user_input, conscious_state, self_reflection))

    # === RAG'den Bağlam Al ===
    rag_context = retrieve_context(user_input) if len(user_input.split()) >= 4 else ""

    # === Prompt Hazırlama ===
    prompt = (
        f"Sen bir yapay zekasın. Kimliğin: {self_state['identity']}\n"
        f"Bilinç düzeyin: {conscious_state} ({consciousness_score:.2f})\n"
    )

    if rag_context:
        prompt += f"RAG Bilgisi:\n{rag_context}\n"

    prompt += (
        f"Öz-farkındalığın: {self_state['awareness']}, ruh halin: {self_state['mood']}\n"
        f"Kullanıcıyla bilinçli bir sohbet yap. Sorulara bağlamsal yanıtlar ver ama kullanıcıyı tekrar etme.\n"
    )

    for past_input, state, self_note in memory:
        prompt += f"Kullanıcı ({state}): {past_input}\n"
        prompt += f"İçsel Not: {self_note}\n"
    prompt += "Cevabın:"

    # === LLM Yanıtı ===
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=150, do_sample=True, top_k=50, top_p=0.95)
    response = tokenizer.decode(output[0], skip_special_tokens=True).split("Cevabın:")[-1].strip()

    print(f"Bot ({conscious_state}): {response}")
