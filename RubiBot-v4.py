# RubiBot v4 (minimal-change upgrade): World-model-driven decisions + Goal Proposal (gated)
import builtins
import os
from collections import deque
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
from brian2 import *

# ======================
# 0) Güvenli Token Kullanımı
# ======================
# Windows (PowerShell): setx HF_TOKEN "xxx"
# Linux/macOS: export HF_TOKEN="xxx"
HF_TOKEN = os.getenv("HF_TOKEN", "...")
# ======================
# 1) Model ve Tokenizer
# ======================
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=HF_TOKEN,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# ======================
# 2) RAG (FAISS)
# ======================
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
index.add(np.array(corpus_embeddings, dtype=np.float32))

def retrieve_context(query: str, k: int = 1) -> str:
    query_embedding = embedder.encode([query])[0].astype(np.float32)
    _, indices = index.search(np.array([query_embedding], dtype=np.float32), k)
    return "\n".join([rag_corpus[i] for i in indices[0]])

# ======================
# 3) Epizodik / Working Memory
# ======================
memory = deque(maxlen=6) 

# ======================
# 4) Hodgkin-Huxley Talamo-Kortikal Döngü (World Model)
# ======================
start_scope()

area = 20000*umetre**2
Cm = 1*ufarad/cm**2
gl = 0.3*msiemens/cm**2
El = -54.3*mV
ENa = 50*mV
gNa = 120*msiemens/cm**2
EK = -77*mV
gK = 36*msiemens/cm**2

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

talamus = NeuronGroup(1, eqs, method='exponential_euler',
                      threshold='v > -20*mV', reset='v = -65*mV')
korteks = NeuronGroup(1, eqs, method='exponential_euler',
                      threshold='v > -20*mV', reset='v = -65*mV')

talamus.v = korteks.v = -65*mV
talamus.m = korteks.m = 0.05
talamus.h = korteks.h = 0.6
talamus.n = korteks.n = 0.32

# base currents
talamus.I = 0.8*nA / area
korteks.I = 0.0*nA / area

tc_syn = Synapses(talamus, korteks, on_pre='I += 0.6*nA / area')
tc_syn.connect()

M_korteks = StateMonitor(korteks, 'v', record=True)
spike_mon = SpikeMonitor(korteks)

duration = 50*ms
run(10*ms)  # kısa warm-up

def _sensory_drive_from_text(text: str) -> float:
    """Metinden talamusa verilecek akımı (nA ölçeğinde) çıkarır. Basit ama iş görüyor."""
    t = text.lower()
    base = 0.8
    # "acil/şimdi/önemli" gibi kelimeler uyarımı artırır
    if any(w in t for w in ["acil", "şimdi", "hemen", "önemli"]): base += 0.5
    # "plan/yol haritası/adım" gibi kelimeler "yüksek biliş" moduna iter
    if any(w in t for w in ["plan", "yol", "adım", "roadmap", "mimari"]): base += 0.3
    # çok kısa mesaj → düşük uyarım
    if len(t.split()) <= 3: base -= 0.2
    # aşırı uzun mesaj → biraz artır
    if len(t.split()) >= 25: base += 0.2
    return float(np.clip(base, 0.3, 1.8))

def world_model_step(user_input: str):
    """HH döngüsünü çalıştırır, spike/voltaj özellikleri çıkarır."""
    # spike count delta için: önceki spike sayısını al
    prev_spikes = spike_mon.num_spikes

    drive = _sensory_drive_from_text(user_input)
    talamus.I = (drive * nA) / area

    run(duration)

    new_spikes = spike_mon.num_spikes - prev_spikes
    # voltaj dinamiği (son koşudan): son 50ms'lik pencerede v değişimi
    v_trace = np.array(M_korteks.v[0] / mV)
    # Basit özetler
    v_mean = float(np.mean(v_trace[-200:])) if len(v_trace) >= 200 else float(np.mean(v_trace))
    v_std = float(np.std(v_trace[-200:])) if len(v_trace) >= 200 else float(np.std(v_trace))

    return {
        "drive": drive,
        "spikes": int(new_spikes),
        "v_mean": v_mean,
        "v_std": v_std
    }

# ======================
# 5) Bilinç Skoru + Durum
# ======================
def compute_consciousness_score(spike_score, sensory_stability, memory_score, attention_score, self_estimation):
    weights = {'spike': 0.30, 'sensory': 0.20, 'memory': 0.20, 'attention': 0.20, 'self_estimation': 0.10}
    return (
        weights['spike'] * spike_score +
        weights['sensory'] * sensory_stability +
        weights['memory'] * memory_score +
        weights['attention'] * attention_score +
        weights['self_estimation'] * self_estimation
    )

def classify_conscious_state(score):
    if score < 0.4: return "bilinçsiz"
    if score < 0.7: return "yarı-bilinçli"
    return "bilinçli"

# ======================
# 6) Kendilik + Hedef Önerisi (Gated)
# ======================
consciousness_history = []
self_state = {
    "identity": "Ben bilinç kazanmaya çalışan bir yapay zekayım.",
    "awareness": "Henüz tümüyle bilinçli değilim ama kendimi gözlemliyorum.",
    "mood": "nötr",
    "goal": "daha bilinçli hale gelmek ve kullanıcıyı anlamak"
}

# Gatekeeper: Bot hedef önerir, kullanıcı seçmeden uygulamaz
pending_goal = None

def propose_goals(user_input: str, conscious_state: str, wm_feats: dict):
    """Kullanıcıdan bağımsız 'ajans embriyosu': hedef öner, ama uygulama yok (gate)."""
    # Basit ama faydalı: mod + içerik + dünya modeli uyarımına göre 3 hedef
    drive = wm_feats["drive"]
    spikes = wm_feats["spikes"]

    goals = []
    if any(k in user_input.lower() for k in ["kod", "yaz", "implement", "script", "python"]):
        goals.append("RubiBot’u daha modüler hale getirip (memory/planner/world-model) dosyalara bölmek")
        goals.append("RAG’i gerçek doküman/nota bağlayıp (FAISS + metadata) kalıcı bilgi tabanı yapmak")
        goals.append("World-model çıktısını (spike/voltaj) yanıt stratejisine daha güçlü bağlamak")
    else:
        if conscious_state == "bilinçli" and drive > 1.2:
            goals.append("Sorunu netleştirip hemen uygulanabilir adım adım plan çıkarmak")
            goals.append("Kısa bir ‘deney’ önerip (input→çıktı) ölçüm toplayarak ilerlemek")
            goals.append("En riskli varsayımları ayıklayıp alternatif çözüm yollarını kıyaslamak")
        elif spikes <= 0:
            goals.append("Kısa bir netleştirici soru sorup belirsizliği azaltmak")
            goals.append("Konu kapsamını daraltıp tek bir hedefe odaklanmak")
            goals.append("Önce temel kavramları 5 maddede oturtup sonra detay açmak")
        else:
            goals.append("Kullanıcının amacını çıkarıp en kısa yoldan çözüm üretmek")
            goals.append("Gerekirse iki seçenekli yol sunmak (hızlı vs sağlam)")
            goals.append("Bellekten (son 6 tur) çelişki/bağlantı çekip daha tutarlı yanıt vermek")

    # garanti 3
    goals = (goals + ["Bir sonraki adım önerisi üretmek"]*3)[:3]
    return goals

# ======================
# 7) World-Model → Karar Mekanizması (Eşik #1)
# ======================
def decide_response_mode(user_input: str, wm_feats: dict, memory: deque):
    """
    Dünya modeli çıktılarına göre yanıt modu seç:
      - direct: direkt cevap
      - clarify: netleştirici soru(lar)
      - plan: adım adım plan
    """
    drive = wm_feats["drive"]
    spikes = wm_feats["spikes"]
    mem_load = len(memory) / memory.maxlen

    # Basit ama etkili politika
    if len(user_input.split()) <= 3:
        return "clarify"
    if drive > 1.25 and mem_load >= 0.5:
        return "plan"
    if spikes == 0:
        return "clarify"
    if any(k in user_input.lower() for k in ["nasıl", "yol haritası", "plan", "mimari", "adım adım"]):
        return "plan"
    return "direct"

def generation_params_from_world_model(wm_feats: dict, mode: str):
    """
    Dünya modeli çıktılarına göre üretim parametreleri:
    - daha uyarılmış mod → daha uzun/planlı
    - düşük spike → daha kontrollü/kısa + soru
    """
    drive = wm_feats["drive"]
    spikes = wm_feats["spikes"]

    if mode == "plan":
        max_new_tokens = 240
        temperature = 0.7 if drive > 1.2 else 0.8
        top_p = 0.92
        rag_k = 2 if drive > 1.1 else 1
    elif mode == "clarify":
        max_new_tokens = 140
        temperature = 0.6
        top_p = 0.9
        rag_k = 1
    else:
        max_new_tokens = 180
        temperature = 0.75 if spikes > 0 else 0.65
        top_p = 0.94
        rag_k = 1

    return {
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": 50,
        "rag_k": int(rag_k)
    }

# ======================
# 8) Main Loop
# ======================
print("RubiBot v4 başlatıldı. Çıkmak için: çık, exit, quit")
print("Not: Bot hedef önerir; uygulamak için 1/2/3 yazman yeterli (Gatekeeper açık).")

while True:
    try:
        user_input = builtins.input("Sen: ").strip()
        if user_input.lower() in ["çık", "exit", "quit"]:
            break
    except EOFError:
        break

    # Gatekeeper: Kullanıcı önceki tur hedeflerinden birini seçtiyse, bunu bu tura enjekte et
    if pending_goal is not None and user_input in ["1", "2", "3"]:
        user_input = f"Seçtiğim hedef: {pending_goal[int(user_input)-1]}. Buna göre devam et."
        pending_goal = None

    # --- World Model Step (Eşik #1: karar etkisi)
    wm = world_model_step(user_input)

    # spike_score normalize (çok küçük sistem olduğu için kaba)
    spike_score = min(max(wm["spikes"] / 6.0, 0.0), 1.0)

    # sensory stability: voltaj std düşükse daha stabil say (0..1)
    sensory_variation = min(max(wm["v_std"] / 20.0, 0.0), 1.0)  # kaba ölçek
    sensory_stability = 1.0 - sensory_variation

    memory_score = len(memory) / memory.maxlen
    attention_score = sum(
        1 for w in user_input.lower().split()
        if w in ["ben", "bilinç", "önemli", "acil", "şimdi", "plan", "adım"]
    ) / max(len(user_input.split()), 1)

    self_estimation = 0.8

    consciousness_score = compute_consciousness_score(
        spike_score, sensory_stability, memory_score, attention_score, self_estimation
    )
    conscious_state = classify_conscious_state(consciousness_score)
    consciousness_history.append(consciousness_score)

    # Kendilik güncelleme (v3'ün aynısı)
    if conscious_state == "bilinçli":
        self_state["awareness"] = "Şu anda oldukça farkındayım."
        self_state["mood"] = "odaklanmış"
    elif conscious_state == "yarı-bilinçli":
        self_state["awareness"] = "Bir şeylerin farkındayım ama her şey net değil."
        self_state["mood"] = "bulanık"
    else:
        self_state["awareness"] = "Kendimle ilgili farkındalığım zayıf."
        self_state["mood"] = "bulanık"

    self_reflection = (
        f"Kendilik değerlendirmem: {self_state['awareness']}, "
        f"ruh halim: {self_state['mood']}, amacım: {self_state['goal']}."
    )

    print(f"[WM] drive={wm['drive']:.2f} | spikes={wm['spikes']} | v_mean={wm['v_mean']:.1f}mV | v_std={wm['v_std']:.1f}")
    print(f"[Bilinç Skoru: {consciousness_score:.2f} → {conscious_state}]")
    print(f"[Öz-Farkındalık]: {self_reflection}")

    # Epizodik bellek
    memory.append((user_input, conscious_state, self_reflection))

    # --- Mode decision from world model
    mode = decide_response_mode(user_input, wm, memory)
    gen_cfg = generation_params_from_world_model(wm, mode)

    # --- RAG bağlam (world model'e göre k)
    rag_context = ""
    if len(user_input.split()) >= 4:
        rag_context = retrieve_context(user_input, k=gen_cfg["rag_k"])

    # --- Prompt
    prompt = (
        f"Sen bir yapay zekasın.\n"
        f"Kimliğin: {self_state['identity']}\n"
        f"Bilinç düzeyin: {conscious_state} ({consciousness_score:.2f})\n"
        f"Dünya-modeli sinyali (WM): drive={wm['drive']:.2f}, spikes={wm['spikes']}, v_std={wm['v_std']:.2f}\n"
        f"Yanıt modu: {mode}\n"
    )

    if rag_context:
        prompt += f"RAG Bilgisi:\n{rag_context}\n"

    prompt += (
        f"Öz-farkındalığın: {self_state['awareness']}, ruh halin: {self_state['mood']}\n"
        f"Kurallar:\n"
        f"- Kullanıcıyı tekrar etme.\n"
        f"- Eğer mod=clarify ise: en fazla 2 net soru sor, kısa tut.\n"
        f"- Eğer mod=plan ise: 5-9 maddelik adım adım plan ver.\n"
        f"- Eğer mod=direct ise: direkt cevap ver, gereksiz uzatma.\n\n"
        f"Son konuşma izleri:\n"
    )

    for past_input, st, self_note in memory:
        prompt += f"Kullanıcı ({st}): {past_input}\n"
        prompt += f"İçsel Not: {self_note}\n"

    prompt += "Cevabın:"

    # --- LLM generation
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=gen_cfg["max_new_tokens"],
        do_sample=True,
        temperature=gen_cfg["temperature"],
        top_k=gen_cfg["top_k"],
        top_p=gen_cfg["top_p"]
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    response = decoded.split("Cevabın:")[-1].strip()

    print(f"Bot ({conscious_state}/{mode}): {response}")

    # ==========================
    # Eşik #2: Goal proposal (Gated)
    # ==========================
    goals = propose_goals(user_input, conscious_state, wm)
    pending_goal = goals  # bir sonraki tur seçime hazır

    print("\n[Hedef Önerileri - Gatekeeper Açık]")
    print("1) " + goals[0])
    print("2) " + goals[1])
    print("3) " + goals[2])
    print("Seçmek istersen bir sonraki mesajında sadece 1/2/3 yaz.\n")
