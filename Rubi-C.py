# === 1. GEREKLİ KÜTÜPHANELER ===
import builtins
from brian2 import *
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines import pipeline

# === 2. SNN: Talamokortikal Döngü ===
start_scope()

# Parametreler
tau = 10*ms
eqs = '''
dv/dt = (I - v)/tau : 1
I : 1
'''

# Thalamus
input_group = PoissonGroup(10, rates=100*Hz)

# V1 görsel korteks
v1 = NeuronGroup(10, eqs, threshold='v > 1', reset='v = 0', method='exact')
v1_monitor = SpikeMonitor(v1)

# Association cortex
assoc = NeuronGroup(5, eqs, threshold='v > 1', reset='v = 0', method='exact')
assoc_monitor = SpikeMonitor(assoc)

# Bağlantılar
syn_input_v1 = Synapses(input_group, v1, on_pre='v += 0.2')
syn_input_v1.connect()

syn_v1_assoc = Synapses(v1, assoc, on_pre='v += 0.4')
syn_v1_assoc.connect()

# Simülasyon
print("Talamokortikal döngü simülasyonu çalışıyor...")
run(1*second)

# Bilinç eşiği
conscious_threshold = 5

# === 3. BİLİNÇ ALGILANDIYSA: LLaMA 2 Tabanlı Chatbot ===
if assoc_monitor.num_spikes > conscious_threshold:
    print(f"\nBilinç algılandı! ({assoc_monitor.num_spikes} spike) LLaMA-2 başlatılıyor...\n")

    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
    model = AutoModelForCausalLM.from_pretrained(
        "NousResearch/Llama-2-7b-chat-hf",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    llama_chat = pipeline("text-generation", model=model, tokenizer=tokenizer)

    print("Sohbete hoş geldin! Çıkmak için 'çık' yaz.\n")

    try:
        while True:
            try:
                user_input = builtins.input("Sen: ")  # builtins üzerinden input çağrısı
            except EOFError:
                break  # Colab gibi ortamlar için

            if user_input.lower() in ["çık", "exit", "quit"]:
                print("Görüşmek üzere!")
                break

            prompt = f"[INST] {user_input} [/INST]"
            output = llama_chat(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)[0]["generated_text"]

            # Cevabı ayıkla
            if "[/INST]" in output:
                response = output.split("[/INST]")[-1].strip()
            else:
                response = output.strip()

            print("AI:", response)

    except KeyboardInterrupt:
        print("\nSohbet sonlandırıldı.")

else:
    print(f"\nBilinç algılanamadı. ({assoc_monitor.num_spikes} spike) Chatbot devreye girmedi.")
