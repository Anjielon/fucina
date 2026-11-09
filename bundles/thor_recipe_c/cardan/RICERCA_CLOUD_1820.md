# Ricerca cloud 3/10 18:20 — legame lettera (sintesi)
- Nessun lavoro su distorsione-lettera DOPO quantizzazione: letteratura mancante.
- BNP 2409.18857: direzione b (giuste vs sbagliate a rotazioni) su ~32 domande, azzera k nodi di lm_head che interagiscono con b (0,002% pesi); b si trasferisce fra dataset. Il +24,9% e' con AOI (vietata: cambia il prompt).
- PriDe 2309.03882: priore delle lettere sottratto a test-time -> NON lecito come risultato, utile come TETTO dell'esperimento E2 (diagnosi su dati non-MMLU).
- PIF 2406.01026: fine-tune con negativi (contenuti sbagliati legati a tutte le lettere), batte DPO e SFT; LoRA r16 a64, 3 epoche, lr 1e-5..2e-4.
- LoRA-1/PBM 2511.21709: coerenza fra permutazioni senza etichette. PA-GRPO 2603.21016 (RL, piu' pesante).
- Meccanismo 2407.15018: la scelta della lettera nasce in poche teste di attenzione degli strati CENTRALI -> lm_head corregge il priore ma non ricostruisce il legame: E1 deve allenare anche attenzione centrale, non solo FFN/esperti.
- Ricetta E2: ottimizzare direttamente le 4 righe ternarie (scambi greedy / STE), obiettivo KL sulle lettere del maestro (gia' cosi' in e2_riternarizza). BNP k 8..64.
- Ricetta E1: 20-40k domande x4 rotazioni; KL lettere maestro + coerenza rotazioni + negativi PIF peso ~0,3; teste attenzione centrali + lm_head; 1-2 epoche lr 1e-5..5e-5; selezione su LogiQA/MedMCQA heldout, MMLU una volta sola.
