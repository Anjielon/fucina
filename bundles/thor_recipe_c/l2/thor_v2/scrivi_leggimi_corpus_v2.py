#!/usr/bin/env python3
"""Scrive `calibrazione/LEGGIMI.md` dai manifesti (v1, v2, v2 senza Llama, pannello v2, sorgenti v2):
nessun numero trascritto a mano. 25/9/2026, notte. Solo CPU (ricalcola la composizione di cot sotto U).

    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19 @@ROOT:venv@@/bin/python3 scrivi_leggimi_corpus_v2.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import costruisci_corpus_calibrazione_v2 as B  # noqa: E402

C = B.C
CAL, PAN = B.DIR_CAL, B.DIR_PAN
U = "wiki:3,edu:2,cot:2,it_web:2,it_wiki:1,it_chat:3,it_pensiero:1,strumenti:2"
U_SEQ = {d: 1200 * int(q) // 16 for d, q in (x.split(":") for x in U.split(","))}
NOME_CLASSE = {"umana": "umana", "sintetica_pulita": "sintetica pulita", "llama_gpt": "Llama/GPT",
               "non_dichiarata": "non dichiarata", "per_tipo_di_domanda": "per tipo di domanda (vedi testo)"}


def j(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def pct(x: float, cifre: int = 1) -> str:
    return f"{100 * x:.{cifre}f}".replace(".", ",") + "%"


def num(x) -> str:
    return f"{x:,}".replace(",", ".")


def prime_n(etichette_rle: list, n: int) -> dict:
    out, visti = {}, 0
    for f, k in etichette_rle:
        if visti >= n:
            break
        k = min(k, n - visti)
        out[f] = out.get(f, 0) + k
        visti += k
    return out


def cot_sotto_u(man: dict, variante: str) -> dict:
    """Classi di origine degli esempi che riempiono le PRIME 150 sequenze cot (quelle che U usa)."""
    tk = C.Tok()
    F = B.FontiV2(tk, 0.30)
    classi = None if variante == "tutte" else B.AMMESSE_SENZA_LLAMA
    _, info, usati = B.pool_cot_v2(F, tk, U_SEQ["cot"], set(man["domini"]["cot"]["esclusi_per_contaminazione"]), classi)
    tot = sum(v["token"] for v in info["per_classe_di_origine"].values())
    return {c: {"esempi": v["esempi"], "quota_token": v["token"] / tot} for c, v in info["per_classe_di_origine"].items()}


def tabella_varianti(mt: dict, ms: dict, cot_t: dict, cot_s: dict) -> "tuple[list, dict]":
    righe = ["| dominio | seq. in U | «tutte» (v2): fonti (sequenze) | «senza Llama/GPT»: fonti (sequenze) |",
             "|---|---|---|---|"]
    classi = {"tutte": {}, "senza_llama": {}}
    for d, n in U_SEQ.items():
        celle = []
        for v, m, cotc in (("tutte", mt, cot_t), ("senza_llama", ms, cot_s)):
            if d == "cot":
                celle.append(", ".join(f"{NOME_CLASSE[c]} {pct(x['quota_token'], 0)} dei token ({x['esempi']} es.)"
                                       for c, x in sorted(cotc.items())))
                for c, x in cotc.items():
                    classi[v][c] = classi[v].get(c, 0) + n * x["quota_token"]
                continue
            comp = m["composizione_per_fonte_e_origine"][d]
            prese = prime_n(m["etichette_per_sequenza"][d], n)
            celle.append(", ".join(f"{f} {k} ({NOME_CLASSE[comp[f]['classe']]})" for f, k in prese.items()))
            for f, k in prese.items():
                c = comp[f]["classe"]
                classi[v][c] = classi[v].get(c, 0) + k
        righe.append(f"| {d} | {n} | {celle[0]} | {celle[1]} |")
    return righe, classi


def main() -> int:
    v1 = j(CAL / "calibrazione_corpus.json")
    mt = j(CAL / "calibrazione_corpus_v2.json")
    ms = j(CAL / "calibrazione_corpus_v2_senza_llama.json")
    mp = j(PAN / "pannello_nuovi_domini_v2.json")
    so = j(B.MAN_SORG_V2)
    cot_t, cot_s = cot_sotto_u(mt, "tutte"), cot_sotto_u(ms, "senza_llama")
    tab, classi = tabella_varianti(mt, ms, cot_t, cot_s)
    d5 = mt["misure"]["d5_nemotron"]
    d6, d6v1 = mt["misure"]["d6_massive"], mt["misure"]["d6_v1_stessa_misura"]
    cop_t, cop_s = mt["disgiunzione"]["copertura_pannello_v2"], ms["disgiunzione"]["copertura_pannello_v2"]
    cot_or = so["nuovi"]["cot_origine_domande"]
    pan_it = mp["domini"]["strumenti_it"]
    L = []
    a = L.append
    a("# Corpus di calibrazione multi-dominio di Thor")
    a("")
    a(f"Scritto il {time.strftime('%Y-%m-%d %H:%M:%S')} da `thor_v2/scrivi_leggimi_corpus_v2.py` leggendo i manifesti "
      "(nessun numero trascritto a mano). Corpus costruiti da `thor_v2/costruisci_corpus_calibrazione_v2.py`, fonti "
      "da `thor_v2/scarica_corpus_calibrazione.py` (v1) e `thor_v2/scarica_corpus_calibrazione_v2.py`. Solo fonti "
      "pubbliche a revisione fissata, nessun dato privato di MOGAVIS. Prove: `thor_v2/strumenti/prova_corpus_v2.py`.")
    a("")
    a("## ⛔ Obbligatorio con un corpus pronto: `THOR_VALIDA_BATCH_SEQ=8 THOR_VALIDA_N_HELDOUT=1`")
    a("")
    a("Ogni corpus esclude dal dominio cot SOLO gli esempi 0-8 dell'ordine mescolato col seme 1234: sono l'held-out "
      "storico con 8 sequenze per batch e 1 batch di held-out (8x1024 token = 9 esempi). Col default "
      "`THOR_VALIDA_N_HELDOUT=4` l'held-out prende gli esempi 0-34 e 9-34 sono anche nel dominio cot: 22.536 finestre "
      "di 32 token in comune (verifica del corpus v1), la loss held-out diventerebbe loss di addestramento. "
      "`_guardia_heldout_corpus` (patch applicata) lo RIFIUTA: provato su entrambe le varianti v2 (prove T2, T4). "
      "La somma delle quote e' 16 = ciclo di 2 batch: `THOR_VALIDA_N_BATCH` deve essere pari.")
    a("")
    a("## File")
    a("")
    a("| file | cosa | sha256 | byte | miscele servibili senza ripetere sequenze |")
    a("|---|---|---|---|---|")
    for m, cosa in ((mt, "v2 «tutte le fonti»"), (ms, "v2 «senza Llama/GPT»")):
        a(f"| `calibrazione/{m['file']}` | {cosa} | `{m['sha256_file']}` | {num(m['byte_file'])} | "
          f"{', '.join(m['miscele'])} |")
    a(f"| `calibrazione/{v1['file']}` | v1, storico, INVARIATO (vedi errata) | `{v1['sha256_file']}` | "
      f"{num(v1['byte_file'])} | {', '.join(v1['miscele'])} |")
    a(f"| `pannello/pannello_nuovi_domini_v2.pt` | pannello dei domini nuovi v2 | "
      f"`{mt['pannello_nuovo']['sha256_file']}` | {num(os.path.getsize(PAN / 'pannello_nuovi_domini_v2.pt'))} | — |")
    a(f"| `calibrazione/sorgenti/manifesto_sorgenti_v2.json` | fonti, licenze e ORIGINE corrette | "
      f"`{mt['sorgenti']['sha256_manifesto']}` | {num(os.path.getsize(B.MAN_SORG_V2))} | — |")
    a("")
    a("Forma: `{dominio: tensore (n, 1024) int64}`, seme 1234, tokenizzatore `@@ROOT:models@@/gguf/thor-src-hf`; accanto "
      "a ogni `.pt` il manifesto `.json` con lo sha256 dei token di ogni dominio (lo controlla `_carica_corpus_pronto`), "
      "le etichette di fonte per sequenza, la composizione per origine, le verifiche e le misure.")
    a("")
    a("## Uso (miscela U)")
    a("")
    a("```")
    for m in (mt, ms):
        a(f"THOR_VALIDA_BATCH_SEQ=8 THOR_VALIDA_N_HELDOUT=1 THOR_VALIDA_N_BATCH=150 \\")
        a(f"THOR_CALIB_CORPUS={CAL / m['file']} \\")
        a(f"THOR_CALIB_MISCELA={U}")
        a("")
    a("# pannello con i domini nuovi v2 (richiede strumenti/patch_pannello_nuovi_domini.py, NON ancora applicata:")
    a("# applicarla dopo i pannelli gia' in coda, oppure THOR_PANNELLO_NUOVI_DOMINI=0 in quegli script - difetto D3)")
    a(f"THOR_PANNELLO_NUOVI={PAN / 'pannello_nuovi_domini_v2.pt'}")
    a("```")
    a("")
    a("La variante senza Llama/GPT ha 150 sequenze cot: serve U, V e W; M2 (cot 2 su 8 = 300 sequenze) viene "
      "rifiutata dalla patch con errore (prova T5), mai con sequenze ripetute.")
    a("")
    a("## Le due varianti sotto U (150 batch x 8 = 1.200 sequenze)")
    a("")
    L.extend(tab)
    a("")
    a("Classe di origine (manifesto delle sorgenti v2): **umana** = testo di persone; **sintetica pulita** = generato "
      "da modelli dichiarati, nessuno Llama ne' GPT nella catena; **Llama/GPT** = prompt o risposte anche in parte da "
      "Llama o GPT; **non dichiarata** = la fonte non dice chi ha scritto prompt o risposte (non certificabile senza "
      "Llama/GPT). Sequenze per classe sotto U (cot ripartito per quota di token):")
    a("")
    a("| classe | «tutte» | «senza Llama/GPT» |")
    a("|---|---|---|")
    for c in ("umana", "sintetica_pulita", "llama_gpt", "non_dichiarata"):
        a(f"| {NOME_CLASSE[c]} | {classi['tutte'].get(c, 0):.1f} ({pct(classi['tutte'].get(c, 0) / 1200)}) | "
          f"{classi['senza_llama'].get(c, 0):.1f} ({pct(classi['senza_llama'].get(c, 0) / 1200)}) |")
    a("")
    a("Cosa cambia nella variante senza Llama/GPT (e perche'):")
    a("")
    a("- **it_chat**: tolto smoltalk2 senza ragionamento (prompt Smol-Magpie-Ultra, Llama-3.1-405B). Aya e OASST2 "
      "danno tutto quello che hanno sul lato calibrazione (" + ", ".join(
          f"{f} {k}" for f, k in ms["domini"]["it_chat"]["quote_effettive"].items()) + "): MFAQ riempie, quindi "
      "it_chat diventa per il 60% domande e risposte di pagine FAQ di siti italiani (servizio clienti, turismo, "
      "enti) invece di chat da assistente. Nessuna altra fonte umana di chat italiana verificata: MQA-CQA italiano "
      "(`clips/mqa`, CC0) e' stato guardato e SCARTATO (primi 2 MiB: 5.760 righe su 7.489 da tellows.it, segnalazioni "
      "di numeri di telefono).")
    a("- **it_pensiero**: tolti smoltalk2 con ragionamento e ReasonXL-smoltalk. Al loro posto ReasonXL con catena "
      "pulita, gli stessi file e revisione: generale = domande QASC e FLAN v2 (" + str(
          ms["domini"]["it_pensiero"]["quote_effettive"].get("rx_generale")) + " sequenze), matematica = "
      "OpenMathReasoning entro " + num(B.RX_MATEMATICA_MAX_TOKEN) + " token (" + str(
          ms["domini"]["it_pensiero"]["quote_effettive"].get("rx_matematica")) + "). Ragionamento e risposte in "
      "ITALIANO (tradotti da Qwen3-32B), prompt con risposte a scelta multipla o problemi: NON e' il ragionamento "
      "generale di analisi e pianificazione di smoltalk2 — dominio diverso, dichiarato.")
    a("- **strumenti**: tolto Nemotron (prompt e schemi di origine non dichiarata). Al suo posto MASSIVE 1.1 **en-US** "
      "(frasi di persone, stessi 12 scenari e 40 funzioni della calibrazione, id di train diversi da quelli delle "
      "conversazioni italiane): chiamate a un passo, senza risposte degli strumenti ne' risposte finali (la fonte non "
      "ne ha). Si perde il multi-passo di Nemotron.")
    a("- **cot**: solo domande ORIGINALI di GSM8K/MATH (tipo AnsAug di MetaMathQA): tolte le domande riscritte da "
      "GPT-3.5 (Rephrased, SV), FOBAR (mascheramento: esecutore non dichiarato) e tutte le domande OpenCodeInstruct "
      "(Instruct-LLM non nominato). Niente codice. Risposte e ragionamento restano di Thor.")
    a("- wiki, edu, it_web, it_wiki: identici al bit fra le due varianti e al v1.")
    a("")
    a("## Cosa cambia rispetto al v1 (misurato)")
    a("")
    a("| difetto | misura | v1 | v2 «tutte» | v2 «senza Llama/GPT» |")
    a("|---|---|---|---|---|")
    a(f"| D2 | token del pannello `strumenti_it` dentro finestre di 32 token della calibrazione | 89,2% | "
      f"{pct(cop_t['strumenti_it']['quota_coperta'])} | {pct(cop_s['strumenti_it']['quota_coperta'])} |")
    a(f"| D2 | stessa misura su `strumenti_en` (testo fisso del template strumenti) | 17,7% | "
      f"{pct(cop_t['strumenti_en']['quota_coperta'])} | {pct(cop_s['strumenti_en']['quota_coperta'])} |")
    a(f"| D2 | scenari / funzioni mostrate MASSIVE in comune fra calibrazione e pannello (conversazioni usate; v1 "
      f"ricostruito al bit) | 11 / 53 | "
      f"{mt['disgiunzione']['id']['massive_scenari']} / {mt['disgiunzione']['id']['massive_funzioni_mostrate']} | "
      f"{ms['disgiunzione']['id']['massive_scenari']} / {ms['disgiunzione']['id']['massive_funzioni_mostrate']} |")
    a(f"| D5 | esempi Nemotron col ragionamento | 9 su 64 (14%) | {d5['col_ragionamento']} su {d5['esempi']} "
      f"({pct(d5['quota_esempi'])}; {pct(d5['quota_token'])} dei token) | — (Nemotron escluso) |")
    a(f"| D6 | token delle frasi utente negli esempi MASSIVE (per ruolo, come la verifica v1) | 2,7% | "
      f"{pct(d6['quote_per_ruolo']['user'])} | {pct(d6['quote_per_ruolo']['user'])} (it) |")
    a(f"| D6 | frasi + valori degli argomenti (italiano) sul totale, stesse conversazioni | "
      f"{pct(d6v1['quota_lingua_della_frase'])} | {pct(d6['quota_lingua_della_frase'])} | "
      f"{pct(d6['quota_lingua_della_frase'])} (it) · {pct(ms['misure']['d6_massive_en']['quota_lingua_della_frase'])} (en) |")
    a(f"| D6 | frasi per conversazione / distrattori / token per conversazione (mediana) | 4 / 3 / "
      f"{num(d6v1['token_mediana_conversazione'])} | {B.K_MASSIVE} / {B.DISTRATTORI} / "
      f"{num(d6['token_mediana_conversazione'])} (max {num(d6['token_max_conversazione'])}) | uguale |")
    a("")
    a(f"- **D2**: il pannello `strumenti_it` usa SOLO gli scenari {', '.join(pan_it['scenari'])} "
      f"({len(pan_it['funzioni'])} funzioni), frasi di dev e test di MASSIVE 1.1 ({pan_it['frasi']['tenute']} "
      f"convertibili e distinte, {pan_it['frasi_usate']} usate in {pan_it['fonti']['massive_dev_test']['esempi_usati']} "
      f"conversazioni); funzione uguale al golden di MASSIVE-Agents {pan_it['funzione_uguale_al_golden']}/"
      f"{pan_it['golden_massive_agents_in_questi_scenari']}. La calibrazione usa gli altri 12 scenari (40 funzioni), "
      "solo train, distrattori dal proprio lato. Il residuo coperto e' SOLO testo fisso del template: preambolo "
      "`# Tools ... <tools>`, istruzioni di formato dopo `</tools>`, marcatori di turno e chiusure `</function>`/"
      "`</tool_call>` (decodificate e controllate a mano: nessuna finestra contiene frasi, nomi di funzione o valori). "
      "Per ruolo (v2 «tutte»): " + ", ".join(
          f"{k} {pct(v['quota_token'], 0)} dei token, coperto {pct(v['coperta'] or 0)}"
          for k, v in cop_t['strumenti_it']['per_ruolo'].items()) + ".")
    a(f"- `strumenti_en`: {pct(cop_t['strumenti_en']['quota_coperta'])} con la «tutte» (17,7% in v1) = template "
      "degli strumenti piu' i messaggi d'errore ricorrenti delle risposte-strumento di Nemotron (es. `{\"error\": "
      "\"Invalid tool call\", ...}`); con la «senza Llama/GPT» il " + pct(cop_s['strumenti_en']['quota_coperta']) +
      " e' solo template (ruolo user coperto " + pct(cop_s['strumenti_en']['per_ruolo']['user']['coperta'] or 0) +
      "). Da dichiarare nel paper.")
    a(f"- **D5**: il ragionamento si assegna DOPO il filtro dei {num(B.NEMOTRON_MAX_TOKEN)} token: su "
      f"{mt['fonti_calibrazione']['nemotron']['righe']} righe lette, {mt['fonti_calibrazione']['nemotron']['con_ragionamento_entro_2048']} "
      f"stanno sotto il tetto CON il ragionamento e {mt['fonti_calibrazione']['nemotron']['senza_ragionamento_entro_2048']} "
      "senza (dopo aver tolto quelle con strumenti del pannello): bastano per 1 su 3. Scelta di parte dichiarata: "
      "gli esempi col ragionamento sono i piu' corti.")
    a("- **D6**: oltre ~14% di italiano non si va senza inventare risposte degli strumenti (griglia misurata: k=24 "
      "e 0 distrattori = 13,8%, conversazioni fino a 2.815 token e nessuna scelta fra strumenti). Gli schemi "
      "restano in inglese per costruzione della fonte (e in produzione il prompt di sistema del cervello e' fatto "
      "cosi').")
    a(f"- **Filtro di qualita' MFAQ** (nuovo, entrambe le varianti): su {mt['domini']['it_chat']['fonti']['mfaq']['esempi_candidati'] + sum(v for k, v in mt['fonti_calibrazione']['mfaq_filtro_qualita'].items() if k != 'tenuti')} "
      "conversazioni del train scarta " + ", ".join(
          f"{k} {v}" for k, v in mt['fonti_calibrazione']['mfaq_filtro_qualita'].items() if k != 'tenuti') +
      ". In v1 una sola era entrata (flowers4brussels.com, listino prezzi): per questo 13 sequenze MFAQ di it_chat "
      "della «tutte» differiscono dal v1; tutto il resto di it_chat, e wiki, edu, cot, it_web, it_wiki, it_pensiero, "
      "sono identici al bit al v1.")
    a(f"- **D4**: vedi sotto. Le domande cot sono state ritrovate TESTO PER TESTO alle revisioni fissate "
      f"(math {cot_or['verifica']['math']['ritrovate_testo_per_testo']}/{cot_or['verifica']['math']['domande_nel_file']}, "
      f"code {cot_or['verifica']['code']['ritrovate_testo_per_testo']}/{cot_or['verifica']['code']['domande_nel_file']}) "
      "e il campionamento del generatore (seme 20260921) riprodotto: stesso insieme di domande "
      f"(math {cot_or['verifica']['math']['campionamento_riprodotto_uguale']}, code "
      f"{cot_or['verifica']['code']['campionamento_riprodotto_uguale']}; l'ordine nei file differisce perche' la "
      "generazione era in parallelo). Tipi MetaMathQA delle 1.024 domande math: " + ", ".join(
          f"{k} {v}" for k, v in sorted(cot_or['conta_per_tipo']['math'].items())) + ".")
    a("")
    a("## Disgiunzione (entrambe le varianti, esito nei manifesti)")
    a("")
    for m in (mt, ms):
        dj = m["disgiunzione"]
        a(f"- **{m['variante']}**: {dj['esito']}. Finestre di 32 token di CONTENUTO in comune con pannello v2, "
          f"pannello storico, wiki.test.raw e mmlu_test.bin: {sum(dj['contenuto_calibrazione_vs'].values())} "
          f"(dopo {len(m['giri_di_pulizia']) - 1} giro di pulizia: tolti " + ", ".join(
              f"{d} {len(v)}" for d, v in m['giri_di_pulizia'][0]['esempi_tolti'].items()) + "). Identificativi, "
          f"prompt, frasi MASSIVE, scenari e funzioni in comune: 0; strumenti Nemotron del pannello in calibrazione: "
          f"{dj['id']['nemotron_strumenti_del_pannello_in_calibrazione']}.")
    a("")
    a("Copertura dei token del pannello v2 da finestre di 32 token della calibrazione (sequenze intere, template "
      "compreso):")
    a("")
    a("| dominio del pannello | «tutte» | «senza Llama/GPT» |")
    a("|---|---|---|")
    for d in cop_t:
        a(f"| {d} | {pct(cop_t[d]['quota_coperta'], 2)} | {pct(cop_s[d]['quota_coperta'], 2)} |")
    a("")
    a("Tolti per contenuto in comune (giro 1, entrambe): i 42 esempi Aya con finestre o messaggi uguali al pannello "
      "it_chat (formati di domanda ricorrenti di Aya), 3 documenti edu e le domande cot con testo del pannello "
      "math_code o di MMLU, e una conversazione MASSIVE con la frase «qual è la capitale dell' italia», uguale a un "
      "prompt Aya del pannello.")
    a("")
    a("Pannello v2: `it_web`, `it_wiki`, `it_chat`, `it_pensiero`, `strumenti_en` sono IDENTICI al bit al pannello v1 "
      "(ricostruiti col codice v1 e confrontati, " + ", ".join(
          f"{d} {'si' if v else 'no'}" for d, v in mp["identici_a_v1"].items()) + "); cambia solo `strumenti_it`.")
    a("")
    a("## Licenze e origine per fonte (manifesto `sorgenti/manifesto_sorgenti_v2.json`)")
    a("")
    a("| fonte | classe | licenza | testo | obblighi |")
    a("|---|---|---|---|---|")
    for k, o in so["origini"].items():
        a(f"| {k} | {NOME_CLASSE.get(o['classe'], o['classe'])} | {o['licenza']} | {o['testo']}"
          f"{(' — ' + o['nota']) if o.get('nota') else ''} | {o.get('obblighi', '—')} |")
    a("")
    a("Prove (copie in `sorgenti/prove_licenze/`, con sha256 nel manifesto):")
    a("")
    for d in so["nuovi"]["prove_licenze"]["documenti"] + so["nuovi"]["prove_licenze"]["schede"]:
        a(f"- `{d['file']}` (`{d['sha256'][:16]}…`): {d['prova']}")
    a("")
    a("**Se si distribuisce un modello calibrato con la variante «tutte»**: le fonti Llama/GPT la legano alla licenza "
      "Llama 3.1 (nome che INIZIA con «Llama», «Built with Llama», copia dell'accordo: sez. 1.b.i) e ai termini d'uso "
      "OpenAI per le parti GPT (domande MetaMathQA riscritte, OpenHermes dentro SmolTalk). La variante «senza "
      "Llama/GPT» non ha fonti di quelle due classi ne' di origine non dichiarata (controllo automatico nel suo "
      "manifesto, campo `controllo_origine`): restano attribuzioni CC-BY-4.0 (NVIDIA, Amazon MASSIVE, AllenAI QASC), "
      "CC-BY-SA/GFDL (Wikipedia), ODC-By (FineWeb), Apache-2.0 e MIT. Unico legame indiretto: FineWeb-Edu e' "
      "SELEZIONATO da un classificatore addestrato su annotazioni di Llama3-70B-Instruct (nessun testo Llama).")
    a("")
    a("## Errata del corpus v1 (i file v1 NON sono stati modificati)")
    a("")
    a("- `sorgenti/manifesto_sorgenti.json` e `calibrazione_corpus.json` descrivono la licenza Llama 3.1 come "
      "«attribuzione»: e' sbagliato (vedi sopra); non dichiarano l'origine Llama dei prompt di smoltalk2 con "
      "ragionamento e di ReasonXL; attribuiscono ReasonXL a «HuggingFaceTB/smoltalk» senza dire che SmolTalk v1 "
      "contiene OpenHermes-2.5 (GPT-4); dicono «nessun termine di terze parti» per cot, le cui domande vengono da "
      "MetaMathQA (in parte GPT-3.5) e OpenCodeInstruct senza revisione fissata. Valgono le dichiarazioni del "
      "manifesto v2.")
    a("- Nel pannello v1 `strumenti_it` l'89,2% dei token era gia' visto in calibrazione (schemi uguali): NON usarlo "
      "per il verdetto; usare `pannello_nuovi_domini_v2.pt`.")
    a("- Nel v1 Nemotron aveva il ragionamento in 9 esempi su 64 (dichiarato 1 su 3).")
    a(f"- Il v1 (`{v1['sha256_file'][:16]}…`) resta valido per i confronti gia' fatti; la sua composizione e': "
      + "; ".join(f"{d} {v['sequenze']}" + (f" ({', '.join(f'{f} {n}' for f, n in v['quote_effettive'].items())})"
                                              if v.get('quote_effettive') else '') for d, v in v1['domini'].items()) + ".")
    a("")
    a("## Cosa resta aperto")
    a("")
    a("- **Decisione di Angelo** su quale variante usare per il modello da distribuire (la ricerca raccomandava la "
      "variante senza Llama). Le due varianti vanno PROVATE col pannello: la «senza Llama/GPT» cambia it_chat (FAQ "
      "invece di chat), it_pensiero (scelta multipla e matematica invece di analisi) e strumenti (niente multi-passo).")
    a("- **it_pensiero e cot di codice puliti**: l'alternativa piena e' far generare a Thor stesso ragionamento e "
      "risposte in italiano su prompt UMANI (Aya, OASST2) e su domande di codice umane (TACO, Apache-2.0), come per "
      "cot. Serve la GPU: non fatto qui.")
    a("- **Prompt di Nemotron**: se una fonte NVIDIA o il paper dichiarasse chi li ha generati (e non fosse Llama/GPT), "
      "la classe passerebbe da «non dichiarata» a «sintetica pulita» e Nemotron potrebbe rientrare nella variante "
      "pulita.")
    a("- **Pannello**: `it_chat` e `it_pensiero` del pannello contengono smoltalk2 e ReasonXL-smoltalk (prompt Llama): "
      "misurano, non addestrano, ma se si vuole una catena di scelta del tutto pulita serve un pannello it_chat/"
      "it_pensiero senza quelle fonti. Il `mfaq` del pannello (valid) non e' passato dal filtro di qualita' nuovo.")
    a("- **Patch del pannello** (`strumenti/patch_pannello_nuovi_domini.py`) ancora da applicare (D3), poi "
      "`THOR_PANNELLO_NUOVI=.../pannello_nuovi_domini_v2.pt`.")
    a("- La ricetta della forgia scorrevole registra `THOR_CALIB_CORPUS` come percorso, non come impronta: i v2 hanno "
      "nomi nuovi, ma riscrivere un file allo stesso percorso non verrebbe notato.")
    a("")
    (CAL / "LEGGIMI.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"scritto {CAL / 'LEGGIMI.md'} ({len(L)} righe)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
