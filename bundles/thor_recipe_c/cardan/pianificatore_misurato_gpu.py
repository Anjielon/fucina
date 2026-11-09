"""Esecutore comune, decisioni pesate e telemetria continua.

Nessuna esplorazione di compresenza con corse protette. Senza tempi misurati
si calibra UNA unita' isolata; il parallelismo ignoto resta vietato.
"""
from __future__ import annotations

import dataclasses
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import shutil
import time

import cardan_obiettivo_gpu as O
from cardan_telemetria_gpu import Telemetria
import pianificatore_gpu as PG


def carica_rendimenti(spec, storia, firma):
    """La storia e' uno snapshot completo, non un delta del profilo iniziale."""
    avvii = dict(spec.get("avvio_s", {}))
    campioni = spec.get("tempi", [])
    if storia.exists():
        h = json.loads(storia.read_text())
        if h["firma"] != firma:
            raise ValueError("rendimenti di un'altra ricetta")
        campioni = h["tempi"]
        avvii.update(h.get("avvio_s", {}))
    return O.Tempi(campioni), avvii


def ondata_calibrazione(lavori, posti, massimo):
    """Prova esplicita, distinta da una decisione di produzione certificata."""
    slots = [d for d, n in sorted(posti.items()) for _ in range(min(n, massimo))]
    onda = [(u.chiave, d) for u, d in zip(sorted(lavori, key=lambda u: u.chiave), slots)]
    return {"criterio": "misura_esplicita_non_ottimizzazione",
            "alternative": [], "scelta": {"ondate": [onda], "somma_pesata": None,
                                         "calibrazione_compresenza": True}}


def esegui(unita, factory, config, root, firma, posti, budget):
    # Riusa il worker transazionale con pipe/PDEATHSIG, non il vecchio loop.
    import pianificatore_guardia_gpu as G
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if len({u.chiave for u in unita}) != len(unita):
        raise ValueError("unita' duplicate")
    (root / "unita").mkdir(exist_ok=True)
    piano = {"firma": firma, "unita": [dataclasses.asdict(u) for u in unita],
             "factory": factory, "config": config}
    meta = root / "piano.json"
    if meta.exists():
        if json.loads(meta.read_text()) != piano:
            raise RuntimeError("piano/config/input diversi; scegliere una nuova --out")
    else:
        if any((root / "unita").iterdir()):
            raise RuntimeError("unita' senza piano: non le adotto")
        PG.scrivi_json(meta, piano)
    pendenti = {u.chiave: u for u in unita
                if not PG.completa(root / "unita" / u.chiave, firma, u)}
    if not pendenti:
        return
    if shutil.disk_usage(root).free < budget.disco * len(pendenti) + PG.GIB:
        raise RuntimeError("spazio insufficiente per risultati + 1 GiB")
    if not posti or any(type(n) is not int or n < 0 for n in posti.values()) or not sum(posti.values()):
        raise ValueError("nessun posto valido per lavoratori")
    spec = config.get("pianificazione", {})
    storia = root / "rendimenti_misurati.json"
    tempi, avvii = carica_rendimenti(spec, storia, firma)
    scaduti = set()
    scad_min = float(spec.get("scadenza_min_s", os.environ.get("FORGIA_SCADENZA_MIN_S", 1800)))
    scad_fatt = float(spec.get("scadenza_fattore", os.environ.get("FORGIA_SCADENZA_FATTORE", 4)))

    def scadenza(classe):
        # max(minimo, fattore x il piu' lungo tempo misurato per la classe, in qualunque compresenza)
        misurati = [s for (_, _, c), vv in tempi.dati.items() if c == classe for s in vv]
        return max(scad_min, scad_fatt * max(misurati)) if misurati else scad_min
    lavori = {}
    for u in pendenti.values():
        classe = spec.get("classi", {}).get(u.chiave)
        if classe is None:
            # Non trasferire picchi/tempi fra strati diversi senza misura.
            classe = u.chiave
        gruppo = spec.get("lavori", {}).get(u.chiave, u.chiave)
        peso = float(spec.get("pesi", {}).get(gruppo, spec.get("priorita", 1)))
        lavori[u.chiave] = O.Lavoro(u.chiave, classe, gruppo, peso)
    ctx = mp.get_context("spawn")
    figli, invii, ricezioni, devices = {}, {}, {}, {}
    attivi, contesti, inizi = {}, {}, {}
    telemetria = Telemetria()
    errore = None
    pausa = None
    pausa_da = None
    sani = 0
    blocco_device = None
    richiesta_host = False
    richiesta_cpu = 0
    invalidati = set()
    intervallo = float(spec.get("intervallo_risorse_s", 1))
    attesa_massima = float(spec.get("attesa_massima_s", 0))
    campioni_ripresa = spec.get("campioni_ripresa", 2)
    if (not 0 < intervallo <= 60 or not 0 <= attesa_massima < float("inf") or
            type(campioni_ripresa) is not int or campioni_ripresa < 2):
        raise ValueError("intervallo/attesa/campioni_ripresa non validi")
    ultimo_campione = 0.0
    journal = open(root / "decisioni.jsonl", "a", buffering=1)

    def registra(tipo, **dati):
        r = {"evento": tipo, "ora": time.time(), **dati}
        testo = json.dumps(r, allow_nan=False)
        journal.write(testo + "\n")
        if tipo != "risorse":
            print(testo, flush=True)

    def blocca(e):
        nonlocal errore
        if errore is None:
            errore = e
            registra("DRAIN", errore=str(e), in_volo=list(attivi.values()),
                     azione="nessuna nuova assegnazione; finire le unita' attive")

    def guardia():
        PG.guardia_corse()
        PG.guardia_host(*PG.misura_host(.05))

    def sospendi(e):
        nonlocal pausa, pausa_da, sani
        if pausa is None:
            pausa_da = time.monotonic()
            registra("ATTESA", motivo=str(e), in_volo=list(attivi.values()),
                     azione="finire unita' in volo; nessun nuovo avvio o assegnazione")
        pausa, sani = e, 0
        invalidati.update(attivi)

    def ammissione():
        try:
            guardia()
        except RuntimeError as e:
            sospendi(e)
            return False
        return pausa is None

    def controlla(forza=False):
        nonlocal ultimo_campione, pausa, pausa_da, sani, blocco_device, richiesta_host, richiesta_cpu
        distanziato = time.monotonic() - ultimo_campione >= intervallo
        if not forza and not distanziato:
            return None
        ultimo_campione = time.monotonic()
        s = telemetria.campiona()
        if spec.get("code_lavoratore") is not None:
            import cardan_code_gpu as CQ
            s["code_gpu"] = {str(g): v for g, v in CQ.uso()[0].items()}
        registra("risorse", **s)
        try:
            guardia()
            PG.guardia_host(s["mem_available"], s["swap_in_byte_s"])
            minimo_cpu = max(richiesta_cpu, budget.fili + budget.fili_io)
            if not attivi and int(s["cpu_liberi_stima"]) < minimo_cpu:
                raise RuntimeError("AMMISSIONE NEGATA: CPU libere esaurite")
            if blocco_device is not None:
                import torch
                if torch.cuda.mem_get_info(blocco_device)[0] < budget.gpu_prenotata + PG.GIB:
                    raise RuntimeError("AMMISSIONE NEGATA: VRAM libera insufficiente")
            if (richiesta_host or not figli) and s["mem_available"] - budget.host_prenotato(
                    spec.get("uma", True)) < PG.RISERVA_HOST:
                raise RuntimeError("AMMISSIONE NEGATA: picco del nuovo worker oltre riserva host")
        except RuntimeError as e:
            sospendi(e)
        else:
            if pausa is not None and distanziato:
                sani += 1
                if sani >= campioni_ripresa:
                    registra("RIPRESA", campioni_sani=sani,
                             attesa_s=time.monotonic() - pausa_da)
                    pausa, pausa_da, sani, blocco_device = None, None, 0, None
                    richiesta_host, richiesta_cpu = False, 0
        return s

    def attendi():
        while pausa is not None and errore is None:
            if attesa_massima and time.monotonic() - pausa_da >= attesa_massima:
                raise RuntimeError(f"attesa massima superata: {pausa}")
            # Dopo il drain restituisce RAM/device se la pressione persiste.
            if not attivi and time.monotonic() - pausa_da >= 5:
                for pid in list(figli):
                    ritira(pid)
            time.sleep(intervallo)
            controlla()

    def ritira(pid):
        if figli[pid].is_alive():
            invii[pid].send(None)
        figli[pid].join(5)
        if figli[pid].is_alive():
            raise RuntimeError(f"lavoratore inattivo {pid} non esce; nessun kill di unita' attiva")
        invii.pop(pid).close()
        ricezioni.pop(pid).close()
        figli.pop(pid)
        devices.pop(pid)

    def avvia(d):
        nonlocal blocco_device, richiesta_host
        if not ammissione():
            return None
        disponibile, swap = PG.misura_host(.05)
        try:
            PG.guardia_host(disponibile, swap)
        except RuntimeError as e:
            sospendi(e)
            return None
        if disponibile - budget.host_prenotato(spec.get("uma", True)) < PG.RISERVA_HOST:
            richiesta_host = True
            sospendi(RuntimeError("AMMISSIONE NEGATA: picco del nuovo worker oltre riserva host"))
            return None
        if spec.get("gpu_reale"):
            import torch
            libera, totale = torch.cuda.mem_get_info(d)
            registra("gpu_ammissione", device=d, libera_byte=libera, totale_byte=totale,
                     richiesta_byte=budget.gpu_prenotata + PG.GIB)
            if libera < budget.gpu_prenotata + PG.GIB:
                blocco_device = d
                sospendi(RuntimeError(f"AMMISSIONE NEGATA: VRAM libera insufficiente per nuovo worker GPU {d}"))
                return None
            # Priorita' 0 (21/10): code hardware KFD misurate ADESSO (tutti i processi, corse vive comprese).
            cw = spec.get("code_lavoratore")
            if cw is not None:
                import cardan_code_gpu as CQ
                n_code = 0 if cw == "non_misurate" and any(dd == d for dd in devices.values()) else (
                    PG.limite_code_non_misurate(d) if cw == "non_misurate" else
                    CQ.ammissibili(d, cw["compute"], cw["sdma"]))
                registra("code_ammissione", device=d, code_lavoratore=cw, ammissibili=n_code,
                         stato=CQ.stato())
                if n_code is not None and n_code < 1:
                    blocco_device = d
                    sospendi(RuntimeError(f"AMMISSIONE NEGATA: code GPU esaurite per nuovo worker GPU {d}"))
                    return None
        ingresso, invio = ctx.Pipe(duplex=False)
        ricezione, uscita = ctx.Pipe(duplex=False)
        p = ctx.Process(target=G._lavoratore, args=(
            factory, config, d, str(root), firma, ingresso, uscita, budget.fili, os.getpid()))
        t0 = time.monotonic()
        p.start()
        ingresso.close()
        uscita.close()
        pid = p.pid
        figli[pid], invii[pid], ricezioni[pid], devices[pid] = p, invio, ricezione, d
        while True:
            controlla()
            if pausa is not None:
                registra("avvio_annullato", pid=pid, device=d, motivo=str(pausa),
                         unita_assegnate=0)
                if p.is_alive():
                    p.terminate()
                    p.join(5)
                if p.is_alive():
                    raise RuntimeError(f"lavoratore {pid} non esce dal caricamento annullato")
                ritira(pid)
                return None
            if ricezione.poll(.1):
                try:
                    stato, pp, msg = ricezione.recv()
                except EOFError as e:
                    raise G.ErroreLavoratore(f"pipe chiusa all'avvio: {pid}") from e
                if stato != "pronto" or pp != pid:
                    raise G.ErroreLavoratore(f"avvio {pid}: {msg}")
                avvii[str(d)] = time.monotonic() - t0
                registra("avvio", pid=pid, device=d, secondi=avvii[str(d)])
                return pid
            if p.exitcode is not None:
                raise G.ErroreLavoratore(f"lavoratore {pid} morto all'avvio: {p.exitcode}")

    try:
        time.sleep(intervallo)
        s = controlla(True)
        attendi()
        limiti = dict(posti)
        while pendenti and errore is None:
            attendi()
            s = controlla(True)
            if pausa is not None:
                continue
            liberi_cpu = int(s["cpu_liberi_stima"]) // (budget.fili + budget.fili_io)
            if liberi_cpu < 1:
                sospendi(RuntimeError("AMMISSIONE NEGATA: CPU libere esaurite, nessuna nuova ondata"))
                continue
            residenti = {d: sum(dd == d for dd in devices.values()) for d in limiti}
            if spec.get("calibra_compresenza"):
                richiesti = min(len(pendenti), sum(limiti.values()))
                if richiesti > liberi_cpu:
                    richiesta_cpu = richiesti * (budget.fili + budget.fili_io)
                    sospendi(RuntimeError("AMMISSIONE NEGATA: CPU insufficienti per calibrazione richiesta"))
                    continue
                scelta = ondata_calibrazione([lavori[k] for k in pendenti], limiti,
                                             len(pendenti))
            else:
                scelta = O.scegli([lavori[k] for k in pendenti], limiti, tempi, avvii, residenti,
                                 max_lavoratori=liberi_cpu)
            registra("decisione", **scelta)
            onda = scelta["scelta"]["ondate"][0]
            richiesti = {d: sum(dd == d for _, dd in onda) for d in limiti}
            for d, n in richiesti.items():
                presenti = [pid for pid in figli if devices[pid] == d]
                for pid in presenti[n:]:
                    ritira(pid)
                for _ in range(max(0, n - len(presenti))):
                    if avvia(d) is None:
                        break
            if pausa is not None:
                continue
            liberi = list(figli)
            ctx_onda = O.contesto([(d, lavori[k].classe) for k, d in onda])
            inizio_onda = time.monotonic()
            for k, d in onda:
                if not ammissione():
                    break
                pid = next(p for p in liberi if devices[p] == d)
                liberi.remove(pid)
                attivi[pid], contesti[pid] = k, ctx_onda
                inizi[pid] = time.monotonic()
                try:
                    invii[pid].send(pendenti[k])
                except BrokenPipeError:
                    del attivi[pid]
                    blocca(G.ErroreLavoratore(f"pipe chiusa durante assegnazione: {pid}/{k}"))
                    break
            while attivi:
                controlla()
                pronti = wait([ricezioni[p] for p in attivi], timeout=.1)
                for conn in pronti:
                    pid = next(p for p in attivi if ricezioni[p] is conn)
                    k = attivi[pid]
                    try:
                        stato, pp, msg = conn.recv()
                    except EOFError:
                        blocca(G.ErroreLavoratore(f"pipe chiusa: {pid}, unita' {k}"))
                        del attivi[pid]
                        continue
                    if stato != "fatta" or pp != pid or msg["chiave"] != k:
                        blocca(G.ErroreLavoratore(f"lavoratore {pid}: {msg}"))
                        del attivi[pid]
                        continue
                    try:
                        if not PG.completa(root / "unita" / k, firma, pendenti[k]):
                            raise RuntimeError(f"completamento assente: {k}")
                    except (RuntimeError, OSError, ValueError) as e:
                        blocca(e)
                    # D17: durata dall'avvio della PROPRIA unita' (con il riempimento non coincide
                    # con l'inizio dell'ondata); lo stesso contesto di compresenza resta valido.
                    wall = time.monotonic() - inizi.pop(pid, inizio_onda)
                    # Il modello usa completamenti di ondate intere, non
                    # finti campioni di una velocita' a contesa costante.
                    ammesso = errore is None and pausa is None and pid not in invalidati
                    if ammesso:
                        tempi.aggiungi(contesti[pid], devices[pid], lavori[k].classe, wall)
                    registra("completamento", pid=pid, chiave=k, contesto=contesti[pid],
                             tipo_campione="completamento_wall_da_inizio_ondata",
                             campione_ammesso=ammesso, wall_s=wall, misure=msg)
                    del attivi[pid], pendenti[k]
                    invalidati.discard(pid)
                    PG.scrivi_json(storia, {"firma": firma, "tempi": tempi.esporta(), "avvio_s": avvii})
                    if PG.picco_oltre_budget(msg, budget):
                        blocca(RuntimeError("picco oltre budget: risultati conservati, ricalibrare"))
                    # D17: riempimento senza barriera. Il lavoratore libero riceve subito un'unita'
                    # pendente della STESSA classe sullo stesso device: la compresenza decisa (contesto
                    # dell'ondata) resta quella misurata/stimata, nessun lavoratore resta fermo ad
                    # aspettare l'unita' piu' lunga. Classe diversa -> si torna alla decisione.
                    if errore is None and pausa is None and pid in figli and figli[pid].exitcode is None:
                        in_volo = set(attivi.values())
                        cl = lavori[k].classe
                        k2 = next((kk for kk in pendenti if kk not in in_volo and kk != k
                                   and lavori[kk].classe == cl), None)
                        if k2 is not None and ammissione():
                            attivi[pid], contesti[pid] = k2, ctx_onda
                            inizi[pid] = time.monotonic()
                            registra("riempimento", pid=pid, chiave=k2, contesto=ctx_onda,
                                     motivo="lavoratore libero, stessa classe: nessuna barriera")
                            try:
                                invii[pid].send(pendenti[k2])
                            except BrokenPipeError:
                                del attivi[pid]
                                blocca(G.ErroreLavoratore(f"pipe chiusa durante riempimento: {pid}/{k2}"))
                for pid in list(attivi):
                    if figli[pid].exitcode is not None:
                        blocca(G.ErroreLavoratore(
                            f"lavoratore {pid} morto: {figli[pid].exitcode}; unita' {attivi[pid]}"))
                        del attivi[pid]
                        continue
                    # Stallo (es. GPU bloccata, 21/10 tappa 150: 38 min senza uscite, coda HIP ferma):
                    # senza scadenza il coordinatore attende per sempre e l'orchestratore non riprende
                    # le corse sospese. Si blocca l'esecuzione; il finally termina i figli, le unita'
                    # in volo non sono pubblicate (scrittura transazionale) e la ripresa le rifa'.
                    trascorso = time.monotonic() - inizi.get(pid, inizio_onda)
                    limite = scadenza(lavori[attivi[pid]].classe)
                    if trascorso > limite and pid not in scaduti:
                        scaduti.add(pid)
                        registra("STALLO", pid=pid, chiave=attivi[pid], trascorso_s=trascorso,
                                 scadenza_s=limite, device=devices[pid])
                        blocca(G.ErroreLavoratore(
                            f"STALLO: unita' {attivi[pid]} su lavoratore {pid} da {trascorso:.0f} s "
                            f"(scadenza {limite:.0f} s)"))
                        del attivi[pid]
            # Nessun nuovo task dopo un drain, nemmeno su un altro device.
        if errore:
            raise errore
    finally:
        # I guasti sono fatali, la pressione transitoria attende. Un'interruzione
        # esterna riguarda solo questi figli, mai PID scoperti da /proc.
        for pid, p in figli.items():
            if p.is_alive():
                try:
                    invii[pid].send(None)
                except BrokenPipeError:
                    registra("figlio_gia_terminato", pid=pid)
        for p in figli.values():
            p.join(5)
            if p.is_alive():
                p.terminate()
                p.join(5)
            if p.is_alive():
                # Figlio sordo a SIGTERM (stallo): senza SIGKILL resta orfano e l'uscita dell'interprete
                # lo attende per sempre (join dei processi non daemon).
                p.kill()
                p.join(5)
                registra("figlio_ucciso_sigkill", pid=p.pid, vivo=p.is_alive())
            if p.is_alive():
                registra("errore_uscita_figlio", pid=p.pid)
        for conn in [*invii.values(), *ricezioni.values()]:
            conn.close()
        journal.close()
