from core.memoria_classificatore import giudica, spiega
lezione = ("Quando si ottimizza un modello quantizzato contro una sua SIMULAZIONE riscritta in un altro "
 "framework (pesi letti dal formato del motore di inferenza e riportati al formato di addestramento con "
 "una mappa inversa), la fedelta' della simulazione va misurata PER TOKEN contro le attivazioni intermedie "
 "del motore vero, strato per strato e su ogni tipo di strato, non con una correlazione sull'uscita: un "
 "segno sbagliato nell'inversa di una trasformazione come x = -exp(y) (scritta -log(-x) invece di log(-x)), "
 "su un vettore di poche decine di numeri, superava un controllo di correlazione a 0,98 mentre produceva "
 "circa il 90% di errore per token contro un errore vero di quantizzazione del 7%. L'ottimizzatore impara "
 "allora a compensare un errore che in produzione non esiste. Controllo complementare ed economico: ogni "
 "peso, dopo la mappa inversa, contro l'originale ad alta precisione — un coseno di -1 rivela un segno "
 "invertito, uno vicino a 0 una permutazione sbagliata; i tensori salvati senza quantizzazione devono "
 "combaciare esattamente.")
print(spiega(lezione))
if giudica(lezione).generale:
    from mogavis.repositories.ultron_repo import UltronRepo
    from mogavis.services.ultron_service import UltronService
    print(UltronService(UltronRepo()).impara(lezione, trasversale=True))
