// e2e_dump_logit — logit di TUTTI i token (non solo l'ultimo) per piu' sequenze di id,
// in un solo caricamento del modello, con i tensori intermedi del grafo che si vogliono
// (l_out-<il>, ffn_moe_topk-<il>, qualunque altro per nome). Fase E2E di Thor, strumento
// F0a/F0b/F0c di e2e_progetto_scelto.md (§2.6, §6, §9.3 punto 2). Scritto il 25/9/2026.
//
// Perche' non basta strumenti/dump_strati: usa llama_batch_get_one, cioe' batch.logits ==
// nullptr, e llama.cpp da' allora l'uscita del SOLO ultimo token (src/llama-batch.cpp:119-127
// della build k1024). Qui il batch e' costruito con llama_batch_init e logits[i] = 1 per ogni
// posizione voluta, esattamente come fa llama-perplexity (tools/perplexity/perplexity.cpp).
//
// ─── Uso ────────────────────────────────────────────────────────────────────────────────
//   e2e_dump_logit logit -m MODELLO.gguf -o CARTELLA [opzioni] SEQ.i32 [SEQ.i32 ...]
//     -t N              thread CPU (default 2)
//     -c N              n_ctx (default: la sequenza piu' lunga)
//     -b N              n_batch  (default 2048, come llama-perplexity; ridotto a n_ctx)
//     -ub N             n_ubatch (default 512,  come llama-perplexity)
//     --lunghezza L     ogni file e' spezzato in sequenze consecutive da L token
//                       (errore se la taglia non e' un multiplo di L: nessun resto taciuto)
//     --uscite M        di quali posizioni si calcolano e scrivono i logit:
//                       tutte (default) | meta (p >= n/2, il grafo di llama-perplexity) |
//                       ultima (p = n-1, il grafo di dump_strati) | da:P (p >= P)
//     --l-out S         tutti | lista di strati "0,5,39"  (tensori l_out-<il>)
//     --topk            anche ffn_moe_topk-<il> di tutti gli strati (int32 [k, token])
//     --tensore NOME    un tensore del grafo per nome esatto; "PREFISSO*" = tutti quelli col
//                       prefisso (ripetibile)
//     --no-logit        non scrive i logit (solo i tensori)
//     --kld-base FILE   scrive ANCHE FILE nel formato di `llama-perplexity --kl-divergence-base`
//                       (serve --uscite tutte|meta; tutte le sequenze della stessa lunghezza):
//                       e' il riscontro al byte con llama-perplexity
//     --vulkan          percorso di produzione, tutto su Vulkan (-ngl 999). Default: SOLO CPU,
//                       nessun dispositivo di offload (ne' pesi ne' operazioni)
//     --salva-tipo T    (26/9) f32 (default) | f16 | bf16: i tensori f32 del grafo si scrivono
//                       convertiti con arrotondamento al piu' vicino pari (RNE: ggml_fp32_to_bf16_row_ref,
//                       ggml_fp32_to_fp16_row), estensione .bf16/.f16. I tensori f16/i32 non si toccano.
//                       Con f16 o bf16 un valore non finito, o un trabocco f16 (|x| > 65504), ferma la
//                       corsa (codice 1): meglio fermarsi che scrivere un inf. I logit restano f32.
//     --segmenti F.i32  (26/9) uno per file di id, nello stesso ordine: int32 0/1 lungo quanto il file
//                       (spezzato come lui da --lunghezza); 1 = qui comincia un segmento (cu_seqlens del
//                       corpus). Ogni segmento si calcola come una sequenza a se': memoria azzerata
//                       (KV e stato GDN) e posizioni da 0. Le righe dei tensori e dei logit restano
//                       nell'ordine della sequenza intera. Il primo valore di ogni sequenza deve essere 1.
//                       Incompatibile con --kld-base (il protocollo di llama-perplexity non ha segmenti).
//
//   e2e_dump_logit tokenizza MODELLO.gguf TESTO N_CTX CARTELLA [N_CHUNK_MAX]
//     id identici a llama-perplexity: il file si legge in binario, si toglie UN '\n' finale
//     (common/arg.cpp, opzione -f), llama_tokenize(add_special=true, parse_special=false)
//     (common_tokenize), pezzi da N_CTX (n_chunk = n_token / N_CTX), BOS al posto del primo
//     token di ogni pezzo se il vocabolario lo chiede (perplexity.cpp). Carica SOLO il
//     vocabolario (vocab_only): nessun peso, memoria minima.
//     Scrive CARTELLA/seq_NNNN.i32 (gli id DA DARE al modello), CARTELLA/token_tutti.i32
//     (tutti gli id del testo, senza sostituzioni) e CARTELLA/tokenizza.json.
//
// ─── Formato di uscita (logit) ──────────────────────────────────────────────────────────
//   CARTELLA/manifesto.json      scritto PER ULTIMO e atomicamente (rename): esiste <=> la
//                                corsa e' completa e tutti i controlli interni sono passati
//   CARTELLA/seq_NNNN/ids.i32    gli id dati al modello (int32 grezzi)
//   CARTELLA/seq_NNNN/logit.f32  intestazione 4 x int64 (ne0 ne1 ne2 ne3) = (n_vocab, righe,
//                                1, 1), poi float32 contigui, riga r = logit alla posizione
//                                manifesto.sequenze[i].logit.posizioni[r] (in ordine
//                                crescente): e' la previsione del token in posizione p+1.
//                                Stessa intestazione di dump_strati (fedelta_b9._leggi_f32).
//   CARTELLA/seq_NNNN/nll.f64    stessa intestazione (1, righe, 1, 1), poi float64: per ogni
//                                riga -log softmax(logit)[id[p+1]] in doppia precisione; NaN
//                                sull'ultima posizione (non c'e' un token da prevedere)
//   CARTELLA/seq_NNNN/<NOME>.f32|.f16|.i32
//                                tensore del grafo, intestazione 4 x int64 come sopra, dati
//                                del tipo dell'estensione; i pezzi dei micro-batch sono
//                                concatenati lungo l'asse dei token (il piu' esterno non
//                                unitario: ne1 per [x, token], ne2 per [x, y, token]), che
//                                nell'intestazione riporta il totale. Le righe sono i token
//                                (0..n-1) oppure, per i tensori tagliati sulle uscite
//                                (inp_out_ids), le sole posizioni con logit: il manifesto lo
//                                dice ("righe_uguali_a": "token" | "uscite"). Un micro-batch
//                                senza uscite da' un pezzo vuoto (0 righe), contato.
//                                Limite noto: un tensore 3D in un micro-batch da 1 solo token
//                                ha l'asse ambiguo; la corsa si ferma, non scrive un file errato.
//   Nomi dei tensori: quelli del grafo llama.cpp, "l_out-<il>", "ffn_moe_topk-<il>",
//   "ffn_moe_probs-<il>", "attn_post_norm-<il>", "result_norm", ...
//
// ─── Garanzie ───────────────────────────────────────────────────────────────────────────
//   - una sequenza alla volta, con llama_memory_clear(mem, true) prima di ciascuna (KV e
//     stati ricorrenti GDN): il risultato di una sequenza non dipende dalle altre
//   - ogni tensore voluto deve comparire UNA volta per micro-batch, con forma coerente; un
//     nome esatto mai visto, un prefisso senza tensori, un numero di pezzi inatteso o un
//     totale di righe che non e' ne' n_token ne' n_uscite fermano la corsa (codice 1):
//     meglio fermarsi che scrivere un file parziale
//   - senza tensori richiesti non si installa nessun cb_eval: il grafo e' identico a quello
//     di llama-perplexity (con cb_eval il pianificatore spezza il grafo ai tensori chiesti;
//     su CPU non cambia i conti, su Vulkan puo' impedire fusioni: si misura, non si assume)

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <sys/resource.h>
#include <sys/stat.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

static const char * VERSIONE = "e2e_dump_logit 2 (26/9/2026: --salva-tipo, --segmenti)";

[[noreturn]] static void muori(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "ERRORE: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

static uint64_t fnv1a64(const void * p, size_t n) {
    const uint8_t * b = static_cast<const uint8_t *>(p);
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static std::string esadecimale(uint64_t v) {
    char s[17];
    snprintf(s, sizeof(s), "%016llx", (unsigned long long) v);
    return s;
}

static std::vector<int32_t> leggi_ids(const std::string & percorso) {
    std::ifstream f(percorso, std::ios::binary | std::ios::ate);
    if (!f) {
        muori("non leggo %s", percorso.c_str());
    }
    const std::streamsize byte = f.tellg();
    if (byte <= 0 || byte % (std::streamsize) sizeof(int32_t) != 0) {
        muori("%s: %lld byte, non e' un multiplo positivo di 4 (int32 grezzi)", percorso.c_str(), (long long) byte);
    }
    std::vector<int32_t> v(byte / sizeof(int32_t));
    f.seekg(0);
    f.read(reinterpret_cast<char *>(v.data()), byte);
    if (!f) {
        muori("lettura di %s incompleta", percorso.c_str());
    }
    return v;
}

static void scrivi_file(const std::string & percorso, const void * dati, size_t n) {
    FILE * f = fopen(percorso.c_str(), "wb");
    if (!f) {
        muori("non creo %s", percorso.c_str());
    }
    if (n && fwrite(dati, 1, n, f) != n) {
        muori("scrittura di %s fallita", percorso.c_str());
    }
    if (fclose(f) != 0) {
        muori("chiusura di %s fallita", percorso.c_str());
    }
}

static void crea_cartella(const std::string & p) {
    if (mkdir(p.c_str(), 0755) != 0 && errno != EEXIST) {
        muori("non creo la cartella %s", p.c_str());
    }
}

static long rss_max_kib() {
    struct rusage u;
    getrusage(RUSAGE_SELF, &u);
    return u.ru_maxrss;
}

// ─── Copia ESATTA delle funzioni di llama-perplexity (tools/perplexity/perplexity.cpp della
//     build k1024, righe 71-107): servono a scrivere il file --kl-divergence-base identico ──
static inline int nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static double log_softmax_perplexity(int n_vocab, const float * logits, uint16_t * log_prob, int tok) {
    float max_logit = logits[0];
    float min_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
        min_logit = std::min(min_logit, logits[i]);
    }
    min_logit = std::max(min_logit, max_logit - 16);
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }
    const float log_sum_exp = log(sum_exp);
    const float min_log_prob = min_logit - max_logit - log_sum_exp;
    const float scale = (max_logit - min_logit)/65535.f;
    float * d = (float *)log_prob;
    d[0] = scale;
    d[1] = min_log_prob;
    log_prob += 4;
    if (scale) {
        const float inv_scale = 1/scale;
        for (int i = 0; i < n_vocab; ++i) {
            log_prob[i] = logits[i] > min_logit ? nearest_int(inv_scale*(logits[i] - min_logit)) : 0;
        }
    } else {
        std::memset(log_prob, 0, n_vocab*sizeof(uint16_t));
    }
    return max_logit + log_sum_exp - logits[tok];
}

// NLL in doppia precisione (per i confronti con torch; non e' quella di perplexity)
static double nll_doppia(int n_vocab, const float * logits, int tok) {
    double m = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        m = std::max(m, (double) logits[i]);
    }
    double s = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        s += std::exp((double) logits[i] - m);
    }
    return m + std::log(s) - (double) logits[tok];
}

// ─── Raccolta dei tensori del grafo ─────────────────────────────────────────────────────
struct Flusso {
    std::string nome;
    std::string percorso;
    FILE *      f = nullptr;
    ggml_type   tipo = GGML_TYPE_F32;       // tipo nel grafo
    ggml_type   tipo_file = GGML_TYPE_F32;  // tipo scritto su disco (26/9: --salva-tipo)
    int64_t     ne[4] = {0, 0, 0, 0};
    int         asse = 1;
    int64_t     righe = 0;
    int         pezzi = 0;
    std::vector<int64_t> righe_pezzo;
};

struct Raccolta {
    std::set<std::string>    esatti;
    std::vector<std::string> prefissi;
    std::string              cartella;  // cartella della sequenza corrente
    std::map<std::string, Flusso> flussi;
    std::vector<uint8_t>     buf;
    std::vector<uint8_t>     grezzo;
    std::vector<uint8_t>     conv;                       // 26/9: dati convertiti (--salva-tipo)
    ggml_type                salva_tipo = GGML_TYPE_F32; // tipo su disco dei tensori f32 del grafo
    bool                     errore = false;

    bool voluto(const char * nome) const {
        if (esatti.count(nome)) {
            return true;
        }
        for (const auto & p : prefissi) {
            if (strncmp(nome, p.c_str(), p.size()) == 0) {
                return true;
            }
        }
        return false;
    }

    bool fallisci(const char * fmt, ...) {
        va_list ap;
        va_start(ap, fmt);
        fprintf(stderr, "ERRORE (raccolta): ");
        vfprintf(stderr, fmt, ap);
        fprintf(stderr, "\n");
        va_end(ap);
        errore = true;
        return false;  // ferma il calcolo del grafo
    }

    bool accogli(const ggml_tensor * t) {
        if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_I32) {
            return fallisci("%s: tipo %s non gestito (solo f32, f16, i32)", t->name, ggml_type_name(t->type));
        }
        const size_t es = ggml_type_size(t->type);
        auto it = flussi.find(t->name);
        if (it == flussi.end()) {
            Flusso fl;
            fl.nome = t->name;
            fl.tipo = t->type;
            for (int i = 0; i < 4; ++i) {
                fl.ne[i] = t->ne[i];
            }
            // asse dei token = il piu' esterno non unitario (0 righe comprese). Ambiguo solo per
            // un tensore 3D in un micro-batch da 1 token: il pezzo dopo non combacerebbe e la
            // corsa si ferma (controllo di forma qui sotto), non scrive un file sbagliato
            fl.asse = t->ne[3] != 1 ? 3 : (t->ne[2] != 1 ? 2 : 1);
            fl.tipo_file = t->type == GGML_TYPE_F32 ? salva_tipo : t->type;
            const char * est = fl.tipo_file == GGML_TYPE_F32 ? ".f32" :
                               fl.tipo_file == GGML_TYPE_F16 ? ".f16" :
                               fl.tipo_file == GGML_TYPE_BF16 ? ".bf16" : ".i32";
            fl.percorso = cartella + "/" + fl.nome + est;
            fl.f = fopen(fl.percorso.c_str(), "wb");
            if (!fl.f) {
                return fallisci("non creo %s", fl.percorso.c_str());
            }
            const int64_t zero[4] = {0, 0, 0, 0};  // intestazione provvisoria, riscritta alla fine
            if (fwrite(zero, sizeof(zero), 1, fl.f) != 1) {
                return fallisci("scrittura di %s fallita", fl.percorso.c_str());
            }
            it = flussi.emplace(fl.nome, fl).first;
        }
        Flusso & fl = it->second;
        if (t->type != fl.tipo) {
            return fallisci("%s: tipo cambiato fra micro-batch", t->name);
        }
        for (int i = 0; i < 4; ++i) {
            if (i == fl.asse) {
                continue;
            }
            if (t->ne[i] != fl.ne[i] || (i > fl.asse && t->ne[i] != 1)) {
                return fallisci("%s: forma incoerente fra micro-batch (asse %d: %lld contro %lld)",
                                t->name, i, (long long) t->ne[i], (long long) fl.ne[i]);
            }
        }
        const int64_t n_el = ggml_nelements(t);
        if (n_el == 0) {
            // tensore tagliato sulle uscite in un lotto senza uscite (es. --uscite meta, primo
            // lotto): pezzo vuoto, contato ma senza dati
            fl.pezzi += 1;
            fl.righe_pezzo.push_back(0);
            return true;
        }
        buf.resize((size_t) n_el * es);
        const uint8_t * src = nullptr;
        if (t->buffer == nullptr || t->data == nullptr) {
            return fallisci("%s: tensore senza dati", t->name);
        }
        if (ggml_backend_buffer_is_host(t->buffer)) {
            src = static_cast<const uint8_t *>(t->data);
        } else {
            grezzo.resize(ggml_nbytes(t));
            ggml_backend_tensor_get(const_cast<ggml_tensor *>(t), grezzo.data(), 0, grezzo.size());
            src = grezzo.data();
        }
        if (ggml_is_contiguous(t)) {
            memcpy(buf.data(), src, buf.size());
        } else {
            // vista non contigua (es. ffn_moe_topk = prime k colonne di un argsort): si
            // raccoglie elemento per elemento seguendo i passi nb[]
            uint8_t * dst = buf.data();
            for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
                for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
                    for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                        for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                            memcpy(dst, src + i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3], es);
                            dst += es;
                        }
                    }
                }
            }
        }
        const uint8_t * da_scrivere = buf.data();
        size_t n_da_scrivere = buf.size();
        if (fl.tipo_file != fl.tipo) {
            // 26/9: f32 del grafo -> f16/bf16 su disco, RNE; nessun valore non finito ammesso
            const float * x = reinterpret_cast<const float *>(buf.data());
            int64_t non_finiti = 0;
            for (int64_t i = 0; i < n_el; ++i) {
                non_finiti += !std::isfinite(x[i]);
            }
            if (non_finiti) {
                return fallisci("%s: %lld valori non finiti nel grafo", t->name, (long long) non_finiti);
            }
            conv.resize((size_t) n_el * 2);
            if (fl.tipo_file == GGML_TYPE_BF16) {
                ggml_fp32_to_bf16_row_ref(x, reinterpret_cast<ggml_bf16_t *>(conv.data()), n_el);
            } else {
                ggml_fp32_to_fp16_row(x, reinterpret_cast<ggml_fp16_t *>(conv.data()), n_el);
                const ggml_fp16_t * y = reinterpret_cast<const ggml_fp16_t *>(conv.data());
                int64_t trabocchi = 0;
                for (int64_t i = 0; i < n_el; ++i) {
                    trabocchi += (y[i] & 0x7c00) == 0x7c00;  // esponente pieno: inf (i NaN sono esclusi sopra)
                }
                if (trabocchi) {
                    return fallisci("%s: %lld valori oltre il campo f16 (|x| > 65504)", t->name, (long long) trabocchi);
                }
            }
            da_scrivere = conv.data();
            n_da_scrivere = conv.size();
        }
        if (fwrite(da_scrivere, 1, n_da_scrivere, fl.f) != n_da_scrivere) {
            return fallisci("scrittura di %s fallita", fl.percorso.c_str());
        }
        fl.righe += t->ne[fl.asse];
        fl.pezzi += 1;
        fl.righe_pezzo.push_back(t->ne[fl.asse]);
        return true;
    }
};

static bool richiamo(struct ggml_tensor * t, bool chiedi, void * dati) {
    auto * r = static_cast<Raccolta *>(dati);
    if (r->errore) {
        return false;
    }
    const bool vuole = r->voluto(t->name);
    if (chiedi) {
        return vuole;
    }
    if (!vuole) {
        return true;
    }
    return r->accogli(t);
}

// ─── Opzioni ────────────────────────────────────────────────────────────────────────────
struct Opzioni {
    std::string modello;
    std::string cartella;
    std::vector<std::string> file_ids;
    int n_thread  = 2;
    int n_ctx     = 0;
    int n_batch   = 2048;
    int n_ubatch  = 512;
    int lunghezza = 0;
    std::string uscite = "tutte";
    int uscite_da = 0;
    std::string l_out;
    bool topk = false;
    std::vector<std::string> tensori;
    bool logit = true;
    std::string kld_base;
    bool vulkan = false;
    std::string salva_tipo = "f32";          // 26/9
    std::vector<std::string> file_segmenti;  // 26/9: uno per file di id, stesso ordine
};

static int intero(const char * s, const char * cosa) {
    char * fine = nullptr;
    const long v = strtol(s, &fine, 10);
    if (!s[0] || *fine || v < 0 || v > std::numeric_limits<int>::max()) {
        muori("%s: \"%s\" non e' un intero valido", cosa, s);
    }
    return (int) v;
}

static Opzioni leggi_opzioni(int argc, char ** argv) {
    Opzioni o;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](const char * cosa) -> const char * {
            if (i + 1 >= argc) {
                muori("%s vuole un valore", cosa);
            }
            return argv[++i];
        };
        if (a == "-m")               { o.modello = val("-m"); }
        else if (a == "-o")          { o.cartella = val("-o"); }
        else if (a == "-t")          { o.n_thread = intero(val("-t"), "-t"); }
        else if (a == "-c")          { o.n_ctx = intero(val("-c"), "-c"); }
        else if (a == "-b")          { o.n_batch = intero(val("-b"), "-b"); }
        else if (a == "-ub")         { o.n_ubatch = intero(val("-ub"), "-ub"); }
        else if (a == "--lunghezza") { o.lunghezza = intero(val("--lunghezza"), "--lunghezza"); }
        else if (a == "--uscite")    { o.uscite = val("--uscite"); }
        else if (a == "--l-out")     { o.l_out = val("--l-out"); }
        else if (a == "--topk")      { o.topk = true; }
        else if (a == "--tensore")   { o.tensori.push_back(val("--tensore")); }
        else if (a == "--no-logit")  { o.logit = false; }
        else if (a == "--kld-base")  { o.kld_base = val("--kld-base"); }
        else if (a == "--vulkan")    { o.vulkan = true; }
        else if (a == "--salva-tipo") { o.salva_tipo = val("--salva-tipo"); }
        else if (a == "--segmenti")  { o.file_segmenti.push_back(val("--segmenti")); }
        else if (!a.empty() && a[0] == '-') { muori("opzione sconosciuta %s", a.c_str()); }
        else                         { o.file_ids.push_back(a); }
    }
    if (o.modello.empty() || o.cartella.empty() || o.file_ids.empty()) {
        muori("servono -m MODELLO, -o CARTELLA e almeno un file di id");
    }
    if (o.n_thread < 1 || o.n_batch < 1 || o.n_ubatch < 1) {
        muori("-t, -b, -ub devono essere >= 1");
    }
    if (o.uscite.rfind("da:", 0) == 0) {
        o.uscite_da = intero(o.uscite.c_str() + 3, "--uscite da:P");
        o.uscite = "da";
    } else if (o.uscite != "tutte" && o.uscite != "meta" && o.uscite != "ultima") {
        muori("--uscite: tutte | meta | ultima | da:P");
    }
    if (!o.logit && !o.kld_base.empty()) {
        muori("--kld-base richiede i logit (niente --no-logit)");
    }
    if (o.salva_tipo != "f32" && o.salva_tipo != "f16" && o.salva_tipo != "bf16") {
        muori("--salva-tipo: f32 | f16 | bf16");
    }
    if (!o.file_segmenti.empty() && o.file_segmenti.size() != o.file_ids.size()) {
        muori("--segmenti: %zu file per %zu file di id (ne serve uno per ciascuno, stesso ordine)",
              o.file_segmenti.size(), o.file_ids.size());
    }
    if (!o.file_segmenti.empty() && !o.kld_base.empty()) {
        muori("--segmenti e --kld-base non vanno insieme (llama-perplexity non ha segmenti)");
    }
    return o;
}

static bool vuole_uscita(const Opzioni & o, int p, int n) {
    if (o.uscite == "tutte")  { return true; }
    if (o.uscite == "meta")   { return p >= n / 2; }
    if (o.uscite == "ultima") { return p == n - 1; }
    return p >= o.uscite_da;
}

struct Sequenza {
    std::string sorgente;
    int64_t     offset = 0;  // in token, dentro il file sorgente
    std::vector<int32_t> ids;
    std::vector<int>     inizi = {0};  // 26/9: inizi dei segmenti (sempre 0 in testa)
};

static std::vector<ggml_backend_dev_t> & nessun_dispositivo() {
    static std::vector<ggml_backend_dev_t> v = {nullptr};
    return v;
}

static json dispositivi_registrati() {
    json d = json::array();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto tipo = ggml_backend_dev_type(dev);
        const char * nt = tipo == GGML_BACKEND_DEVICE_TYPE_CPU ? "cpu" :
                          tipo == GGML_BACKEND_DEVICE_TYPE_GPU ? "gpu" :
                          tipo == GGML_BACKEND_DEVICE_TYPE_IGPU ? "igpu" : "altro";
        d.push_back({{"nome", ggml_backend_dev_name(dev)}, {"tipo", nt}});
    }
    return d;
}

// ─── Sottocomando: tokenizza come llama-perplexity ──────────────────────────────────────
static int tokenizza(int argc, char ** argv) {
    if (argc < 6) {
        muori("uso: %s tokenizza MODELLO.gguf TESTO N_CTX CARTELLA [N_CHUNK_MAX]", argv[0]);
    }
    const std::string modello = argv[2], testo_p = argv[3], cartella = argv[5];
    const int n_ctx = intero(argv[4], "N_CTX");
    const int n_chunk_max = argc > 6 ? intero(argv[6], "N_CHUNK_MAX") : -1;
    if (n_ctx < 2) {
        muori("N_CTX >= 2");
    }
    std::ifstream f(testo_p, std::ios::binary);
    if (!f) {
        muori("non leggo %s", testo_p.c_str());
    }
    std::string testo((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const size_t byte_file = testo.size();
    if (!testo.empty() && testo.back() == '\n') {
        testo.pop_back();  // come common/arg.cpp (-f)
    }

    llama_backend_init();
    auto pm = llama_model_default_params();
    pm.vocab_only = true;
    pm.n_gpu_layers = 0;
    pm.devices = nessun_dispositivo().data();
    llama_model * m = llama_model_load_from_file(modello.c_str(), pm);
    if (!m) {
        muori("vocabolario non caricato da %s", modello.c_str());
    }
    const llama_vocab * vocab = llama_model_get_vocab(m);

    // = common_tokenize(vocab, testo, add_special=true, parse_special=false)
    int n = (int) testo.size() + 2;
    std::vector<llama_token> tok(n);
    n = llama_tokenize(vocab, testo.data(), (int32_t) testo.size(), tok.data(), (int32_t) tok.size(), true, false);
    if (n == std::numeric_limits<int32_t>::min()) {
        muori("tokenizzazione oltre il limite int32");
    }
    if (n < 0) {
        tok.resize(-n);
        const int chk = llama_tokenize(vocab, testo.data(), (int32_t) testo.size(), tok.data(), (int32_t) tok.size(), true, false);
        if (chk != -n) {
            muori("tokenizzazione incoerente");
        }
    } else {
        tok.resize(n);
    }
    const int n_tok = (int) tok.size();
    if (n_tok < 2 * n_ctx) {
        muori("%d token: llama-perplexity ne vuole almeno 2*N_CTX = %d", n_tok, 2 * n_ctx);
    }
    const int n_chunk_tot = n_tok / n_ctx;
    const int n_chunk = n_chunk_max < 0 ? n_chunk_tot : std::min(n_chunk_max, n_chunk_tot);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const llama_token bos = llama_vocab_bos(vocab);

    crea_cartella(cartella);
    scrivi_file(cartella + "/token_tutti.i32", tok.data(), tok.size() * sizeof(int32_t));
    json seqs = json::array();
    for (int i = 0; i < n_chunk; ++i) {
        std::vector<int32_t> s(tok.begin() + (size_t) i * n_ctx, tok.begin() + (size_t) (i + 1) * n_ctx);
        if (add_bos) {
            s[0] = bos;  // come perplexity.cpp: BOS al posto del primo token del pezzo
        }
        char nome[64];
        snprintf(nome, sizeof(nome), "seq_%04d.i32", i);
        scrivi_file(cartella + "/" + nome, s.data(), s.size() * sizeof(int32_t));
        seqs.push_back({{"file", nome}, {"offset", (int64_t) i * n_ctx}, {"fnv1a64", esadecimale(fnv1a64(s.data(), s.size() * 4))}});
    }
    json j = {
        {"versione", VERSIONE}, {"modello", modello}, {"testo", testo_p}, {"byte_file", byte_file},
        {"tolto_newline_finale", testo.size() != byte_file}, {"n_token", n_tok}, {"n_ctx", n_ctx},
        {"n_chunk", n_chunk}, {"n_chunk_nel_testo", n_chunk_tot}, {"add_bos", add_bos}, {"bos", bos},
        {"add_special", true}, {"parse_special", false}, {"sequenze", seqs},
    };
    const std::string man = cartella + "/tokenizza.json";
    const std::string s = j.dump(1);
    scrivi_file(man, s.data(), s.size());
    llama_model_free(m);
    llama_backend_free();
    fprintf(stderr, "FATTO: %d token, %d pezzi da %d, add_bos=%d\n", n_tok, n_chunk, n_ctx, (int) add_bos);
    return 0;
}

// ─── Sottocomando: logit (e tensori) ────────────────────────────────────────────────────
static int logit(int argc, char ** argv) {
    const auto t_inizio = std::chrono::steady_clock::now();
    Opzioni o = leggi_opzioni(argc, argv);

    // sequenze
    std::vector<Sequenza> seqs;
    for (size_t fi = 0; fi < o.file_ids.size(); ++fi) {
        const std::string & p = o.file_ids[fi];
        std::vector<int32_t> ids = leggi_ids(p);
        // 26/9: segmenti (inizi 0/1 lungo quanto il file), spezzati come gli id
        std::vector<int32_t> seg;
        if (!o.file_segmenti.empty()) {
            seg = leggi_ids(o.file_segmenti[fi]);
            if (seg.size() != ids.size()) {
                muori("%s: %zu valori, %s ha %zu token", o.file_segmenti[fi].c_str(), seg.size(), p.c_str(), ids.size());
            }
            for (size_t q = 0; q < seg.size(); ++q) {
                if (seg[q] != 0 && seg[q] != 1) {
                    muori("%s: valore %d in posizione %zu (ammessi 0 e 1)", o.file_segmenti[fi].c_str(), seg[q], q);
                }
            }
        }
        const size_t L = o.lunghezza > 0 ? (size_t) o.lunghezza : ids.size();
        if (o.lunghezza > 0 && ids.size() % L != 0) {
            muori("%s: %zu token non e' un multiplo di --lunghezza %d", p.c_str(), ids.size(), o.lunghezza);
        }
        for (size_t k = 0; k < ids.size(); k += L) {
            Sequenza s;
            s.sorgente = p;
            s.offset = (int64_t) k;
            s.ids.assign(ids.begin() + k, ids.begin() + k + L);
            if (!seg.empty()) {
                if (seg[k] != 1) {
                    muori("%s: la sequenza che inizia al token %zu non comincia un segmento (primo valore 0)",
                          o.file_segmenti[fi].c_str(), k);
                }
                s.inizi.clear();
                for (size_t q = 0; q < L; ++q) {
                    if (seg[k + q] == 1) {
                        s.inizi.push_back((int) q);
                    }
                }
            }
            seqs.push_back(std::move(s));
        }
    }
    int n_max = 0;
    for (const auto & s : seqs) {
        n_max = std::max(n_max, (int) s.ids.size());
    }
    for (size_t k = 0; k < seqs.size(); ++k) {
        if (seqs[k].ids.size() < 2) {
            muori("sequenza %zu (%s): servono almeno 2 token", k, seqs[k].sorgente.c_str());
        }
    }
    const int n_ctx_rich = o.n_ctx > 0 ? o.n_ctx : n_max;
    if (n_ctx_rich < n_max) {
        muori("-c %d minore della sequenza piu' lunga (%d)", n_ctx_rich, n_max);
    }
    if (!o.kld_base.empty()) {
        for (const auto & s : seqs) {
            if ((int) s.ids.size() != (int) seqs[0].ids.size()) {
                muori("--kld-base: tutte le sequenze devono avere la stessa lunghezza");
            }
        }
        const int n = (int) seqs[0].ids.size();
        if (!(o.uscite == "tutte" || o.uscite == "meta" || (o.uscite == "da" && o.uscite_da <= n / 2))) {
            muori("--kld-base: le uscite devono coprire le posizioni >= n/2 (tutte | meta | da:P con P <= n/2)");
        }
    }

    // richieste di tensori
    Raccolta r;
    if (!o.l_out.empty()) {
        if (o.l_out == "tutti") {
            r.prefissi.push_back("l_out-");
        } else {
            size_t i = 0;
            while (i <= o.l_out.size()) {
                size_t j = o.l_out.find(',', i);
                if (j == std::string::npos) {
                    j = o.l_out.size();
                }
                const std::string pezzo = o.l_out.substr(i, j - i);
                r.esatti.insert("l_out-" + std::to_string(intero(pezzo.c_str(), "--l-out")));
                i = j + 1;
            }
        }
    }
    if (o.topk) {
        r.prefissi.push_back("ffn_moe_topk-");
    }
    for (const auto & t : o.tensori) {
        if (!t.empty() && t.back() == '*') {
            r.prefissi.push_back(t.substr(0, t.size() - 1));
        } else {
            r.esatti.insert(t);
        }
    }
    const bool con_tensori = !r.esatti.empty() || !r.prefissi.empty();
    r.salva_tipo = o.salva_tipo == "bf16" ? GGML_TYPE_BF16 : (o.salva_tipo == "f16" ? GGML_TYPE_F16 : GGML_TYPE_F32);

    llama_backend_init();
    const json dispositivi = dispositivi_registrati();
    auto pm = llama_model_default_params();
    pm.n_gpu_layers = o.vulkan ? 999 : 0;
    if (!o.vulkan) {
        pm.devices = nessun_dispositivo().data();
    }
    llama_model * modello = llama_model_load_from_file(o.modello.c_str(), pm);
    if (!modello) {
        muori("modello non caricato: %s", o.modello.c_str());
    }
    const llama_vocab * vocab = llama_model_get_vocab(modello);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    for (size_t k = 0; k < seqs.size(); ++k) {
        for (int32_t id : seqs[k].ids) {
            if (id < 0 || id >= n_vocab) {
                muori("sequenza %zu (%s): id %d fuori da [0, %d)", k, seqs[k].sorgente.c_str(), id, n_vocab);
            }
        }
    }

    auto pc = llama_context_default_params();
    pc.n_ctx      = (uint32_t) n_ctx_rich;
    pc.n_batch    = (uint32_t) std::min(o.n_batch, n_ctx_rich);  // come perplexity: min(n_batch, n_ctx)
    pc.n_ubatch   = (uint32_t) o.n_ubatch;
    pc.n_seq_max  = 1;
    pc.n_threads  = o.n_thread;
    pc.n_threads_batch = o.n_thread;
    pc.no_perf    = true;
    pc.op_offload = o.vulkan;
    pc.offload_kqv = o.vulkan;
    if (con_tensori) {
        pc.cb_eval = richiamo;
        pc.cb_eval_user_data = &r;
    }
    llama_context * ctx = llama_init_from_model(modello, pc);
    if (!ctx) {
        muori("contesto non creato");
    }
    const int n_batch  = (int) llama_n_batch(ctx);
    const int n_ubatch = (int) llama_n_ubatch(ctx);

    crea_cartella(o.cartella);

    // --kld-base: intestazione come perplexity.cpp (righe 464-469 e 518-523)
    FILE * fk = nullptr;
    const int nv = 2 * ((n_vocab + 1) / 2) + 4;
    std::vector<uint16_t> log_probs(nv, 0);  // un record, azzerato una volta come in perplexity
    if (!o.kld_base.empty()) {
        fk = fopen(o.kld_base.c_str(), "wb");
        if (!fk) {
            muori("non creo %s", o.kld_base.c_str());
        }
        const int32_t n_ctx_k = (int32_t) seqs[0].ids.size();
        const int32_t n_chunk_k = (int32_t) seqs.size();
        fwrite("_logits_", 8, 1, fk);
        fwrite(&n_ctx_k, sizeof(n_ctx_k), 1, fk);
        fwrite(&n_vocab, sizeof(n_vocab), 1, fk);
        fwrite(&n_chunk_k, sizeof(n_chunk_k), 1, fk);
        for (const auto & s : seqs) {
            fwrite(s.ids.data(), sizeof(int32_t), s.ids.size(), fk);
        }
    }

    llama_batch lotto = llama_batch_init(n_batch, 0, 1);
    json man_seqs = json::array();
    double nll_pp_tot = 0.0;   // somma stile perplexity (posizioni n/2 .. n-2)
    int64_t n_pp_tot = 0;
    std::set<std::string> nomi_prima_seq;

    for (size_t k = 0; k < seqs.size(); ++k) {
        const auto t_seq = std::chrono::steady_clock::now();
        const auto & ids = seqs[k].ids;
        const int n = (int) ids.size();
        char nome_seq[32];
        snprintf(nome_seq, sizeof(nome_seq), "seq_%04zu", k);
        const std::string cart = o.cartella + "/" + nome_seq;
        crea_cartella(cart);
        scrivi_file(cart + "/ids.i32", ids.data(), ids.size() * sizeof(int32_t));
        r.cartella = cart;
        r.flussi.clear();

        std::vector<int> posizioni;
        for (int p = 0; p < n; ++p) {
            if (vuole_uscita(o, p, n)) {
                posizioni.push_back(p);
            }
        }
        const int n_uscite = (int) posizioni.size();
        if (n_uscite == 0) {
            muori("sequenza %zu: nessuna posizione d'uscita con --uscite %s", k, o.uscite.c_str());
        }
        FILE * fl = nullptr;
        FILE * fn = nullptr;
        if (o.logit) {
            fl = fopen((cart + "/logit.f32").c_str(), "wb");
            fn = fopen((cart + "/nll.f64").c_str(), "wb");
            if (!fl || !fn) {
                muori("non creo i file dei logit in %s", cart.c_str());
            }
            const int64_t hl[4] = {n_vocab, n_uscite, 1, 1};
            const int64_t hn[4] = {1, n_uscite, 1, 1};
            fwrite(hl, sizeof(hl), 1, fl);
            fwrite(hn, sizeof(hn), 1, fn);
        }

        double nll_tutte = 0.0, nll_meta = 0.0, nll_pp = 0.0;
        int64_t c_tutte = 0, c_meta = 0, c_pp = 0;
        int righe_scritte = 0;
        int attese_pezzi = 0;
        const int first = n / 2;

        // 26/9: un segmento alla volta (senza --segmenti: un solo segmento [0, n), come prima).
        // Ogni segmento riparte da memoria vuota (KV e stati ricorrenti) e da posizione 0.
        const auto & inizi = seqs[k].inizi;
        for (size_t si = 0; si < inizi.size(); ++si) {
        const int seg_a = inizi[si];
        const int seg_b = si + 1 < inizi.size() ? inizi[si + 1] : n;
        llama_memory_clear(llama_get_memory(ctx), true);  // KV e stati ricorrenti
        for (int inizio = seg_a; inizio < seg_b; inizio += n_batch) {
            const int taglia = std::min(n_batch, seg_b - inizio);
            lotto.n_tokens = taglia;
            int uscite_lotto = 0;
            for (int q = 0; q < taglia; ++q) {
                const int p = inizio + q;
                lotto.token[q]     = ids[p];
                lotto.pos[q]       = p - seg_a;
                lotto.n_seq_id[q]  = 1;
                lotto.seq_id[q][0] = 0;
                lotto.logits[q]    = vuole_uscita(o, p, n) ? 1 : 0;
                uscite_lotto += lotto.logits[q];
            }
            std::map<std::string, int> pezzi_prima;
            for (const auto & kv : r.flussi) {
                pezzi_prima[kv.first] = kv.second.pezzi;
            }
            const int rc = llama_decode(ctx, lotto);
            if (rc != 0 || r.errore) {
                muori("decode fallito sulla sequenza %zu, posizioni %d-%d (rc=%d, errore raccolta=%d)",
                      k, inizio, inizio + taglia - 1, rc, (int) r.errore);
            }
            // ogni tensore voluto: una volta per micro-batch
            const int micro = (taglia + n_ubatch - 1) / n_ubatch;
            attese_pezzi += micro;
            for (const auto & kv : r.flussi) {
                const int prima = pezzi_prima.count(kv.first) ? pezzi_prima[kv.first] : 0;
                if (kv.second.pezzi - prima != micro) {
                    muori("%s: %d pezzi in un lotto da %d micro-batch (sequenza %zu, posizione %d)",
                          kv.first.c_str(), kv.second.pezzi - prima, micro, k, inizio);
                }
            }
            if (!o.logit) {
                continue;
            }
            for (int q = 0; q < taglia; ++q) {
                if (!lotto.logits[q]) {
                    continue;
                }
                const int p = inizio + q;
                const float * riga = llama_get_logits_ith(ctx, q);
                if (!riga) {
                    muori("logit assenti per la posizione %d della sequenza %zu", p, k);
                }
                if (posizioni[righe_scritte] != p) {
                    muori("ordine delle righe incoerente (%d contro %d)", posizioni[righe_scritte], p);
                }
                if (fwrite(riga, sizeof(float), n_vocab, fl) != (size_t) n_vocab) {
                    muori("scrittura dei logit fallita");
                }
                double v = std::numeric_limits<double>::quiet_NaN();
                if (p + 1 < n) {
                    v = nll_doppia(n_vocab, riga, ids[p + 1]);
                    nll_tutte += v;
                    c_tutte += 1;
                    if (p >= first) {
                        nll_meta += v;
                        c_meta += 1;
                        // stesso valore, stessa aritmetica di perplexity (e record kld)
                        const double vp = log_softmax_perplexity(n_vocab, riga, log_probs.data(), ids[p + 1]);
                        nll_pp += vp;
                        c_pp += 1;
                        if (fk) {
                            fwrite(log_probs.data(), sizeof(uint16_t), nv, fk);
                        }
                    }
                }
                fwrite(&v, sizeof(double), 1, fn);
                righe_scritte += 1;
            }
            (void) uscite_lotto;
        }
        }  // segmenti
        if (o.logit) {
            if (righe_scritte != n_uscite) {
                muori("sequenza %zu: %d righe di logit scritte, attese %d", k, righe_scritte, n_uscite);
            }
            if (fclose(fl) != 0 || fclose(fn) != 0) {
                muori("chiusura dei file dei logit fallita");
            }
        }
        // chiusura dei tensori: intestazione vera + controlli
        json man_t = json::object();
        std::set<std::string> nomi_visti;
        for (auto & kv : r.flussi) {
            Flusso & f = kv.second;
            nomi_visti.insert(f.nome);
            if (f.pezzi != attese_pezzi) {
                muori("%s: %d pezzi, attesi %d", f.nome.c_str(), f.pezzi, attese_pezzi);
            }
            std::string uguali;
            if (f.righe == n) {
                uguali = "token";
            } else if (f.righe == n_uscite) {
                uguali = "uscite";
            } else {
                muori("%s: %lld righe, ne' %d token ne' %d uscite", f.nome.c_str(), (long long) f.righe, n, n_uscite);
            }
            int64_t ne[4] = {f.ne[0], f.ne[1], f.ne[2], f.ne[3]};
            ne[f.asse] = f.righe;
            if (fseek(f.f, 0, SEEK_SET) != 0 || fwrite(ne, sizeof(ne), 1, f.f) != 1 || fclose(f.f) != 0) {
                muori("chiusura di %s fallita", f.percorso.c_str());
            }
            f.f = nullptr;
            const char * nt = f.tipo_file == GGML_TYPE_F32 ? "f32" : f.tipo_file == GGML_TYPE_F16 ? "f16" :
                              f.tipo_file == GGML_TYPE_BF16 ? "bf16" : "i32";
            man_t[f.nome] = {
                {"file", f.percorso.substr(o.cartella.size() + 1)}, {"tipo", nt},
                {"tipo_grafo", ggml_type_name(f.tipo)},
                {"ne", {ne[0], ne[1], ne[2], ne[3]}}, {"asse_token", f.asse}, {"righe", f.righe},
                {"righe_uguali_a", uguali}, {"pezzi", f.pezzi}, {"righe_pezzo", f.righe_pezzo},
            };
        }
        for (const auto & e : r.esatti) {
            if (!nomi_visti.count(e)) {
                muori("tensore %s mai visto nel grafo", e.c_str());
            }
        }
        json per_prefisso = json::object();
        for (const auto & p : r.prefissi) {
            std::vector<int> indici;
            bool tutti_numerici = true;
            int quanti = 0;
            for (const auto & nm : nomi_visti) {
                if (nm.rfind(p, 0) == 0) {
                    quanti += 1;
                    // indice di strato: solo cifre, dopo un '-' facoltativo ("l_out-" + "3" oppure
                    // "l_out" + "-3"); qualunque altra coda (es. "_biased-3") non e' un indice
                    std::string resto = nm.substr(p.size());
                    if (!resto.empty() && resto[0] == '-') {
                        resto = resto.substr(1);
                    }
                    const bool cifre = !resto.empty() && resto.size() <= 6 &&
                        std::all_of(resto.begin(), resto.end(), [](char c) { return c >= '0' && c <= '9'; });
                    if (!cifre) {
                        tutti_numerici = false;
                    } else {
                        indici.push_back(std::stoi(resto));
                    }
                }
            }
            if (quanti == 0) {
                muori("nessun tensore col prefisso \"%s\" nel grafo", p.c_str());
            }
            json jp = {{"tensori", quanti}};
            if (tutti_numerici) {
                std::sort(indici.begin(), indici.end());
                bool contigui = true;
                for (size_t i = 0; i < indici.size(); ++i) {
                    contigui = contigui && indici[i] == (int) i;
                }
                // l_out-/ffn_moe_topk-: devono essere gli strati 0..K-1, nessun buco
                if (!contigui) {
                    muori("prefisso \"%s\": strati non contigui da 0", p.c_str());
                }
                jp["strati"] = (int) indici.size();
            }
            per_prefisso[p] = jp;
        }
        if (k == 0) {
            nomi_prima_seq = nomi_visti;
        } else if (nomi_visti != nomi_prima_seq) {
            muori("sequenza %zu: insieme di tensori diverso dalla sequenza 0", k);
        }
        nll_pp_tot += nll_pp;
        n_pp_tot += c_pp;
        const double secondi = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_seq).count();
        json js = {
            {"indice", k}, {"cartella", nome_seq}, {"sorgente", seqs[k].sorgente}, {"offset_token", seqs[k].offset},
            {"n_token", n}, {"fnv1a64_ids", esadecimale(fnv1a64(ids.data(), ids.size() * 4))},
            {"secondi", secondi}, {"tensori", man_t}, {"prefissi", per_prefisso},
        };
        if (!o.file_segmenti.empty()) {
            js["segmenti"] = seqs[k].inizi;  // 26/9: inizi dei segmenti, locali alla sequenza
        }
        if (o.logit) {
            js["logit"] = {
                {"file", std::string(nome_seq) + "/logit.f32"}, {"righe", n_uscite},
                {"posizione_prima", posizioni.front()}, {"posizione_ultima", posizioni.back()},
                {"contigue", posizioni.back() - posizioni.front() + 1 == n_uscite},
            };
            js["nll"] = {
                {"file", std::string(nome_seq) + "/nll.f64"},
                {"somma", nll_tutte}, {"n", c_tutte},
                {"somma_meta", nll_meta}, {"n_meta", c_meta},
                {"somma_stile_perplexity", nll_pp}, {"n_stile_perplexity", c_pp},
            };
        }
        man_seqs.push_back(js);
        fprintf(stderr, "sequenza %zu/%zu: %d token, %d righe di logit, %zu tensori, %.2f s\n",
                k + 1, seqs.size(), n, o.logit ? n_uscite : 0, r.flussi.size(), secondi);
    }
    llama_batch_free(lotto);
    if (fk && fclose(fk) != 0) {
        muori("chiusura di %s fallita", o.kld_base.c_str());
    }

    json j;
    j["versione"] = VERSIONE;
    std::vector<std::string> args(argv, argv + argc);
    j["argv"] = args;
    j["modello"] = {{"percorso", o.modello}, {"n_vocab", n_vocab}, {"n_embd", llama_model_n_embd(modello)},
                    {"n_layer", llama_model_n_layer(modello)}, {"add_bos", llama_vocab_get_add_bos(vocab)}};
    j["dispositivi_registrati"] = dispositivi;
    j["backend"] = o.vulkan ? "vulkan (-ngl 999)" : "cpu (nessun dispositivo di offload)";
    j["contesto"] = {{"n_ctx_richiesto", n_ctx_rich}, {"n_ctx", llama_n_ctx(ctx)}, {"n_batch", n_batch},
                     {"n_ubatch", n_ubatch}, {"n_thread", o.n_thread}, {"flash_attn_type", "auto (default)"},
                     {"type_k", "f16"}, {"type_v", "f16"}, {"cb_eval", con_tensori},
                     {"salva_tipo", o.salva_tipo}, {"segmenti", !o.file_segmenti.empty()}};
    j["uscite"] = o.uscite == "da" ? "da:" + std::to_string(o.uscite_da) : o.uscite;
    j["tensori_richiesti"] = {{"esatti", r.esatti}, {"prefissi", r.prefissi}};
    j["sequenze"] = man_seqs;
    if (fk) {
        j["kld_base"] = {{"file", o.kld_base}, {"n_ctx", seqs[0].ids.size()}, {"n_chunk", seqs.size()},
                         {"nll_stile_perplexity", nll_pp_tot}, {"n", n_pp_tot},
                         {"ppl", std::exp(nll_pp_tot / std::max<int64_t>(n_pp_tot, 1))}};
    }
    j["rss_max_kib"] = rss_max_kib();
    j["secondi_totali"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_inizio).count();
    j["esito"] = "ok";

    llama_free(ctx);
    llama_model_free(modello);
    llama_backend_free();

    const std::string tmp = o.cartella + "/manifesto.json.tmp";
    const std::string s = j.dump(1);
    scrivi_file(tmp, s.data(), s.size());
    if (rename(tmp.c_str(), (o.cartella + "/manifesto.json").c_str()) != 0) {
        muori("rename del manifesto fallito");
    }
    fprintf(stderr, "FATTO: %zu sequenze in %s (RSS max %ld KiB)\n", seqs.size(), o.cartella.c_str(), rss_max_kib());
    return 0;
}

int main(int argc, char ** argv) {
    if (argc >= 2 && std::string(argv[1]) == "logit") {
        return logit(argc, argv);
    }
    if (argc >= 2 && std::string(argv[1]) == "tokenizza") {
        return tokenizza(argc, argv);
    }
    fprintf(stderr,
            "%s\nuso:\n"
            "  %s logit -m MODELLO.gguf -o CARTELLA [-t N] [-c N] [-b N] [-ub N] [--lunghezza L]\n"
            "        [--uscite tutte|meta|ultima|da:P] [--l-out tutti|0,5,39] [--topk] [--tensore NOME|PREFISSO*]\n"
            "        [--no-logit] [--kld-base FILE] [--vulkan] [--salva-tipo f32|f16|bf16]\n"
            "        [--segmenti INIZI.i32 ...] SEQ.i32 [SEQ.i32 ...]\n"
            "  %s tokenizza MODELLO.gguf TESTO N_CTX CARTELLA [N_CHUNK_MAX]\n"
            "(formato di uscita documentato in testa a e2e_dump_logit.cpp)\n",
            VERSIONE, argv[0], argv[0]);
    return 2;
}
