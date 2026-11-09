// dump_strati — salva per INTERO (tutti i token, tutte le dimensioni) i tensori
// intermedi del grafo llama.cpp scelti per nome, per il confronto di fedelta'
// torch <-> llama.cpp (audit B9, 24/9/2026).
//
// Perche' non llama-eval-callback: stampa solo 3 valori per estremo di ogni asse
// e la somma; qui serve l'errore relativo PER TOKEN su tutti i 1024 token.
//
// Uso:
//   dump_strati MODELLO.gguf TOKENS.i32 CARTELLA_OUT N_THREAD NOME [NOME...]
//   (TOKENS.i32 = int32 grezzi, gli stessi id che vede il ramo torch: niente
//    tokenizzazione qui, cosi' le due parti leggono gli STESSI token)
// Ogni tensore NOME finisce in CARTELLA_OUT/NOME.f32:
//   intestazione 4 x int64 (ne0 ne1 ne2 ne3) + dati float32 contigui.
//
// Default: SOLO backend CPU (nessuna contesa con la GPU); DUMP_VULKAN=1 per il
// percorso di produzione.

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

struct Raccolta {
    std::set<std::string> nomi;
    std::set<std::string> scritti;
    std::string cartella;
    std::vector<uint8_t> buf;
    bool errore = false;
};

static bool richiamo(struct ggml_tensor * t, bool chiedi, void * dati) {
    auto * r = static_cast<Raccolta *>(dati);
    const bool voluto = r->nomi.count(t->name) > 0;
    if (chiedi) {
        return voluto;
    }
    if (!voluto) {
        return true;
    }
    if (r->scritti.count(t->name)) {
        // un secondo passaggio vorrebbe dire micro-batch multipli: il file
        // conterrebbe solo una parte dei token. Meglio fermarsi che mentire.
        fprintf(stderr, "ERRORE: %s visto due volte (n_ubatch < n_token?)\n", t->name);
        r->errore = true;
        return false;
    }
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        fprintf(stderr, "ERRORE: %s tipo %s contiguo=%d: atteso f32 contiguo\n",
                t->name, ggml_type_name(t->type), (int) ggml_is_contiguous(t));
        r->errore = true;
        return false;
    }
    const size_t n = ggml_nbytes(t);
    r->buf.resize(n);
    if (ggml_backend_buffer_is_host(t->buffer)) {
        memcpy(r->buf.data(), t->data, n);
    } else {
        ggml_backend_tensor_get(t, r->buf.data(), 0, n);
    }
    const std::string percorso = r->cartella + "/" + t->name + ".f32";
    std::ofstream f(percorso, std::ios::binary);
    const int64_t ne[4] = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
    f.write(reinterpret_cast<const char *>(ne), sizeof(ne));
    f.write(reinterpret_cast<const char *>(r->buf.data()), (std::streamsize) n);
    if (!f) {
        fprintf(stderr, "ERRORE: scrittura %s fallita\n", percorso.c_str());
        r->errore = true;
        return false;
    }
    r->scritti.insert(t->name);
    fprintf(stderr, "scritto %s [%lld %lld %lld %lld]\n", t->name,
            (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3]);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "uso: %s MODELLO.gguf TOKENS.i32 CARTELLA_OUT N_THREAD NOME [NOME...]\n", argv[0]);
        return 2;
    }
    Raccolta r;
    r.cartella = argv[3];
    const int n_thread = std::stoi(argv[4]);
    for (int i = 5; i < argc; ++i) {
        r.nomi.insert(argv[i]);
    }

    std::ifstream ft(argv[2], std::ios::binary | std::ios::ate);
    if (!ft) {
        fprintf(stderr, "ERRORE: non leggo %s\n", argv[2]);
        return 2;
    }
    const std::streamsize byte = ft.tellg();
    ft.seekg(0);
    std::vector<llama_token> token(byte / sizeof(int32_t));
    ft.read(reinterpret_cast<char *>(token.data()), byte);
    const int n_tok = (int) token.size();
    fprintf(stderr, "token: %d\n", n_tok);

    llama_backend_init();
    // DUMP_VULKAN=1 -> percorso di PRODUZIONE (tutto sulla GPU, -ngl 999);
    // default: SOLO CPU, nessun dispositivo di offload (ne' pesi ne' operazioni:
    // con n_gpu_layers=0 llama.cpp sposta comunque i matmul grandi su Vulkan).
    const bool vulkan = getenv("DUMP_VULKAN") && std::string(getenv("DUMP_VULKAN")) == "1";
    static ggml_backend_dev_t nessun_dispositivo[1] = {nullptr};
    auto pm = llama_model_default_params();
    pm.n_gpu_layers = vulkan ? 999 : 0;
    if (!vulkan) {
        pm.devices = nessun_dispositivo;
    }
    llama_model * modello = llama_model_load_from_file(argv[1], pm);
    if (!modello) {
        fprintf(stderr, "ERRORE: modello non caricato\n");
        return 1;
    }

    auto pc = llama_context_default_params();
    pc.n_ctx = n_tok + 16;
    pc.n_batch = n_tok;
    pc.n_ubatch = n_tok;  // UN solo micro-batch: ogni tensore passa una volta sola
    pc.n_threads = n_thread;
    pc.n_threads_batch = n_thread;
    pc.op_offload = vulkan;
    pc.offload_kqv = vulkan;
    pc.cb_eval = richiamo;
    pc.cb_eval_user_data = &r;
    llama_context * ctx = llama_init_from_model(modello, pc);
    if (!ctx) {
        fprintf(stderr, "ERRORE: contesto non creato\n");
        return 1;
    }

    llama_batch lotto = llama_batch_get_one(token.data(), n_tok);
    const int rc = llama_decode(ctx, lotto);
    llama_free(ctx);
    llama_model_free(modello);
    llama_backend_free();

    if (rc != 0 || r.errore) {
        fprintf(stderr, "ERRORE: decode rc=%d errore=%d\n", rc, (int) r.errore);
        return 1;
    }
    for (const auto & nome : r.nomi) {
        if (!r.scritti.count(nome)) {
            fprintf(stderr, "ERRORE: tensore %s mai visto nel grafo\n", nome.c_str());
            return 1;
        }
    }
    fprintf(stderr, "FATTO: %zu tensori\n", r.scritti.size());
    return 0;
}
