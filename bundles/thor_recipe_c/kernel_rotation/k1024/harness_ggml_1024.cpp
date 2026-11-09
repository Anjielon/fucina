// Banco di verifica del tipo GGML_TYPE_TQ1_0R_1024 (MOGAVIS Thor-v9, 20/9)
// contro il codec numpy (tq1_0r_1024_codec.py), attraverso le API ggml VERE
// (non un dispatcher Vulkan a parte): stesso ggml_quantize_chunk, stessa
// to_float, stesso ggml_mul_mat sul backend Vulkan che usera' llama-server.
//
// Uso:
//   harness quant   <W.f32>  <R> <C>            <out_packed.bin>
//   harness dequant <packed> <R> <C>            <out_W.f32>
//   harness matmul  <packed> <R> <C> <X.f32> <n> <out_Y.f32> [backend: cpu|vulkan]
// W.f32: R*C float32 riga-maggiore (riga = C pesi). X.f32: n*C float32.
// Y: n*R float32 (colonna j = W x_j), layout ggml: dst ne0=R, ne1=n.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<uint8_t> read_all(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "impossibile aprire %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(n);
    if (fread(buf.data(), 1, n, f) != (size_t) n) { fprintf(stderr, "lettura corta %s\n", path); exit(1); }
    fclose(f);
    return buf;
}
static void write_all(const char * path, const void * p, size_t n) {
    FILE * f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "impossibile scrivere %s\n", path); exit(1); }
    fwrite(p, 1, n, f); fclose(f);
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "uso: vedi intestazione\n"); return 1; }
    const std::string cmd = argv[1];
    // K1024_TYPE=<nome ggml> per usare un altro tipo come RIFERIMENTO di
    // precisione del percorso (es. q8_0: stesso mul_mm, stesso fp16acc).
    ggml_type T = GGML_TYPE_TQ1_0R_1024;
    if (const char * tn = getenv("K1024_TYPE")) {
        T = GGML_TYPE_COUNT;
        for (int i = 0; i < GGML_TYPE_COUNT; ++i) { const char * nm = ggml_type_name((ggml_type) i); if (nm && std::string(nm) == tn) { T = (ggml_type) i; break; } }
        if (T == GGML_TYPE_COUNT) { fprintf(stderr, "tipo sconosciuto %s\n", tn); return 2; }
    } else {
        if (std::string(ggml_type_name(T)) != "tq1_0r_1024") { fprintf(stderr, "tipo 45 non e' tq1_0r_1024: %s\n", ggml_type_name(T)); return 2; }
        const size_t blk = ggml_blck_size(T), tsz = ggml_type_size(T);
        if (blk != 1024 || tsz != 224) { fprintf(stderr, "blck/type_size inattesi: %zu %zu\n", blk, tsz); return 2; }
    }

    if (cmd == "quant") {
        const int64_t R = atoll(argv[3]), C = atoll(argv[4]);
        auto W = read_all(argv[2]);
        if (W.size() != (size_t)(R*C*4)) { fprintf(stderr, "taglia W errata\n"); return 1; }
        const size_t row_bytes = ggml_row_size(T, C);
        std::vector<uint8_t> out(R * row_bytes);
        const size_t n = ggml_quantize_chunk(T, (const float *) W.data(), out.data(), 0, R, C, nullptr);
        if (n != out.size()) { fprintf(stderr, "quantize_chunk: %zu byte attesi %zu\n", n, out.size()); return 1; }
        write_all(argv[5], out.data(), out.size());
        printf("QUANT_OK %zu byte\n", out.size());
        return 0;
    }
    if (cmd == "dequant") {
        const int64_t R = atoll(argv[3]), C = atoll(argv[4]);
        auto P = read_all(argv[2]);
        if (P.size() != (size_t)(R * ggml_row_size(T, C))) { fprintf(stderr, "taglia packed errata\n"); return 1; }
        std::vector<float> out(R * C);
        const auto * tr = ggml_get_type_traits(T);
        tr->to_float(P.data(), out.data(), R * C);
        write_all(argv[5], out.data(), out.size() * 4);
        printf("DEQUANT_OK\n");
        return 0;
    }
    if (cmd == "matmul") {
        const int64_t R = atoll(argv[3]), C = atoll(argv[4]);
        auto P = read_all(argv[2]);
        auto X = read_all(argv[5]);
        const int64_t n = atoll(argv[6]);
        const std::string be = argc > 8 ? argv[8] : "vulkan";
        if (P.size() != (size_t)(R * ggml_row_size(T, C)) || X.size() != (size_t)(n*C*4)) { fprintf(stderr, "taglie errate\n"); return 1; }

        ggml_backend_load_all();
        ggml_backend_t backend = nullptr;
        if (be == "cpu") {
            backend = ggml_backend_cpu_init();
        } else {
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                const auto dt = ggml_backend_dev_type(dev);
                if (dt == GGML_BACKEND_DEVICE_TYPE_GPU || dt == GGML_BACKEND_DEVICE_TYPE_IGPU) { // la 8060S e' UMA: iGPU
                    backend = ggml_backend_dev_init(dev, nullptr); break;
                }
            }
        }
        if (!backend) { fprintf(stderr, "backend %s non disponibile\n", be.c_str()); return 3; }
        printf("BACKEND %s\n", ggml_backend_name(backend));

        ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * a = ggml_new_tensor_2d(ctx, T, C, R);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, n);
        ggml_tensor * d = ggml_mul_mat(ctx, a, b);
        if (getenv("K1024_PREC_F32")) { ggml_mul_mat_set_prec(d, GGML_PREC_F32); printf("PREC F32\n"); }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) { fprintf(stderr, "alloc fallita\n"); return 3; }
        ggml_backend_tensor_set(a, P.data(), 0, P.size());
        ggml_backend_tensor_set(b, X.data(), 0, X.size());

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, d);
        if (!ggml_backend_supports_op(backend, d)) { fprintf(stderr, "MUL_MAT non supportato dal backend per questo tipo\n"); return 4; }
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute fallito\n"); return 5; }
        std::vector<float> Y(n * R);
        ggml_backend_tensor_get(d, Y.data(), 0, Y.size() * 4);
        write_all(argv[7], Y.data(), Y.size() * 4);
        printf("MATMUL_OK n=%lld\n", (long long) n);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 0;
    }
    fprintf(stderr, "comando sconosciuto\n");
    return 1;
}
