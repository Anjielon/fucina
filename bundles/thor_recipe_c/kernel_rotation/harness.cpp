// Banco Vulkan isolato per il matvec TQ1_0 (con/senza de-rotazione Hadamard
// k=256). NON e' integrato in ggml/llama.cpp: e' un dispatcher Vulkan grezzo,
// scritto apposta per misurare SOLO il costo del kernel (dequant [+ FWHT] +
// prodotto scalare), come richiesto dal compito ("isolato, non serve
// caricare un modello intero").
//
// Uso:
//   harness <shader.spv> <weights.bin> <x.bin> <R> <blocks_per_row> <y_out.bin> <iters>
//
// weights.bin: R * blocks_per_row * 54 byte (blocchi TQ1_0 grezzi, per riga)
// x.bin      : blocks_per_row * 256 float32
// y_out.bin  : R float32 (scritto a fine run, per il confronto con numpy)
//
// Stampa su stdout, in ordine:
//   MS_TOTAL <ms totali per gli 'iters' dispatch, mediana su 5 prove>
//   MS_PER_ITER <ms per singolo dispatch>
//   ROWS_PER_S <righe/s>
//   GFLOPS <stima: 2*R*C flop/dispatch, piu' 8*R*C/256 flop di FWHT se derot>

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <fstream>
#include <algorithm>
#include <chrono>

#define VK_CHECK(x) do { VkResult r = (x); if (r != VK_SUCCESS) { \
    fprintf(stderr, "Vulkan error %d at %s:%d (%s)\n", r, __FILE__, __LINE__, #x); exit(1); } } while (0)

static std::vector<uint8_t> read_file(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "impossibile aprire %s\n", path); exit(1); }
    size_t n = (size_t)f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(n);
    f.read(reinterpret_cast<char*>(buf.data()), n);
    return buf;
}

static uint32_t find_memory_type(VkPhysicalDevice pdev, uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    fprintf(stderr, "nessun memory type con le proprieta' richieste\n");
    exit(1);
}

struct Buf {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
};

static Buf make_buffer(VkDevice dev, VkPhysicalDevice pdev, VkDeviceSize size, VkBufferUsageFlags usage) {
    Buf b; b.size = size;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size; bi.usage = usage; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(dev, &bi, nullptr, &b.buf));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, b.buf, &mr);
    VkMemoryPropertyFlags props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t mti = find_memory_type(pdev, mr.memoryTypeBits, props);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size; ai.memoryTypeIndex = mti;
    VK_CHECK(vkAllocateMemory(dev, &ai, nullptr, &b.mem));
    VK_CHECK(vkBindBufferMemory(dev, b.buf, b.mem, 0));
    VK_CHECK(vkMapMemory(dev, b.mem, 0, size, 0, &b.mapped));
    return b;
}

int main(int argc, char** argv) {
    if (argc != 8) {
        fprintf(stderr, "uso: %s shader.spv weights.bin x.bin R blocks_per_row y_out.bin iters\n", argv[0]);
        return 1;
    }
    const char* shader_path = argv[1];
    const char* weights_path = argv[2];
    const char* x_path = argv[3];
    uint32_t R = (uint32_t)atoi(argv[4]);
    uint32_t blocks_per_row = (uint32_t)atoi(argv[5]);
    const char* y_out_path = argv[6];
    uint32_t iters = (uint32_t)atoi(argv[7]);

    auto spirv = read_file(shader_path);
    auto weights = read_file(weights_path);
    auto xdata = read_file(x_path);

    uint32_t C = blocks_per_row * 256u;
    if (weights.size() != (size_t)R * blocks_per_row * 54u) {
        fprintf(stderr, "weights.bin dimensione inattesa: %zu attesi %zu\n",
                weights.size(), (size_t)R * blocks_per_row * 54u);
        return 1;
    }
    if (xdata.size() != (size_t)C * sizeof(float)) {
        fprintf(stderr, "x.bin dimensione inattesa\n"); return 1;
    }

    // ── instance ──
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "tq1-derot-bench"; app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance instance;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));

    uint32_t ndev = 0; vkEnumeratePhysicalDevices(instance, &ndev, nullptr);
    std::vector<VkPhysicalDevice> pdevs(ndev);
    vkEnumeratePhysicalDevices(instance, &ndev, pdevs.data());
    VkPhysicalDevice pdev = VK_NULL_HANDLE;
    for (auto d : pdevs) {
        VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(d, &props);
        fprintf(stderr, "GPU trovata: %s (tipo=%d)\n", props.deviceName, props.deviceType);
        if (pdev == VK_NULL_HANDLE) pdev = d;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
            props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) pdev = d;
    }
    if (pdev == VK_NULL_HANDLE) { fprintf(stderr, "nessuna GPU Vulkan\n"); return 1; }

    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(pdev, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qprops(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(pdev, &nq, qprops.data());
    uint32_t qfam = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++) {
        if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; break; }
    }
    if (qfam == UINT32_MAX) { fprintf(stderr, "nessuna coda compute\n"); return 1; }
    uint32_t timestamp_valid_bits = qprops[qfam].timestampValidBits;

    float qprio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfam; qci.queueCount = 1; qci.pQueuePriorities = &qprio;

    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.shaderInt8 = VK_TRUE;
    f12.storageBuffer8BitAccess = VK_TRUE;
    f12.uniformAndStorageBuffer8BitAccess = VK_TRUE;

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f12;
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    VkDevice dev;
    VK_CHECK(vkCreateDevice(pdev, &dci, nullptr, &dev));
    VkQueue queue; vkGetDeviceQueue(dev, qfam, 0, &queue);

    VkPhysicalDeviceProperties pprops; vkGetPhysicalDeviceProperties(pdev, &pprops);
    double timestamp_period_ns = pprops.limits.timestampPeriod;

    // ── buffer ──
    Buf wbuf = make_buffer(dev, pdev, weights.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buf xbuf = make_buffer(dev, pdev, xdata.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buf ybuf = make_buffer(dev, pdev, (VkDeviceSize)R * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    memcpy(wbuf.mapped, weights.data(), weights.size());
    memcpy(xbuf.mapped, xdata.data(), xdata.size());
    memset(ybuf.mapped, 0, (size_t)R * sizeof(float));

    // ── shader module ──
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = spirv.size(); smi.pCode = reinterpret_cast<const uint32_t*>(spirv.data());
    VkShaderModule shader;
    VK_CHECK(vkCreateShaderModule(dev, &smi, nullptr, &shader));

    // ── descriptor set layout: 3 SSBO ──
    VkDescriptorSetLayoutBinding binds[3]{};
    for (int i = 0; i < 3; i++) {
        binds[i].binding = i; binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1; binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 3; dslci.pBindings = binds;
    VkDescriptorSetLayout dsl;
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    VkPipelineLayout playout;
    VK_CHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &playout));

    VkPipelineShaderStageCreateInfo ssci{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT; ssci.module = shader; ssci.pName = "main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = ssci; cpci.layout = playout;
    VkPipeline pipeline;
    VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline));

    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &dps;
    VkDescriptorPool dpool;
    VK_CHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool));

    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dpool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
    VkDescriptorSet dset;
    VK_CHECK(vkAllocateDescriptorSets(dev, &dsai, &dset));

    VkDescriptorBufferInfo bi0{wbuf.buf, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo bi1{xbuf.buf, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo bi2{ybuf.buf, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet writes[3]{};
    VkDescriptorBufferInfo* infos[3] = {&bi0, &bi1, &bi2};
    for (int i = 0; i < 3; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = dset; writes[i].dstBinding = i; writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[i].pBufferInfo = infos[i];
    }
    vkUpdateDescriptorSets(dev, 3, writes, 0, nullptr);

    // ── command pool / buffer ──
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = qfam;
    VkCommandPool cmdpool;
    VK_CHECK(vkCreateCommandPool(dev, &cpi, nullptr, &cmdpool));

    VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpci.queryType = VK_QUERY_TYPE_TIMESTAMP; qpci.queryCount = 2;
    VkQueryPool qpool = VK_NULL_HANDLE;
    bool have_timestamps = timestamp_valid_bits > 0 && timestamp_period_ns > 0.0;
    if (have_timestamps) VK_CHECK(vkCreateQueryPool(dev, &qpci, nullptr, &qpool));

    auto record_and_run = [&](uint32_t n_dispatch) -> double {
        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = cmdpool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
        VkCommandBuffer cmd;
        VK_CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd));

        VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(cmd, &cbbi));
        if (have_timestamps) {
            vkCmdResetQueryPool(cmd, qpool, 0, 2);
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);
        }
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, playout, 0, 1, &dset, 0, nullptr);
        vkCmdPushConstants(cmd, playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &blocks_per_row);
        for (uint32_t it = 0; it < n_dispatch; it++) {
            vkCmdDispatch(cmd, R, 1, 1);
        }
        if (have_timestamps) {
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 1);
        }
        VK_CHECK(vkEndCommandBuffer(cmd));

        auto t0 = std::chrono::high_resolution_clock::now();
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(queue));
        auto t1 = std::chrono::high_resolution_clock::now();
        double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        double ms = wall_ms;
        if (have_timestamps) {
            uint64_t ts[2];
            VK_CHECK(vkGetQueryPoolResults(dev, qpool, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                                            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            double gpu_ms = (double)(ts[1] - ts[0]) * timestamp_period_ns / 1.0e6;
            fprintf(stderr, "  [%u dispatch] wall=%.3f ms  gpu_timestamp=%.3f ms\n", n_dispatch, wall_ms, gpu_ms);
            ms = gpu_ms;
        } else {
            fprintf(stderr, "  [%u dispatch] wall=%.3f ms (nessun timestamp GPU disponibile)\n", n_dispatch, wall_ms);
        }
        vkFreeCommandBuffers(dev, cmdpool, 1, &cmd);
        return ms;
    };

    // correttezza: un solo dispatch, poi leggo Y
    record_and_run(1);
    std::ofstream yout(y_out_path, std::ios::binary);
    yout.write(reinterpret_cast<char*>(ybuf.mapped), (size_t)R * sizeof(float));
    yout.close();

    // prestazioni: 5 prove da 'iters' dispatch ciascuna, mediana
    std::vector<double> trial_ms;
    for (int t = 0; t < 5; t++) {
        double ms = record_and_run(iters);
        trial_ms.push_back(ms / iters);
    }
    std::sort(trial_ms.begin(), trial_ms.end());
    double median_ms_per_iter = trial_ms[trial_ms.size() / 2];

    double rows_per_s = (median_ms_per_iter > 0) ? (R / (median_ms_per_iter / 1000.0)) : 0.0;
    // flop stimati: 2*R*C (fma dequant*x) + eventuale FWHT (8*R*C flop, 8 add/sub per elemento)
    bool is_derot = strstr(shader_path, "derot") != nullptr;
    double flops = 2.0 * R * C + (is_derot ? 8.0 * R * C : 0.0);
    double gflops = (median_ms_per_iter > 0) ? (flops / (median_ms_per_iter / 1000.0) / 1e9) : 0.0;

    printf("MS_PER_ITER %.5f\n", median_ms_per_iter);
    printf("ROWS_PER_S %.1f\n", rows_per_s);
    printf("GFLOPS %.3f\n", gflops);
    printf("TIMESTAMPS_USED %d\n", have_timestamps ? 1 : 0);

    return 0;
}
