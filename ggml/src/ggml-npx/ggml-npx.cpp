#include "ggml-impl.h"
#include "ggml-npx.h"
#include "ggml-backend-impl.h"

#include <cstring>

// ============================================================================
 // SKELETON ONLY — this backend currently performs NO computation on the NPX.
 //
 // While GGML_NPX_OFFLOAD_MATMUL == 0 the device registers and is visible to the
 // scheduler, but it claims no compute ops (supports_op returns false for MUL_MAT),
 // so every operation runs on the ARM CPU and inference results are unchanged.
 //
// Bring-up path (see plan):
//   1. Implement ggml_backend_npx_mul_mat() against the vendor NPX runtime SDK.
//   2. Flip GGML_NPX_OFFLOAD_MATMUL to 1 to let the scheduler route FP16
//      matmuls to the NPX (everything else stays on the CPU).
// ============================================================================
#define GGML_NPX_OFFLOAD_MATMUL 0

struct ggml_backend_npx_context {
    // host-side worker threads (e.g. for activation/weight packing before submit)
    int n_threads = GGML_DEFAULT_N_THREADS;

    // TODO: handle(s) returned by the vendor NPX runtime SDK
    // e.g. npx_device_t   device  = nullptr;
    //      npx_context_t  ctx     = nullptr;
    //      npx_stream_t   stream  = nullptr;
    void * npx_device = nullptr;
};

// ----------------------------------------------------------------------------
// op implementations
// ----------------------------------------------------------------------------

// FP16 matrix multiply: dst = src0(weights, F16) x src1(activations).
// TODO: implement against the vendor NPX runtime SDK.
//  - src0: weights, GGML_TYPE_F16, contiguous, shape [ne00(K), ne01(N), ne02, ne03]
//  - src1: activations, GGML_TYPE_F32, contiguous, shape [ne10(K), ne11(M), ...]
//  - dst : GGML_TYPE_F32, shape [ne0(N), ne1(M), ...]
//  Convert/submit activations as FP16, run the NPX GEMM, write F32 results to dst->data.
//  Because the buffer type is host memory (see get_buffer_type), src/dst data
//  pointers are directly readable/writable by both the ARM host and the NPX.
static void ggml_backend_npx_mul_mat(ggml_backend_npx_context * ctx, struct ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("ggml-npx: ggml_backend_npx_mul_mat is not implemented yet");
}

// ----------------------------------------------------------------------------
// backend interface
// ----------------------------------------------------------------------------

static const char * ggml_backend_npx_get_name(ggml_backend_t backend) {
    return "NPX";

    GGML_UNUSED(backend);
}

static void ggml_backend_npx_free(ggml_backend_t backend) {
    ggml_backend_npx_context * ctx = (ggml_backend_npx_context *)backend->context;
    // TODO: release vendor SDK handles held in ctx
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_npx_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_npx_context * ctx = (ggml_backend_npx_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_npx_mul_mat(ctx, node);
                break;

            // pass-through ops carry no computation
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

static const struct ggml_backend_i npx_backend_i = {
    /* .get_name                = */ ggml_backend_npx_get_name,
    /* .free                    = */ ggml_backend_npx_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_npx_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_npx_guid(void) {
    // unique 16-byte identifier for the NPX backend (randomly chosen)
    static ggml_guid guid = { 0x7f, 0x4d, 0x21, 0xc6, 0x9a, 0x0b, 0x48, 0xe2,
                              0xb3, 0x55, 0x10, 0x6e, 0xa7, 0xd9, 0x2c, 0x84 };
    return &guid;
}

ggml_backend_t ggml_backend_npx_init(void) {
    ggml_backend_npx_context * ctx = new ggml_backend_npx_context;

    // TODO: open the vendor NPX device here and store handles in ctx

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_npx_guid(),
        /* .iface   = */ npx_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_npx_reg(), 0),
        /* .context = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_npx(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_npx_guid());
}

// ----------------------------------------------------------------------------
// device interface
// ----------------------------------------------------------------------------

static const char * ggml_backend_npx_device_get_name(ggml_backend_dev_t dev) {
    return "NPX";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_npx_device_get_description(ggml_backend_dev_t dev) {
    return "NPX";

    GGML_UNUSED(dev);
}

static void ggml_backend_npx_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // TODO: query the NPX/host shared memory budget from the vendor SDK
    *free  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_npx_device_get_type(ggml_backend_dev_t dev) {
    // ACCEL: an accelerator that offloads selected ops while the CPU owns the rest.
    // This also makes llama.cpp register the device ahead of the CPU backend.
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_npx_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_npx_device_get_name(dev);
    props->description = ggml_backend_npx_device_get_description(dev);
    props->type        = ggml_backend_npx_device_get_type(dev);
    ggml_backend_npx_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_npx_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_npx_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_npx_device_get_buffer_type(ggml_backend_dev_t dev) {
    // Use host (CPU) memory so tensors are shared with the ARM host with zero copies.
    // On an SoC where the NPU and CPU share DRAM this is the natural choice.
    // TODO (phase 2 / W4A16): a dedicated buffer type can repack weights into the
    // NPU's native quantized layout at load time (init_tensor/set_tensor).
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_npx_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_npx_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
        {
#if GGML_NPU_OFFLOAD_MATMUL
            const struct ggml_tensor * src0 = op->src[0]; // weights
            const struct ggml_tensor * src1 = op->src[1]; // activations

            const int64_t ne10 = src1->ne[0]; // K
            const int64_t ne0  = op->ne[0];    // N
            const int64_t ne1  = op->ne[1];    // M

            // TODO: tune the minimum problem size that is worth offloading to the NPU
            const int64_t min_batch = 32;

            // Phase 1: FP16 weights only. Activations arrive as F32 from ggml and are
            // converted to FP16 for the NPX inside ggml_backend_npx_mul_mat().
            return src0->type == GGML_TYPE_F16 &&
                   src1->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch;
#else
            return false;
#endif
        }

        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_npx_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    // we operate on host memory (see get_buffer_type)
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

#if GGML_NPU_OFFLOAD_MATMUL
static bool ggml_backend_npx_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    // Pull matmuls onto the NPX even when their weights live in CPU/host memory.
    return op->op == GGML_OP_MUL_MAT && ggml_backend_npx_device_supports_op(dev, op);
}
#endif

static const struct ggml_backend_device_i ggml_backend_npx_device_i = {
    /* .get_name             = */ ggml_backend_npx_device_get_name,
    /* .get_description      = */ ggml_backend_npx_device_get_description,
    /* .get_memory           = */ ggml_backend_npx_device_get_memory,
    /* .get_type             = */ ggml_backend_npx_device_get_type,
    /* .get_props            = */ ggml_backend_npx_device_get_props,
    /* .init_backend         = */ ggml_backend_npx_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_npx_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_npx_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_npx_device_supports_op,
    /* .supports_buft        = */ ggml_backend_npx_device_supports_buft,
#if GGML_NPU_OFFLOAD_MATMUL
    /* .offload_op           = */ ggml_backend_npx_device_offload_op,
#else
    /* .offload_op           = */ NULL,
#endif
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// ----------------------------------------------------------------------------
// backend registry interface
// ----------------------------------------------------------------------------

static const char * ggml_backend_npx_reg_get_name(ggml_backend_reg_t reg) {
    return "NPX";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_npx_reg_get_device_count(ggml_backend_reg_t reg) {
    // TODO: query the number of NPUs from the vendor SDK
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_npx_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_npx_device = {
        /* .iface   = */ ggml_backend_npx_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_npx_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static const struct ggml_backend_reg_i ggml_backend_npx_reg_i = {
    /* .get_name         = */ ggml_backend_npx_reg_get_name,
    /* .get_device_count = */ ggml_backend_npx_reg_get_device_count,
    /* .get_device       = */ ggml_backend_npx_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_npx_reg(void) {
    static struct ggml_backend_reg ggml_backend_npx_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_npx_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_npx_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_npx_reg)
