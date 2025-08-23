#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <string>
#include <vector>
#include <filesystem>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

namespace sd {
    struct Model; // forward decl

    struct TextEncoderConfig {
        int32_t hidden_size = 0;
        int32_t num_layers = 0;
        int32_t num_heads = 0;
        int32_t max_position_embeddings = 0;
    };
    class TextEncoder {
    public:
        static TextEncoder from_file(const std::string & path);
        ggml_tensor * build_embeddings(ggml_context * ctx, ggml_tensor * token_ids) const;
        ggml_tensor * forward(ggml_context * ctx, ggml_tensor * token_ids) const;
        const TextEncoderConfig & config() const;
        const Model & model() const;
    };

    class Tokenizer {
    public:
        static Tokenizer from_text_model(const Model & m);
        std::vector<int32_t> encode(const std::string & text, bool add_special=true) const;
        int32_t bos_id() const;
        int32_t eos_id() const;
        int32_t unk_id() const;
    };

    struct UNetConfig { int32_t in_channels=0,out_channels=0,sample_size=0; };
    class UNet {
    public:
        static UNet from_file(const std::string & path);
        ggml_tensor * predict_noise(ggml_context * ctx, ggml_tensor * latents, ggml_tensor * timestep, ggml_tensor * text_context) const;
        const UNetConfig & config() const;
        const Model & model() const;
    };

    struct VAEConfig { int32_t in_channels=0, latent_channels=0; float scaling_factor=0.18215f; };
    class VAE {
    public:
        static VAE from_file(const std::string & path);
        ggml_tensor * decode(ggml_context * ctx, ggml_tensor * latents) const;
        const VAEConfig & config() const;
        const Model & model() const;
    };

    struct SchedulerConfig {
        std::string prediction_type;
        int num_train_timesteps = 1000;
        float beta_start = 0.00085f;
        float beta_end = 0.012f;
        std::string beta_schedule = "scaled_linear";
    };
    class DiffusionScheduler {
    public:
        static DiffusionScheduler from_model_kv(const Model & m);
        void set_num_inference_steps(int n);
        int steps() const;
        int timestep_at(int i) const;
        const SchedulerConfig & config() const;
        ggml_tensor * step(ggml_context * ctx, ggml_tensor * latents, ggml_tensor * noise_pred, int step_index) const;
    };

    struct SDXLProfile {
        std::string dir;
        std::string text_path;
        std::string text2_path;
        std::string unet_path;
        std::string vae_path;
        int32_t text_hidden_size = 0;
        int32_t text_num_layers = 0;
        int32_t text_num_heads = 0;
        int32_t text_max_pos = 0;
        int32_t text2_hidden_size = 0;
        int32_t text2_num_layers = 0;
        int32_t text2_num_heads = 0;
        int32_t text2_max_pos = 0;
        int32_t unet_in_channels = 0;
        int32_t unet_out_channels = 0;
        int32_t unet_sample_size = 0;
        int32_t vae_in_channels = 0;
        int32_t vae_latent_channels = 0;
        float   vae_scaling_factor = 0.18215f;
        std::string sched_prediction_type;
        int32_t bos_id = -1;
        int32_t eos_id = -1;
        int32_t unk_id = -1;
        int32_t pad_id = -1;
        int32_t text2_bos_id = -1;
        int32_t text2_eos_id = -1;
        int32_t text2_unk_id = -1;
        int32_t text2_pad_id = -1;
    };
    SDXLProfile load_or_build_profile(const std::string & model_dir);
    SDXLProfile load_profile_json(const std::string & json_path);
    size_t estimate_memory_requirements(const SDXLProfile & profile);
}

using namespace sd;

namespace sd {
    bool write_ppm(const std::string & path, int w, int h, const std::vector<unsigned char> & rgb);
    bool write_png(const std::string & path, int w, int h, const std::vector<unsigned char> & rgb);
}

static ggml_tensor * build_text_ctx(ggml_context * ctx_build, const sd::SDXLProfile & prof, const std::string & prompt) {
    TextEncoder text = TextEncoder::from_file(prof.text_path);
    Tokenizer tok = Tokenizer::from_text_model(text.model());
    std::vector<int32_t> ids_vec = tok.encode(prompt, true);
    ggml_tensor * token_ids = ggml_new_tensor_1d(ctx_build, GGML_TYPE_I32, ids_vec.size());
    // caller fills token_ids after allocation
    (void)ids_vec;
    return text.forward(ctx_build, token_ids);
}

static ggml_tensor * build_unet_step(ggml_context * ctx_build, const UNet & unet, const DiffusionScheduler & sched,
                                     ggml_tensor * latents_in, ggml_tensor * text_ctx, int step_index) {
    ggml_tensor * tstep = ggml_new_i32(ctx_build, step_index);
    ggml_tensor * noise_pred = unet.predict_noise(ctx_build, latents_in, tstep, text_ctx);
    ggml_tensor * latents_next = sched.step(ctx_build, latents_in, noise_pred, step_index);
    return latents_next;
}

static ggml_tensor * build_decode_image(ggml_context * ctx_build, const VAE & vae, ggml_tensor * latents_in) {
    ggml_tensor * img = vae.decode(ctx_build, latents_in);
    return img;
}

// With ggml_gallocr, we rely on allocator growing as needed, so no global plan is needed.

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model_dir|profile.json> [--prompt P] [--steps N] [--out PATH]\n", argv[0]);
        return 1;
    }
    std::string arg = argv[1];
    std::string prompt = "a cat sitting on a chair";
    int steps = 20;
    std::string out_file = "out.png";
    for (int i = 2; i + 1 < argc; i += 2) {
        std::string k = argv[i];
        std::string v = argv[i+1];
        if (k == "--prompt") prompt = v;
        else if (k == "--steps") steps = std::atoi(v.c_str());
        else if (k == "--out") out_file = v;
    }
    sd::SDXLProfile prof;
    if (arg.size() > 5 && arg.substr(arg.size()-5) == ".json") {
        prof = sd::load_profile_json(arg);
    } else {
        prof = sd::load_or_build_profile(arg);
    }
    const std::string text_path = prof.text_path;
    const std::string unet_path = prof.unet_path;
    const std::string vae_path  = prof.vae_path;

    // Build/eval contexts and allocator
    ggml_init_params ip_build { 0, nullptr, true };
    ggml_context * ctx_build = ggml_init(ip_build);
    ggml_init_params ip_persist { 16ull*1024*1024, nullptr, false };
    ggml_context * ctx_persist = ggml_init(ip_persist);
    if (!ctx_build || !ctx_persist) { std::fprintf(stderr, "ctx init failed\n"); return 1; }
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());

    // Load models
    TextEncoder text = TextEncoder::from_file(text_path);
    bool has_text2 = !prof.text2_path.empty();
    TextEncoder text2;
    if (has_text2) {
        text2 = TextEncoder::from_file(prof.text2_path);
    }
    UNet unet = UNet::from_file(unet_path);
    VAE vae = VAE::from_file(vae_path);
    DiffusionScheduler sched = DiffusionScheduler::from_model_kv(unet.model());
    sched.set_num_inference_steps(steps);

    // Build + compute text context(s), then persist
    ggml_tensor * text_ids = nullptr;
    ggml_tensor * text_node = nullptr;
    ggml_tensor * text2_ids = nullptr;
    ggml_tensor * text2_node = nullptr;
    {
        Tokenizer tok = Tokenizer::from_text_model(text.model());
        std::vector<int32_t> ids_vec = tok.encode(prompt, true);
        text_ids = ggml_new_tensor_1d(ctx_build, GGML_TYPE_I32, ids_vec.size());
        text_node = text.forward(ctx_build, text_ids);
        ggml_set_output(text_node);
        if (has_text2) {
            Tokenizer tok2 = Tokenizer::from_text_model(text2.model());
            std::vector<int32_t> ids_vec2 = tok2.encode(prompt, true);
            text2_ids = ggml_new_tensor_1d(ctx_build, GGML_TYPE_I32, ids_vec2.size());
            text2_node = text2.forward(ctx_build, text2_ids);
            ggml_set_output(text2_node);
            // build a joint graph for both sequences
        }
        ggml_cgraph * gf = ggml_new_graph(ctx_build);
        ggml_build_forward_expand(gf, text_node);
        if (has_text2) ggml_build_forward_expand(gf, text2_node);
        ggml_gallocr_alloc_graph(galloc, gf);
        // fill ids
        int32_t * ids = (int32_t *) text_ids->data;
        for (size_t i = 0; i < ids_vec.size(); ++i) ids[i] = ids_vec[i];
        if (has_text2) {
            Tokenizer tok2 = Tokenizer::from_text_model(text2.model());
            std::vector<int32_t> ids_vec2 = tok2.encode(prompt, true);
            int32_t * ids2 = (int32_t *) text2_ids->data;
            for (size_t i = 0; i < ids_vec2.size(); ++i) ids2[i] = ids_vec2[i];
        }
        ggml_graph_compute_with_ctx(ctx_build, gf, 1);
    }
    ggml_tensor * text_ctx_buf = ggml_dup_tensor(ctx_persist, text_node);
    std::memcpy(text_ctx_buf->data, text_node->data, ggml_nbytes(text_node));
    ggml_tensor * text2_ctx_buf = nullptr;
    if (has_text2 && text2_node) {
        text2_ctx_buf = ggml_dup_tensor(ctx_persist, text2_node);
        std::memcpy(text2_ctx_buf->data, text2_node->data, ggml_nbytes(text2_node));
    }
    // reset build context
    ggml_free(ctx_build);
    ctx_build = ggml_init(ip_build);

    // Latents buffer in persistent ctx
    const int H = 64, W = 64, C = 4;
    ggml_tensor * latents_buf = ggml_new_tensor_4d(ctx_persist, GGML_TYPE_F32, C, W, H, 1);
    {
        std::mt19937 rng(1234);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        float * d = (float *) latents_buf->data;
        const size_t n = (size_t)C*W*H;
        for (size_t i = 0; i < n; ++i) d[i] = nd(rng);
    }

    // Denoising steps
    for (int s = 0; s < steps; ++s) {
        ggml_tensor * lat_in = ggml_new_tensor_4d(ctx_build, GGML_TYPE_F32, C, W, H, 1);
        ggml_set_input(lat_in);
        ggml_tensor * txt_in = ggml_new_tensor_2d(ctx_build, text_ctx_buf->type, text_ctx_buf->ne[0], text_ctx_buf->ne[1]);
        ggml_tensor * lat_next = build_unet_step(ctx_build, unet, sched, lat_in, txt_in, s);
        ggml_set_output(lat_next);
        ggml_cgraph * gf = ggml_new_graph(ctx_build);
        ggml_build_forward_expand(gf, lat_next);
        ggml_gallocr_alloc_graph(galloc, gf);
        std::memcpy(lat_in->data, latents_buf->data, ggml_nbytes(latents_buf));
        std::memcpy(txt_in->data, text_ctx_buf->data, ggml_nbytes(text_ctx_buf));
        ggml_graph_compute_with_ctx(ctx_build, gf, 1);
        std::memcpy(latents_buf->data, lat_next->data, ggml_nbytes(latents_buf));
        ggml_free(ctx_build);
        ctx_build = ggml_init(ip_build);
    }

    // Decode
    ggml_tensor * img_node = nullptr;
    {
        ggml_tensor * lat_in = ggml_new_tensor_4d(ctx_build, GGML_TYPE_F32, C, W, H, 1);
        float scale = vae.config().scaling_factor > 0 ? vae.config().scaling_factor : 1.0f;
        ggml_tensor * lat_scaled = ggml_scale(ctx_build, lat_in, 1.0f / scale);
        img_node = build_decode_image(ctx_build, vae, lat_scaled);
        ggml_set_output(img_node);
        ggml_cgraph * gf = ggml_new_graph(ctx_build);
        ggml_build_forward_expand(gf, img_node);
        ggml_gallocr_alloc_graph(galloc, gf);
        std::memcpy(lat_in->data, latents_buf->data, ggml_nbytes(latents_buf));
        ggml_graph_compute_with_ctx(ctx_build, gf, 1);
    }
    const int Cout = (int)img_node->ne[0];
    const int Wout = (int)img_node->ne[1];
    const int Hout = (int)img_node->ne[2];
    std::vector<unsigned char> rgb((size_t)Wout * Hout * 3);
    float * im = (float *) img_node->data;
    for (int y = 0; y < Hout; ++y) {
        for (int x = 0; x < Wout; ++x) {
            for (int c = 0; c < 3; ++c) {
                size_t idx = (size_t)c + (size_t)x * Cout + (size_t)y * Cout * Wout;
                float v = im[idx];
                v = std::min(1.0f, std::max(-1.0f, v));
                unsigned char u8 = (unsigned char)((v * 0.5f + 0.5f) * 255.0f);
                rgb[(size_t)3*(y*Wout + x) + c] = u8;
            }
        }
    }
    sd::write_png(out_file, Wout, Hout, rgb);

    std::printf("ok: text_ctx persisted, latents shape=(%d,%d,%d)\n", C, W, H);
    ggml_gallocr_free(galloc);
    ggml_free(ctx_build);
    ggml_free(ctx_persist);
    return 0;
}
