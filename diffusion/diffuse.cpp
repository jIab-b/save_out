#include "ggml.h"
#include "ggml-cpu.h"
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
        std::string unet_path;
        std::string vae_path;
        int32_t text_hidden_size = 0;
        int32_t text_num_layers = 0;
        int32_t text_num_heads = 0;
        int32_t text_max_pos = 0;
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

static ggml_tensor * encode_text_ctx(ggml_context * ctx, const sd::SDXLProfile & prof, const std::string & prompt) {
    TextEncoder text = TextEncoder::from_file(prof.text_path);
    Tokenizer tok = Tokenizer::from_text_model(text.model());
    std::vector<int32_t> ids_vec = tok.encode(prompt, true);
    ggml_tensor * token_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ids_vec.size());
    int32_t * ids = (int32_t *) token_ids->data;
    for (size_t i = 0; i < ids_vec.size(); ++i) ids[i] = ids_vec[i];
    ggml_tensor * emb1_dev = text.forward(ctx, token_ids);
    ggml_tensor * emb1 = ggml_new_tensor_2d(ctx, emb1_dev->type, emb1_dev->ne[0], emb1_dev->ne[1]);
    std::memcpy(emb1->data, emb1_dev->data, ggml_nbytes(emb1_dev));

    std::filesystem::path p2 = std::filesystem::path(prof.dir) / "text_encoder2.gguf";
    if (std::filesystem::exists(p2)) {
        TextEncoder t2 = TextEncoder::from_file(p2.string());
        Tokenizer tk2 = Tokenizer::from_text_model(t2.model());
        std::vector<int32_t> ids2 = tk2.encode(prompt, true);
        ggml_tensor * token_ids2 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ids2.size());
        int32_t * id2 = (int32_t *) token_ids2->data;
        for (size_t i = 0; i < ids2.size(); ++i) id2[i] = ids2[i];
        ggml_tensor * emb2_dev = t2.forward(ctx, token_ids2);
        return emb1; // minimal: drop second encoder to avoid concat
    }
    return emb1;
}

static ggml_tensor * denoise_latents(ggml_context * ctx, const std::string & unet_path, ggml_tensor * latents, ggml_tensor * text_ctx, int steps) {
    UNet unet = UNet::from_file(unet_path);
    DiffusionScheduler sched = DiffusionScheduler::from_model_kv(unet.model());
    sched.set_num_inference_steps(steps);
    for (int s = 0; s < steps; ++s) {
        ggml_tensor * tstep = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ((int32_t *) tstep->data)[0] = s;
        ggml_tensor * noise_pred_dev = unet.predict_noise(ctx, latents, tstep, text_ctx);
        ggml_tensor * noise_pred = ggml_new_tensor_4d(ctx, noise_pred_dev->type, noise_pred_dev->ne[0], noise_pred_dev->ne[1], noise_pred_dev->ne[2], noise_pred_dev->ne[3]);
        std::memcpy(noise_pred->data, noise_pred_dev->data, ggml_nbytes(noise_pred_dev));
        latents = sched.step(ctx, latents, noise_pred, s);
    }
    return latents;
}

static ggml_tensor * decode_latents_to_image(ggml_context * ctx, const std::string & vae_path, ggml_tensor * latents) {
    VAE vae = VAE::from_file(vae_path);
    float scale = vae.config().scaling_factor > 0 ? vae.config().scaling_factor : 1.0f;
    {
        float * d = (float *) latents->data;
        const int C = latents->ne[0];
        const int W = latents->ne[1];
        const int H = latents->ne[2];
        const size_t n = (size_t)C * W * H;
        for (size_t i = 0; i < n; ++i) d[i] = d[i] / scale;
    }
    ggml_tensor * img_dev = vae.decode(ctx, latents);
    ggml_tensor * img = ggml_new_tensor_4d(ctx, img_dev->type, img_dev->ne[0], img_dev->ne[1], img_dev->ne[2], img_dev->ne[3]);
    std::memcpy(img->data, img_dev->data, ggml_nbytes(img_dev));
    return img;
}

static size_t plan_required_mem(const sd::SDXLProfile & prof, const std::string & prompt) {
    size_t big = 4ull * 1024ull * 1024ull * 1024ull;
    ggml_init_params ip;
    ip.mem_size = big;
    ip.mem_buffer = nullptr;
    ip.no_alloc = false;
    ggml_context * tmp = ggml_init(ip);
    if (!tmp) return 2ull * 1024ull * 1024ull * 1024ull;

    ggml_tensor * text_ctx = encode_text_ctx(tmp, prof, prompt);

    const int H = 64, W = 64, C = 4;
    ggml_tensor * latents = ggml_new_tensor_4d(tmp, GGML_TYPE_F32, C, W, H, 1);

    UNet unet = UNet::from_file(prof.unet_path);
    ggml_tensor * tstep = ggml_new_tensor_1d(tmp, GGML_TYPE_I32, 1);
    ((int32_t *) tstep->data)[0] = 0;
    ggml_tensor * noise = unet.predict_noise(tmp, latents, tstep, text_ctx);

    VAE vae = VAE::from_file(prof.vae_path);
    ggml_tensor * img = vae.decode(tmp, latents);

    ggml_cgraph * g = ggml_new_graph(tmp);
    ggml_build_forward_expand(g, img);
    ggml_graph_compute_with_ctx(tmp, g, 1);

    size_t used = ggml_used_mem(tmp);
    ggml_free(tmp);

    size_t margin = 256ull * 1024ull * 1024ull;
    size_t align = 64ull * 1024ull * 1024ull;
    size_t need = used + margin;
    return ((need + align - 1) / align) * align;
}

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

    // Estimate memory requirements based on model parameters
    size_t est = estimate_memory_requirements(prof);
    size_t planned = plan_required_mem(prof, prompt);
    size_t mem_size = std::max(est, planned);
    std::printf("Allocating %zu MB of memory for GGML context\n", mem_size / (1024 * 1024));
    
    ggml_init_params ip;
    ip.mem_size   = mem_size;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = false;
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * text_ctx = encode_text_ctx(ctx, prof, prompt);

    const int H = 64, W = 64, C = 4;
    ggml_tensor * latents = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, C, W, H, 1);
    {
        std::mt19937 rng(1234);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        float * d = (float *) latents->data;
        const size_t n = (size_t)C*W*H;
        for (size_t i = 0; i < n; ++i) d[i] = nd(rng);
    }
    latents = denoise_latents(ctx, unet_path, latents, text_ctx, steps);
    ggml_tensor * image = decode_latents_to_image(ctx, vae_path, latents);
    const int Cout = (int)image->ne[0];
    const int Wout = (int)image->ne[1];
    const int Hout = (int)image->ne[2];
    std::vector<unsigned char> rgb((size_t)Wout * Hout * 3);
    float * im = (float *) image->data;
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

    std::printf("ok: text_ctx=(%d,%d) latents=%d image=%d\n",
        (int)text_ctx->ne[1], (int)text_ctx->ne[0], (int)latents->ne[0], (int)image->ne[0]);

    ggml_free(ctx);
    return 0;
}
