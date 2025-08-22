#include "sd_loader.h"
#include "sd_text.h"
#include "sd_unet.h"
#include "sd_vae.h"
#include "sd_scheduler.h"
#include "sd_tokenizer.h"
#include "sd_profile.h"

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <filesystem>
#include <random>
#include <vector>

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
    ggml_tensor * emb1 = text.forward(ctx, token_ids);

    std::filesystem::path p2 = std::filesystem::path(prof.dir) / "text_encoder2.gguf";
    if (std::filesystem::exists(p2)) {
        TextEncoder t2 = TextEncoder::from_file(p2.string());
        Tokenizer tk2 = Tokenizer::from_text_model(t2.model());
        std::vector<int32_t> ids2 = tk2.encode(prompt, true);
        ggml_tensor * token_ids2 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ids2.size());
        int32_t * id2 = (int32_t *) token_ids2->data;
        for (size_t i = 0; i < ids2.size(); ++i) id2[i] = ids2[i];
        ggml_tensor * emb2 = t2.forward(ctx, token_ids2);
        return ggml_concat(ctx, emb1, emb2, 0);
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
        ggml_tensor * noise_pred = unet.predict_noise(ctx, latents, tstep, text_ctx);
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
    return vae.decode(ctx, latents);
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

    ggml_init_params ip;
    ip.mem_size   = 512ull * 1024 * 1024;
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
    const int Wout = 64, Hout = 64;
    std::vector<unsigned char> rgb((size_t)Wout * Hout * 3, 127);
    sd::write_png(out_file, Wout, Hout, rgb);

    std::printf("ok: text_ctx=(%d,%d) latents=%d image=%d\n",
        (int)text_ctx->ne[1], (int)text_ctx->ne[0], (int)latents->ne[0], (int)image->ne[0]);

    ggml_free(ctx);
    return 0;
}


