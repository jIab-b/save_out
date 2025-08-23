#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <cstring>
#include <string>
#include <unordered_map>

namespace sd {

struct Model {
    ggml_context * ctx = nullptr;
    gguf_context * uf = nullptr;
    std::unordered_map<std::string, ggml_tensor *> tensors;
    std::unordered_map<std::string, std::string> kv;
};

struct VAEConfig {
    int32_t in_channels = 0;
    int32_t latent_channels = 0;
    float scaling_factor = 0.18215f;
};

class VAE {
public:
    static VAE from_file(const std::string & path);
    ggml_tensor * decode(ggml_context * ctx, ggml_tensor * latents) const;
    const VAEConfig & config() const;
    const Model & model() const { return m; }
private:
    Model m;
    VAEConfig cfg;
};

const VAEConfig & VAE::config() const { return cfg; }

Model load(const std::string & path);

static int32_t get_i(const Model & m, const std::string & k, int32_t d) {
    auto it = m.kv.find(k);
    if (it == m.kv.end()) return d;
    try { return std::stoi(it->second); } catch (...) { return d; }
}

static float get_f(const Model & m, const std::string & k, float d) {
    auto it = m.kv.find(k);
    if (it == m.kv.end()) return d;
    try { return std::stof(it->second); } catch (...) { return d; }
}

VAE VAE::from_file(const std::string & path) {
    VAE v;
    v.m = load(path);
    v.cfg.in_channels = get_i(v.m, "diffusion.vae.in_channels", 0);
    v.cfg.latent_channels = get_i(v.m, "diffusion.vae.latent_channels", 4);
    v.cfg.scaling_factor = get_f(v.m, "diffusion.vae.scaling_factor", 0.18215f);
    return v;
}

ggml_tensor * VAE::decode(ggml_context * ctx, ggml_tensor * latents) const {
    auto w_in_it = m.tensors.find("decoder.conv_in.weight");
    auto b_in_it = m.tensors.find("decoder.conv_in.bias");
    auto w_out_it = m.tensors.find("decoder.conv_out.weight");
    auto b_out_it = m.tensors.find("decoder.conv_out.bias");
    if (w_in_it == m.tensors.end() || b_in_it == m.tensors.end() || w_out_it == m.tensors.end() || b_out_it == m.tensors.end()) {
        return latents;
    }

    ggml_tensor * w_in_src  = w_in_it->second;
    ggml_tensor * b_in_src  = b_in_it->second;
    ggml_tensor * w_out_src = w_out_it->second;
    ggml_tensor * b_out_src = b_out_it->second;

    ggml_tensor * k_in  = ggml_permute(ctx, w_in_src, 3, 2, 1, 0);
    ggml_tensor * k_out = ggml_permute(ctx, w_out_src, 3, 2, 1, 0);
    ggml_tensor * cur = ggml_conv_2d_s1_ph(ctx, k_in, latents);
    cur = ggml_add(ctx, cur, b_in_src);
    cur = ggml_silu(ctx, cur);
    cur = ggml_conv_2d_s1_ph(ctx, k_out, cur);
    cur = ggml_add(ctx, cur, b_out_src);
    cur = ggml_tanh(ctx, cur);
    // return node for caller to compute
    return cur;
}

}
