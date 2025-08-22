#include "sd_vae.h"

namespace sd {

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
        return ggml_dup_tensor(ctx, latents);
    }
    ggml_tensor * k_in  = ggml_permute(ctx, w_in_it->second, 3, 2, 1, 0);
    ggml_tensor * k_out = ggml_permute(ctx, w_out_it->second, 3, 2, 1, 0);
    ggml_tensor * cur = ggml_conv_2d_s1_ph(ctx, k_in, latents);
    cur = ggml_add(ctx, cur, b_in_it->second);
    cur = ggml_silu(ctx, cur);
    cur = ggml_conv_2d_s1_ph(ctx, k_out, cur);
    cur = ggml_add(ctx, cur, b_out_it->second);
    return cur;
}

}


