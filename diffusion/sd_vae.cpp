#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "sd_model.h"
#include <cstring>
#include <string>
#include <unordered_map>

namespace sd {

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
    ggml_tensor * k_in_cached = nullptr;
    ggml_tensor * k_out_cached = nullptr;
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
    // cache permuted conv weights once
    auto w_in_it  = v.m.tensors.find("decoder.conv_in.weight");
    auto w_out_it = v.m.tensors.find("decoder.conv_out.weight");
    if (w_in_it != v.m.tensors.end() && w_out_it != v.m.tensors.end()) {
        ggml_init_params ip_build { 0, nullptr, true };
        ggml_context * ctx_build = ggml_init(ip_build);
        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        ggml_tensor * k_in  = ggml_permute(ctx_build, w_in_it->second, 3, 2, 1, 0);
        ggml_tensor * k_out = ggml_permute(ctx_build, w_out_it->second, 3, 2, 1, 0);
        ggml_set_output(k_in);
        ggml_set_output(k_out);
        ggml_cgraph * gf = ggml_new_graph(ctx_build);
        ggml_build_forward_expand(gf, k_in);
        ggml_build_forward_expand(gf, k_out);
        ggml_gallocr_alloc_graph(galloc, gf);
        ggml_graph_compute_with_ctx(ctx_build, gf, 1);
        v.k_in_cached  = ggml_dup_tensor(v.m.ctx, k_in);
        v.k_out_cached = ggml_dup_tensor(v.m.ctx, k_out);
        std::memcpy(v.k_in_cached->data,  k_in->data,  ggml_nbytes(k_in));
        std::memcpy(v.k_out_cached->data, k_out->data, ggml_nbytes(k_out));
        ggml_gallocr_free(galloc);
        ggml_free(ctx_build);
    }
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

    ggml_tensor * k_in  = k_in_cached  ? k_in_cached  : ggml_permute(ctx, w_in_src, 3, 2, 1, 0);
    ggml_tensor * k_out = k_out_cached ? k_out_cached : ggml_permute(ctx, w_out_src, 3, 2, 1, 0);
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
