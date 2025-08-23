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
        return ggml_dup_tensor(ctx, latents);
    }

    const int64_t ne0 = latents->ne[0];
    const int64_t ne1 = latents->ne[1];
    const int64_t ne2 = latents->ne[2];
    const int64_t ne3 = latents->ne[3];
    ggml_tensor * lat_in = ggml_new_tensor_4d(ctx, latents->type, ne0, ne1, ne2, ne3);
    const size_t nbytes = ggml_nbytes(latents);
    std::memcpy(lat_in->data, latents->data, nbytes);

    ggml_tensor * w_in_src  = w_in_it->second;
    ggml_tensor * b_in_src  = b_in_it->second;
    ggml_tensor * w_out_src = w_out_it->second;
    ggml_tensor * b_out_src = b_out_it->second;

    ggml_tensor * w_in  = ggml_dup_tensor(ctx, w_in_src);
    ggml_tensor * b_in  = ggml_dup_tensor(ctx, b_in_src);
    ggml_tensor * w_out = ggml_dup_tensor(ctx, w_out_src);
    ggml_tensor * b_out = ggml_dup_tensor(ctx, b_out_src);
    std::memcpy(w_in->data,  w_in_src->data,  ggml_nbytes(w_in_src));
    std::memcpy(b_in->data,  b_in_src->data,  ggml_nbytes(b_in_src));
    std::memcpy(w_out->data, w_out_src->data, ggml_nbytes(w_out_src));
    std::memcpy(b_out->data, b_out_src->data, ggml_nbytes(b_out_src));

    ggml_tensor * k_in  = ggml_permute(ctx, w_in, 3, 2, 1, 0);
    ggml_tensor * k_out = ggml_permute(ctx, w_out, 3, 2, 1, 0);
    ggml_tensor * cur = ggml_conv_2d_s1_ph(ctx, k_in, lat_in);
    cur = ggml_add(ctx, cur, b_in);
    cur = ggml_silu(ctx, cur);
    cur = ggml_conv_2d_s1_ph(ctx, k_out, cur);
    cur = ggml_add(ctx, cur, b_out);
    cur = ggml_tanh(ctx, cur);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cur);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    return cur;
}

}
