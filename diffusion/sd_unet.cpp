#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <stdexcept>
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

struct UNetConfig {
    int32_t in_channels = 0;
    int32_t out_channels = 0;
    int32_t sample_size = 0;
};

class UNet {
public:
    static UNet from_file(const std::string & path);
    ggml_tensor * predict_noise(ggml_context * ctx,
                                ggml_tensor * latents,
                                ggml_tensor * timestep,
                                ggml_tensor * text_context) const;
    const UNetConfig & config() const { return cfg; }
    const Model & model() const;
private:
    Model m;
    UNetConfig cfg;
};

const Model & UNet::model() const { return m; }

Model load(const std::string & path);

static int32_t get_i(const Model & m, const std::string & k, int32_t d) {
    auto it = m.kv.find(k);
    if (it == m.kv.end()) return d;
    try { return std::stoi(it->second); } catch (...) { return d; }
}

UNet UNet::from_file(const std::string & path) {
    UNet u;
    u.m = load(path);
    u.cfg.in_channels = get_i(u.m, "diffusion.unet.in_channels", 0);
    u.cfg.out_channels = get_i(u.m, "diffusion.unet.out_channels", 0);
    u.cfg.sample_size = get_i(u.m, "diffusion.unet.sample_size", 0);
    return u;
}

ggml_tensor * UNet::predict_noise(ggml_context * ctx,
                                   ggml_tensor * latents,
                                   ggml_tensor * timestep,
                                   ggml_tensor * /*text_context*/) const {
    auto w_in_it  = m.tensors.find("conv_in.weight");
    auto b_in_it  = m.tensors.find("conv_in.bias");
    auto w_out_it = m.tensors.find("conv_out.weight");
    auto b_out_it = m.tensors.find("conv_out.bias");
    if (w_in_it == m.tensors.end() || b_in_it == m.tensors.end() || w_out_it == m.tensors.end() || b_out_it == m.tensors.end()) {
        return latents;
    }

    ggml_tensor * w_in_src  = w_in_it->second;
    ggml_tensor * b_in_src  = b_in_it->second;
    ggml_tensor * w_out_src = w_out_it->second;
    ggml_tensor * b_out_src = b_out_it->second;

    // permute weights for convolution if required
    ggml_tensor * k_in  = ggml_permute(ctx, w_in_src, 3, 2, 1, 0);
    ggml_tensor * k_out = ggml_permute(ctx, w_out_src, 3, 2, 1, 0);

    ggml_tensor * cur = ggml_conv_2d_s1_ph(ctx, k_in, latents);
    cur = ggml_add(ctx, cur, b_in_src);
    cur = ggml_silu(ctx, cur);
    // minimal timestep conditioning: scale features based on normalized timestep
    float scale = 1.0f;
    if (timestep && timestep->data) {
        int32_t t = ((int32_t *)timestep->data)[0];
        int denom = std::max(1, cfg.sample_size > 0 ? cfg.sample_size : cfg.out_channels);
        scale = 1.0f - (float)t / (float)std::max(1, cfg.sample_size);
    }
    cur = ggml_scale(ctx, cur, scale);
    cur = ggml_conv_2d_s1_ph(ctx, k_out, cur);
    cur = ggml_add(ctx, cur, b_out_src);
    // Do not compute here; caller manages allocation and compute
    return cur;
}

}
