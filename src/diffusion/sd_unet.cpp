#include "sd_unet.h"

#include <stdexcept>

namespace sd {

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
                                   ggml_tensor * /*timestep*/,
                                   ggml_tensor * /*text_context*/) const {
    auto w_in_it  = m.tensors.find("conv_in.weight");
    auto b_in_it  = m.tensors.find("conv_in.bias");
    auto w_out_it = m.tensors.find("conv_out.weight");
    auto b_out_it = m.tensors.find("conv_out.bias");
    if (w_in_it == m.tensors.end() || b_in_it == m.tensors.end() || w_out_it == m.tensors.end() || b_out_it == m.tensors.end()) {
        return ggml_dup_tensor(ctx, latents);
    }

    ggml_tensor * w_in  = w_in_it->second;
    ggml_tensor * b_in  = b_in_it->second;
    ggml_tensor * w_out = w_out_it->second;
    ggml_tensor * b_out = b_out_it->second;

    ggml_tensor * k_in  = ggml_permute(ctx, w_in, 3, 2, 1, 0);
    ggml_tensor * k_out = ggml_permute(ctx, w_out, 3, 2, 1, 0);

    ggml_tensor * cur = ggml_conv_2d_s1_ph(ctx, k_in, latents);
    cur = ggml_add(ctx, cur, b_in);
    cur = ggml_silu(ctx, cur);
    cur = ggml_conv_2d_s1_ph(ctx, k_out, cur);
    cur = ggml_add(ctx, cur, b_out);
    return cur;
}

}


