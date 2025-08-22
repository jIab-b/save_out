#pragma once

#include "sd_loader.h"
#include "ggml.h"

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

    const VAEConfig & config() const { return cfg; }
    const Model & model() const { return m; }

private:
    Model m;
    VAEConfig cfg;
};

}


