#pragma once

#include "sd_loader.h"
#include "ggml.h"

namespace sd {

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
    const Model & model() const { return m; }

private:
    Model m;
    UNetConfig cfg;
};

}


