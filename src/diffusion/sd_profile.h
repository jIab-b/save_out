#pragma once

#include <string>

namespace sd {

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

}


