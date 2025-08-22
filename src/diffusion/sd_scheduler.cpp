#include "sd_scheduler.h"

namespace sd {

static std::string get_str_kv(const Model & m, const std::string & key, const std::string & def) {
    auto it = m.kv.find(key);
    if (it == m.kv.end()) return def;
    return it->second;
}

DiffusionScheduler DiffusionScheduler::from_model_kv(const Model & m) {
    DiffusionScheduler sch;
    sch.cfg.prediction_type = get_str_kv(m, "diffusion.scheduler.prediction_type", "epsilon");
    int n_train = 1000;
    auto it = m.kv.find("diffusion.scheduler.num_train_timesteps");
    if (it != m.kv.end()) {
        try { n_train = std::stoi(it->second); } catch (...) {}
    }
    sch.cfg.num_train_timesteps = n_train;
    sch.set_num_inference_steps(20);
    return sch;
}

void DiffusionScheduler::set_num_inference_steps(int n) {
    timesteps.clear();
    timesteps.reserve(n);
    for (int i = 0; i < n; ++i) {
        timesteps.push_back((cfg.num_train_timesteps - 1) - (cfg.num_train_timesteps - 1) * i / (n - 1));
    }
}

ggml_tensor * DiffusionScheduler::step(ggml_context * ctx, ggml_tensor * latents, ggml_tensor * noise_pred, int /*step_index*/) const {
    ggml_tensor * scaled = ggml_sub(ctx, latents, noise_pred);
    return scaled;
}

}


