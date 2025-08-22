#pragma once

#include "sd_loader.h"
#include "ggml.h"

#include <string>
#include <vector>

namespace sd {

struct SchedulerConfig {
    std::string prediction_type;
    int num_train_timesteps = 1000;
};

class DiffusionScheduler {
public:
    static DiffusionScheduler from_model_kv(const Model & m);

    void set_num_inference_steps(int n);
    int steps() const { return (int)timesteps.size(); }
    int timestep_at(int i) const { return timesteps[i]; }
    const SchedulerConfig & config() const { return cfg; }

    ggml_tensor * step(ggml_context * ctx, ggml_tensor * latents, ggml_tensor * noise_pred, int step_index) const;

private:
    SchedulerConfig cfg;
    std::vector<int> timesteps;
};

}


