#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace sd {

struct Model {
    ggml_context * ctx = nullptr;
    gguf_context * uf = nullptr;
    std::unordered_map<std::string, ggml_tensor *> tensors;
    std::unordered_map<std::string, std::string> kv;
};

struct SchedulerConfig {
    std::string prediction_type;
    int num_train_timesteps = 1000;
    float beta_start = 0.00085f;
    float beta_end = 0.012f;
    std::string beta_schedule = "scaled_linear";
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
    std::vector<float> alpha_cumprod;
};

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
    auto itb0 = m.kv.find("diffusion.scheduler.beta_start");
    if (itb0 != m.kv.end()) { try { sch.cfg.beta_start = std::stof(itb0->second); } catch (...) {} }
    auto itb1 = m.kv.find("diffusion.scheduler.beta_end");
    if (itb1 != m.kv.end()) { try { sch.cfg.beta_end = std::stof(itb1->second); } catch (...) {} }
    auto itbs = m.kv.find("diffusion.scheduler.beta_schedule");
    if (itbs != m.kv.end()) sch.cfg.beta_schedule = itbs->second;
    sch.set_num_inference_steps(20);
    return sch;
}

void DiffusionScheduler::set_num_inference_steps(int n) {
    timesteps.clear();
    timesteps.reserve(n);
    for (int i = 0; i < n; ++i) {
        timesteps.push_back((cfg.num_train_timesteps - 1) - (cfg.num_train_timesteps - 1) * i / (n - 1));
    }
    alpha_cumprod.clear();
    alpha_cumprod.resize(cfg.num_train_timesteps);
    // simple linear betas per config
    for (int t = 0; t < cfg.num_train_timesteps; ++t) {
        float frac = (float)t / (float)std::max(1, cfg.num_train_timesteps - 1);
        float beta = cfg.beta_start + (cfg.beta_end - cfg.beta_start) * frac;
        float alpha = 1.0f - beta;
        alpha_cumprod[t] = (t == 0) ? alpha : alpha_cumprod[t - 1] * alpha;
    }
}

ggml_tensor * DiffusionScheduler::step(ggml_context * ctx, ggml_tensor * latents, ggml_tensor * noise_pred, int /*step_index*/) const {
    ggml_tensor * scaled = ggml_sub(ctx, latents, noise_pred);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, scaled);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    return scaled;
}

}
