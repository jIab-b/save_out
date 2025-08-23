#include "ggml.h"
#include "gguf.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

namespace fs = std::filesystem;

namespace sd {

struct Model {
    ggml_context * ctx = nullptr;
    gguf_context * uf = nullptr;
    std::unordered_map<std::string, ggml_tensor *> tensors;
    std::unordered_map<std::string, std::string> kv;
};

Model load(const std::string & path);

struct TextEncoderConfig { int32_t hidden_size=0,num_layers=0,num_heads=0,max_position_embeddings=0; };
class TextEncoder {
public:
    static TextEncoder from_file(const std::string & path);
    ggml_tensor * build_embeddings(ggml_context * ctx, ggml_tensor * token_ids) const;
    ggml_tensor * forward(ggml_context * ctx, ggml_tensor * token_ids) const;
    const TextEncoderConfig & config() const { return cfg; }
    const Model & model() const { return m; }
private:
    Model m; TextEncoderConfig cfg;
};

struct UNetConfig { int32_t in_channels=0,out_channels=0,sample_size=0; };
class UNet {
public:
    static UNet from_file(const std::string & path);
    ggml_tensor * predict_noise(ggml_context * ctx, ggml_tensor * latents, ggml_tensor * timestep, ggml_tensor * text_context) const;
    const UNetConfig & config() const { return cfg; }
    const Model & model() const { return m; }
private: Model m; UNetConfig cfg; };

struct VAEConfig { int32_t in_channels=0, latent_channels=0; float scaling_factor=0.18215f; };
class VAE {
public:
    static VAE from_file(const std::string & path);
    ggml_tensor * decode(ggml_context * ctx, ggml_tensor * latents) const;
    const VAEConfig & config() const { return cfg; }
    const Model & model() const { return m; }
private: Model m; VAEConfig cfg; };

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
size_t estimate_memory_requirements(const SDXLProfile & profile);

static bool read_json(const std::string & path, SDXLProfile & p) {
    try {
        std::ifstream f(path);
        if (!f.is_open()) return false;
        std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        size_t pos = 0;
        auto get = [&](const char * key) -> std::string {
            std::string k = std::string("\"") + key + "\"";
            size_t i = s.find(k);
            if (i == std::string::npos) return {};
            i = s.find(":", i);
            if (i == std::string::npos) return {};
            size_t j = s.find_first_of(",}\n", i+1);
            if (j == std::string::npos) j = s.size();
            return s.substr(i+1, j-i-1);
        };
        auto stripq = [](std::string v){
            size_t a = v.find('"'); size_t b = v.rfind('"');
            if (a != std::string::npos && b != std::string::npos && b > a) return v.substr(a+1, b-a-1);
            return v;
        };
        p.text_path  = stripq(get("text_path"));
        p.unet_path  = stripq(get("unet_path"));
        p.vae_path   = stripq(get("vae_path"));
        p.sched_prediction_type = stripq(get("sched_prediction_type"));
        p.text_hidden_size   = std::stoi(get("text_hidden_size"));
        p.text_num_layers    = std::stoi(get("text_num_layers"));
        p.text_num_heads     = std::stoi(get("text_num_heads"));
        p.text_max_pos       = std::stoi(get("text_max_pos"));
        p.unet_in_channels   = std::stoi(get("unet_in_channels"));
        p.unet_out_channels  = std::stoi(get("unet_out_channels"));
        p.unet_sample_size   = std::stoi(get("unet_sample_size"));
        p.vae_in_channels    = std::stoi(get("vae_in_channels"));
        p.vae_latent_channels= std::stoi(get("vae_latent_channels"));
        p.vae_scaling_factor = std::stof(get("vae_scaling_factor"));
        p.bos_id = std::stoi(get("bos_id"));
        p.eos_id = std::stoi(get("eos_id"));
        p.unk_id = std::stoi(get("unk_id"));
        p.pad_id = std::stoi(get("pad_id"));
        return true;
    } catch (...) { return false; }
}

static void write_json(const std::string & path, const SDXLProfile & p) {
    std::ofstream f(path);
    f << "{\n";
    f << "  \"text_path\": \"" << p.text_path << "\",\n";
    f << "  \"unet_path\": \"" << p.unet_path << "\",\n";
    f << "  \"vae_path\": \"" << p.vae_path << "\",\n";
    f << "  \"text_hidden_size\": " << p.text_hidden_size << ",\n";
    f << "  \"text_num_layers\": " << p.text_num_layers << ",\n";
    f << "  \"text_num_heads\": " << p.text_num_heads << ",\n";
    f << "  \"text_max_pos\": " << p.text_max_pos << ",\n";
    f << "  \"unet_in_channels\": " << p.unet_in_channels << ",\n";
    f << "  \"unet_out_channels\": " << p.unet_out_channels << ",\n";
    f << "  \"unet_sample_size\": " << p.unet_sample_size << ",\n";
    f << "  \"vae_in_channels\": " << p.vae_in_channels << ",\n";
    f << "  \"vae_latent_channels\": " << p.vae_latent_channels << ",\n";
    f << "  \"vae_scaling_factor\": " << p.vae_scaling_factor << ",\n";
    f << "  \"sched_prediction_type\": \"" << p.sched_prediction_type << "\",\n";
    f << "  \"bos_id\": " << p.bos_id << ",\n";
    f << "  \"eos_id\": " << p.eos_id << ",\n";
    f << "  \"unk_id\": " << p.unk_id << ",\n";
    f << "  \"pad_id\": " << p.pad_id << "\n";
    f << "}\n";
}

SDXLProfile load_or_build_profile(const std::string & model_dir) {
    SDXLProfile p; p.dir = model_dir;
    const std::string cache = (fs::path(model_dir) / "sdxl.profile.json").string();
    if (fs::exists(cache) && read_json(cache, p)) return p;

    p.text_path = (fs::path(model_dir) / "text_encoder.gguf").string();
    p.unet_path = (fs::path(model_dir) / "unet.gguf").string();
    p.vae_path  = (fs::path(model_dir) / "vae.gguf").string();

    TextEncoder te = TextEncoder::from_file(p.text_path);
    UNet       un = UNet::from_file(p.unet_path);
    VAE         v = VAE::from_file(p.vae_path);
    const Model & um = un.model();

    p.text_hidden_size = te.config().hidden_size;
    p.text_num_layers  = te.config().num_layers;
    p.text_num_heads   = te.config().num_heads;
    p.text_max_pos     = te.config().max_position_embeddings;
    p.unet_in_channels  = un.config().in_channels;
    p.unet_out_channels = un.config().out_channels;
    p.unet_sample_size  = un.config().sample_size;
    p.vae_in_channels   = v.config().in_channels;
    p.vae_latent_channels = v.config().latent_channels;
    p.vae_scaling_factor  = v.config().scaling_factor;
    auto it = um.kv.find("diffusion.scheduler.prediction_type");
    if (it != um.kv.end()) p.sched_prediction_type = it->second;

    // token ids
    auto get_int = [&](const Model & m, const char * k, int def){ auto it = m.kv.find(k); if (it==m.kv.end()) return def; try { return std::stoi(it->second);} catch (...) {return def;}};
    p.bos_id = get_int(te.model(), "tokenizer.ggml.bos_token_id", -1);
    p.eos_id = get_int(te.model(), "tokenizer.ggml.eos_token_id", -1);
    p.unk_id = get_int(te.model(), "tokenizer.ggml.unknown_token_id", -1);
    p.pad_id = get_int(te.model(), "tokenizer.ggml.padding_token_id", -1);

    write_json(cache, p);
    return p;
}

static int get_json_int(const std::string & s, const char * key, int def) {
    auto k = std::string("\"") + key + "\"";
    auto i = s.find(k);
    if (i == std::string::npos) return def;
    i = s.find(":", i);
    if (i == std::string::npos) return def;
    std::stringstream ss(s.substr(i+1));
    int v; ss >> v; return ss.fail() ? def : v;
}

static float get_json_float(const std::string & s, const char * key, float def) {
    auto k = std::string("\"") + key + "\"";
    auto i = s.find(k);
    if (i == std::string::npos) return def;
    i = s.find(":", i);
    if (i == std::string::npos) return def;
    std::stringstream ss(s.substr(i+1));
    float v; ss >> v; return ss.fail() ? def : v;
}

static std::string get_json_str(const std::string & s, const char * key, const std::string & def) {
    auto k = std::string("\"") + key + "\"";
    auto i = s.find(k);
    if (i == std::string::npos) return def;
    i = s.find('"', i + k.size());
    if (i == std::string::npos) return def;
    auto j = s.find('"', i + 1);
    if (j == std::string::npos) return def;
    return s.substr(i + 1, j - i - 1);
}

SDXLProfile load_profile_json(const std::string & json_path) {
    SDXLProfile p;
    std::ifstream f(json_path);
    if (!f.is_open()) return p;
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    p.dir = get_json_str(s, "model_dir", "");
    p.text_path = get_json_str(s, "text_path", "");
    p.unet_path = get_json_str(s, "unet_path", "");
    p.vae_path  = get_json_str(s, "vae_path", "");
    p.text_hidden_size = get_json_int(s, "text_hidden_size", 0);
    p.text_num_layers  = get_json_int(s, "text_num_layers", 0);
    p.text_num_heads   = get_json_int(s, "text_num_heads", 0);
    p.text_max_pos     = get_json_int(s, "text_max_pos", 0);
    p.unet_in_channels  = get_json_int(s, "unet_in_channels", 0);
    p.unet_out_channels = get_json_int(s, "unet_out_channels", 0);
    p.unet_sample_size  = get_json_int(s, "unet_sample_size", 0);
    p.vae_in_channels   = get_json_int(s, "vae_in_channels", 0);
    p.vae_latent_channels = get_json_int(s, "vae_latent_channels", 4);
    p.vae_scaling_factor = get_json_float(s, "vae_scaling_factor", 0.18215f);
    p.sched_prediction_type = get_json_str(s, "sched_prediction_type", "epsilon");
    p.bos_id = get_json_int(s, "text_bos_id", -1);
    p.eos_id = get_json_int(s, "text_eos_id", -1);
    p.unk_id = get_json_int(s, "text_unk_id", -1);
    p.pad_id = get_json_int(s, "text_pad_id", -1);
    return p;
}

size_t estimate_memory_requirements(const SDXLProfile & p) {
    size_t base = 512ull * 1024ull * 1024ull;
    size_t text = (size_t)p.text_hidden_size * (size_t)std::max(1, p.text_num_layers) * (size_t)std::max(1, p.text_num_heads) * 8ull;
    size_t unet = (size_t)std::max(1, p.unet_in_channels) * (size_t)std::max(1, p.unet_out_channels) * 
    (size_t)std::max(1, p.unet_sample_size) * (size_t)std::max(1, p.unet_sample_size) / 4ull;
    size_t vae = (size_t)std::max(1, p.vae_in_channels) * (size_t)std::max(1, p.vae_latent_channels) * 4ull * 4ull;
    size_t total = base + text + unet + vae;
    size_t align = 16ull * 1024ull * 1024ull;
    return ((total + align - 1) / align) * align;
}

}
