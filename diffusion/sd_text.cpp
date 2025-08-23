#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <stdexcept>
#include <cstring>
#include <string>
#include <unordered_map>
#include <cmath>

namespace sd {

struct Model {
    ggml_context * ctx = nullptr;
    gguf_context * uf = nullptr;
    std::unordered_map<std::string, ggml_tensor *> tensors;
    std::unordered_map<std::string, std::string> kv;
};

Model load(const std::string & path);

struct TextEncoderConfig {
    int32_t hidden_size = 0;
    int32_t num_layers = 0;
    int32_t num_heads = 0;
    int32_t max_position_embeddings = 0;
    float   layer_norm_eps = 1e-5f;
};

class TextEncoder {
public:
    static TextEncoder from_file(const std::string & path);

    ggml_tensor * build_embeddings(ggml_context * ctx, ggml_tensor * token_ids) const;
    ggml_tensor * forward(ggml_context * ctx, ggml_tensor * token_ids) const;

    const TextEncoderConfig & config() const;
    const Model & model() const;

    ggml_tensor * find_text_projection() const; // optional (present in text2)

private:
    Model m;
    TextEncoderConfig cfg;

    ggml_tensor * find_token_embedding() const;
    ggml_tensor * find_position_embedding() const;
    ggml_tensor * find_final_ln_weight() const;
    ggml_tensor * find_final_ln_bias() const;
    static int32_t get_int_kv(const Model & m, const std::string & key, int32_t def);
    static float   get_float_kv(const Model & m, const std::string & key, float def);
    static std::string get_str_kv(const Model & m, const std::string & key, const std::string & def);
    ggml_tensor * get(const std::string & name) const {
        auto it = m.tensors.find(name);
        if (it == m.tensors.end()) throw std::runtime_error("missing tensor: " + name);
        return it->second;
    }
};

const TextEncoderConfig & TextEncoder::config() const { return cfg; }
const Model & TextEncoder::model() const { return m; }

int32_t TextEncoder::get_int_kv(const Model & m, const std::string & key, int32_t def) {
    auto it = m.kv.find(key);
    if (it == m.kv.end()) return def;
    try {
        return std::stoi(it->second);
    } catch (...) {
        return def;
    }
}

// (pooled helper removed per revert request)

float TextEncoder::get_float_kv(const Model & m, const std::string & key, float def) {
    auto it = m.kv.find(key);
    if (it == m.kv.end()) return def;
    try { return std::stof(it->second); } catch (...) { return def; }
}

std::string TextEncoder::get_str_kv(const Model & m, const std::string & key, const std::string & def) {
    auto it = m.kv.find(key);
    if (it == m.kv.end()) return def;
    return it->second;
}

TextEncoder TextEncoder::from_file(const std::string & path) {
    TextEncoder te;
    te.m = load(path);
    te.cfg.hidden_size = get_int_kv(te.m, "diffusion.text.hidden_size", 0);
    te.cfg.num_layers = get_int_kv(te.m, "diffusion.text.num_hidden_layers", 0);
    te.cfg.num_heads = get_int_kv(te.m, "diffusion.text.num_attention_heads", 0);
    te.cfg.max_position_embeddings = get_int_kv(te.m, "diffusion.text.max_position_embeddings", 0);
    te.cfg.layer_norm_eps = get_float_kv(te.m, "diffusion.text.layer_norm_eps", 1e-5f);
    return te;
}

ggml_tensor * TextEncoder::find_token_embedding() const {
    auto it = m.tensors.find("text_model.embeddings.token_embedding.weight");
    if (it != m.tensors.end()) return it->second;
    it = m.tensors.find("model.embeddings.token_embedding.weight");
    if (it != m.tensors.end()) return it->second;
    return nullptr;
}

ggml_tensor * TextEncoder::find_position_embedding() const {
    auto it = m.tensors.find("text_model.embeddings.position_embedding.weight");
    if (it != m.tensors.end()) return it->second;
    it = m.tensors.find("model.embeddings.position_embedding.weight");
    if (it != m.tensors.end()) return it->second;
    return nullptr;
}

ggml_tensor * TextEncoder::find_final_ln_weight() const {
    auto it = m.tensors.find("text_model.final_layer_norm.weight");
    return it != m.tensors.end() ? it->second : nullptr;
}

ggml_tensor * TextEncoder::find_final_ln_bias() const {
    auto it = m.tensors.find("text_model.final_layer_norm.bias");
    return it != m.tensors.end() ? it->second : nullptr;
}

ggml_tensor * TextEncoder::find_text_projection() const {
    auto it = m.tensors.find("text_projection.weight");
    return it != m.tensors.end() ? it->second : nullptr;
}

ggml_tensor * TextEncoder::build_embeddings(ggml_context * ctx, ggml_tensor * token_ids) const {
    ggml_tensor * tok_src = find_token_embedding();
    if (!tok_src) throw std::runtime_error("token embedding not found in text gguf");
    // Directly gather from weights; do not copy weights or ids (ids are built by caller)
    ggml_tensor * gathered = ggml_get_rows(ctx, tok_src, token_ids);
    return gathered;
}

ggml_tensor * TextEncoder::forward(ggml_context * ctx, ggml_tensor * token_ids) const {
    ggml_tensor * cur = build_embeddings(ctx, token_ids);
    ggml_tensor * pos_w_src = find_position_embedding();
    if (pos_w_src) {
        ggml_tensor * positions_f = ggml_arange(ctx, 0.0f, (float) token_ids->ne[0], 1.0f);
        ggml_tensor * positions   = ggml_cast(ctx, positions_f, GGML_TYPE_I32);
        ggml_tensor * pos_emb = ggml_get_rows(ctx, pos_w_src, positions);
        cur = ggml_add(ctx, cur, pos_emb);
    }
    // Transformer blocks
    const int H = std::max(1, cfg.num_heads);
    const int C = std::max(1, cfg.hidden_size);
    const int Dh = C / H;
    const float scale = 1.0f / std::sqrt((float)Dh);

    auto apply_ln = [&](ggml_tensor * x, int il, int which) -> ggml_tensor * {
        // which: 1 or 2
        char bufw[128], bufb[128];
        std::snprintf(bufw, sizeof(bufw), "text_model.encoder.layers.%d.layer_norm%d.weight", il, which);
        std::snprintf(bufb, sizeof(bufb), "text_model.encoder.layers.%d.layer_norm%d.bias", il, which);
        auto * w = get(bufw);
        auto * b = get(bufb);
        ggml_tensor * n = ggml_norm(ctx, x, cfg.layer_norm_eps);
        ggml_tensor * s = ggml_mul(ctx, n, w);
        return ggml_add(ctx, s, b);
    };

    auto linear = [&](const char * name_w, const char * name_b, ggml_tensor * x) -> ggml_tensor * {
        ggml_tensor * w = get(name_w);
        ggml_tensor * b = get(name_b);
        ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        return ggml_add(ctx, y, b);
    };

    for (int il = 0; il < cfg.num_layers; ++il) {
        // Self-attention
        ggml_tensor * ln1 = apply_ln(cur, il, 1);
        char q_w[128], q_b[128], k_w[128], k_b[128], v_w[128], v_b[128], o_w[128], o_b[128];
        std::snprintf(q_w, sizeof(q_w), "text_model.encoder.layers.%d.self_attn.q_proj.weight", il);
        std::snprintf(q_b, sizeof(q_b), "text_model.encoder.layers.%d.self_attn.q_proj.bias", il);
        std::snprintf(k_w, sizeof(k_w), "text_model.encoder.layers.%d.self_attn.k_proj.weight", il);
        std::snprintf(k_b, sizeof(k_b), "text_model.encoder.layers.%d.self_attn.k_proj.bias", il);
        std::snprintf(v_w, sizeof(v_w), "text_model.encoder.layers.%d.self_attn.v_proj.weight", il);
        std::snprintf(v_b, sizeof(v_b), "text_model.encoder.layers.%d.self_attn.v_proj.bias", il);
        std::snprintf(o_w, sizeof(o_w), "text_model.encoder.layers.%d.self_attn.out_proj.weight", il);
        std::snprintf(o_b, sizeof(o_b), "text_model.encoder.layers.%d.self_attn.out_proj.bias", il);
        ggml_tensor * q = linear(q_w, q_b, ln1);
        ggml_tensor * k = linear(k_w, k_b, ln1);
        ggml_tensor * v = linear(v_w, v_b, ln1);

        // reshape to [Dh, H, T]
        int64_t T = q->ne[1];
        ggml_tensor * q3 = ggml_reshape_3d(ctx, q, Dh, H, T);
        ggml_tensor * k3 = ggml_reshape_3d(ctx, k, Dh, H, T);
        ggml_tensor * v3 = ggml_reshape_3d(ctx, v, Dh, H, T);

        ggml_tensor * attn_cat = nullptr;
        for (int h = 0; h < H; ++h) {
            // views per head: [Dh, T]
            size_t off = (size_t)h * k3->nb[1];
            ggml_tensor * qh = ggml_view_2d(ctx, q3, Dh, T, q3->nb[2], off);
            ggml_tensor * kh = ggml_view_2d(ctx, k3, Dh, T, k3->nb[2], off);
            ggml_tensor * vh = ggml_view_2d(ctx, v3, Dh, T, v3->nb[2], off);
            ggml_tensor * kht = ggml_transpose(ctx, kh); // [T, Dh]
            ggml_tensor * scores = ggml_mul_mat(ctx, kht, qh); // [T, T]
            scores = ggml_scale(ctx, scores, scale);
            ggml_tensor * probs = ggml_soft_max(ctx, scores); // softmax over keys per query
            ggml_tensor * oh = ggml_mul_mat(ctx, vh, probs); // [Dh, T]
            if (!attn_cat) attn_cat = oh;
            else attn_cat = ggml_concat(ctx, attn_cat, oh, /*dim=*/0);
        }
        // output projection + residual
        ggml_tensor * out = linear(o_w, o_b, attn_cat);
        cur = ggml_add(ctx, cur, out);

        // MLP
        ggml_tensor * ln2 = apply_ln(cur, il, 2);
        char fc1_w[128], fc1_b[128], fc2_w[128], fc2_b[128];
        std::snprintf(fc1_w, sizeof(fc1_w), "text_model.encoder.layers.%d.mlp.fc1.weight", il);
        std::snprintf(fc1_b, sizeof(fc1_b), "text_model.encoder.layers.%d.mlp.fc1.bias", il);
        std::snprintf(fc2_w, sizeof(fc2_w), "text_model.encoder.layers.%d.mlp.fc2.weight", il);
        std::snprintf(fc2_b, sizeof(fc2_b), "text_model.encoder.layers.%d.mlp.fc2.bias", il);
        ggml_tensor * h1 = linear(fc1_w, fc1_b, ln2);
        // activation
        std::string act = get_str_kv(m, "diffusion.text.hidden_act", "quick_gelu");
        ggml_tensor * hact = (act == "gelu") ? ggml_gelu(ctx, h1) : ggml_gelu_quick(ctx, h1);
        ggml_tensor * h2 = linear(fc2_w, fc2_b, hact);
        cur = ggml_add(ctx, cur, h2);
    }

    // final layer norm
    ggml_tensor * ln_w = find_final_ln_weight();
    ggml_tensor * ln_b = find_final_ln_bias();
    if (ln_w && ln_b) {
        ggml_tensor * normed = ggml_norm(ctx, cur, cfg.layer_norm_eps);
        ggml_tensor * scaled = ggml_mul(ctx, normed, ln_w);
        cur = ggml_add(ctx, scaled, ln_b);
    }
    return cur;
}

}
