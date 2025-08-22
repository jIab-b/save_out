#include "sd_text.h"

#include <stdexcept>

namespace sd {

int32_t TextEncoder::get_int_kv(const Model & m, const std::string & key, int32_t def) {
    auto it = m.kv.find(key);
    if (it == m.kv.end()) return def;
    try {
        return std::stoi(it->second);
    } catch (...) {
        return def;
    }
}

TextEncoder TextEncoder::from_file(const std::string & path) {
    TextEncoder te;
    te.m = load(path);
    te.cfg.hidden_size = get_int_kv(te.m, "diffusion.text.hidden_size", 0);
    te.cfg.num_layers = get_int_kv(te.m, "diffusion.text.num_hidden_layers", 0);
    te.cfg.num_heads = get_int_kv(te.m, "diffusion.text.num_attention_heads", 0);
    te.cfg.max_position_embeddings = get_int_kv(te.m, "diffusion.text.max_position_embeddings", 0);
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

ggml_tensor * TextEncoder::build_embeddings(ggml_context * ctx, ggml_tensor * token_ids) const {
    ggml_tensor * tok = find_token_embedding();
    if (!tok) throw std::runtime_error("token embedding not found in text gguf");
    ggml_tensor * gathered = ggml_get_rows(ctx, tok, token_ids);
    return gathered;
}

ggml_tensor * TextEncoder::forward(ggml_context * ctx, ggml_tensor * token_ids) const {
    ggml_tensor * tok_emb = build_embeddings(ctx, token_ids);
    ggml_tensor * pos_w = find_position_embedding();
    if (pos_w) {
        ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_ids->ne[0]);
        int32_t * p = (int32_t *) positions->data;
        for (int i = 0; i < token_ids->ne[0]; ++i) p[i] = i;
        ggml_tensor * pos_emb = ggml_get_rows(ctx, pos_w, positions);
        tok_emb = ggml_add(ctx, tok_emb, pos_emb);
    }
    // TODO: implement transformer blocks; for now return embeddings
    return tok_emb;
}

}


