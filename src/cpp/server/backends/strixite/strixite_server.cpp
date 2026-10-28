#include "lemon/backends/strixite/strixite_server.h"

#include "lemon/backends/backend_utils.h"
#include "lemon/backends/strixite/strixite.h"
#include "lemon/utils/custom_args.h"
#include "lemon/utils/json_utils.h"
#include "lemon/utils/path_utils.h"

#include <algorithm>
#include <filesystem>
#include <set>
#include <stdexcept>

namespace fs = std::filesystem;

namespace lemon::backends {
namespace {

const std::string weights_file = "converted/Qwen3.8-Flash-Next.U-gdn_in-g128/weights.strixw";
const std::string ngram_file = "converted/Qwen3.8-Flash-Next.ngram-q8/ngram.table";
const std::string tokenizer_dir = "Qwen3.8-Flash-Next/";
const std::vector<std::string> model_files = {
    weights_file,
    ngram_file,
    "converted/Qwen3.8-Flash-Next.U-gdn_in-g128/vision.strixw",
    tokenizer_dir + "tokenizer.json",
    tokenizer_dir + "generation_config.json",
    tokenizer_dir + "config.json",
    tokenizer_dir + "preprocessor_config.json",
};

std::string binary_path() {
    auto path = BackendUtils::find_external_backend_binary("strixite", "system");
    if (path.empty()) {
        path = utils::find_executable_in_path(strixite::descriptor.binary);
    }
    if (path.empty()) {
        throw std::runtime_error(
            "Strixite requires a locally built strix_server. Add it to PATH or set "
            "LEMONADE_STRIXITE_SYSTEM_BIN to its absolute path.");
    }
    return path;
}

class StrixiteOps : public BackendOps {
public:
    void populate_metadata(ModelInfo& info, const BackendOpsContext&) const override {
        info.max_context_window = 524288;
    }

    void resolve_runtime_options(const ModelInfo&, RecipeOptions& options) const override {
        const int ctx_size = options.get_option("ctx_size");
        if (ctx_size <= 0) options.set_option("ctx_size", 524288);
    }

    std::optional<std::vector<std::string>> select_checkpoint_files(
        const std::string& variant, const std::vector<std::string>& repo_files) const override {
        if (!variant.empty()) {
            throw std::invalid_argument("Strixite expects a complete model repository without a variant");
        }
        for (const auto& file : model_files) {
            if (std::find(repo_files.begin(), repo_files.end(), file) == repo_files.end()) {
                throw std::runtime_error("Strixite model repository is missing " + file);
            }
        }
        return model_files;
    }

    std::string validate_checkpoint_file(const std::string& path) const override {
        for (const auto& file : model_files) {
            if (!fs::is_regular_file(utils::path_from_utf8(path) / file)) {
                return "Strixite model directory is missing " + file;
            }
        }
        return "";
    }

    InstallCheck check_install(const std::string&, bool) const override {
        try {
            binary_path();
            return {true, ""};
        } catch (const std::exception& e) {
            return {false, e.what()};
        }
    }

    std::optional<UnavailableState> classify_unavailable(
        const std::string&, const std::string& error, const std::string&) const override {
        return UnavailableState{"action_required", error,
                                "https://github.com/shawnshekari/strixite#quick-start", false};
    }
};

}  // namespace

StrixiteServer::StrixiteServer(const std::string& log_level, ModelManager* model_manager,
                             BackendManager* backend_manager)
    : WrappedServer("strix_server", log_level, model_manager, backend_manager) {}

void StrixiteServer::load(const std::string& model_name, const ModelInfo& model_info,
                         const RecipeOptions& options, bool do_not_upgrade) {
    (void)do_not_upgrade;
#ifndef __linux__
    throw std::runtime_error("Strixite is supported only on Linux with AMD Strix Halo (gfx1151)");
#endif
    const std::string custom_args = options.get_option("strixite_args");
    const std::set<std::string> reserved = {
        "--capacity", "--generation-config", "--help", "--host", "--model-id", "--ngram",
        "--port", "--print-config", "--tokenizer", "--weights",
    };
    const auto error = utils::validate_custom_args(custom_args, reserved);
    if (!error.empty()) {
        throw std::invalid_argument("Invalid Strixite arguments: " + error);
    }
    const auto model_path = model_info.resolved_path();
    if (model_path.empty()) {
        throw std::runtime_error("Strixite model directory is not resolved");
    }
    const auto missing = strixite::ops()->validate_checkpoint_file(model_path);
    if (!missing.empty()) {
        throw std::runtime_error(missing);
    }
    const auto root = utils::path_from_utf8(model_path);
    const auto path = [&root](const std::string& file) {
        return utils::path_to_utf8(root / file);
    };
    ServerCommand command;
    command.program = binary_path();
    command.args = {
        "--weights", path(weights_file), "--ngram", path(ngram_file),
        "--tokenizer", path(tokenizer_dir + "tokenizer.json"),
        "--generation-config", path(tokenizer_dir + "generation_config.json"),
        "--model-id", model_name, "--host", "127.0.0.1",
    };
    const int ctx_size = options.get_option("ctx_size");
    if (ctx_size > 0) {
        command.args.insert(command.args.end(), {"--capacity", std::to_string(ctx_size)});
    }
    auto args = utils::parse_custom_args(custom_args);
    command.args.insert(command.args.end(), args.begin(), args.end());
    port_ = choose_port();
    command.args.insert(command.args.end(), {"--port", std::to_string(port_)});
    command.env = {{"STRIX_MODELS_DIR", utils::get_cache_dir() + "/strixite"}};
    command.port = port_;
    command.stop_timeout_seconds = 90;
    device_type_ = DEVICE_GPU;
    start_server(std::make_unique<NativeProcess>(ProcessOutput{true, false}), command);
}

json StrixiteServer::chat_completion(const json& request) {
    return forward_request("/v1/chat/completions", utils::JsonUtils::with_legacy_max_tokens_alias(request));
}

json StrixiteServer::completion(const json& request) {
    return forward_request("/v1/completions", utils::JsonUtils::with_legacy_max_tokens_alias(request));
}

void StrixiteServer::forward_streaming_request(const std::string& endpoint,
                                              const std::string& request_body,
                                              httplib::DataSink& sink, bool sse,
                                              long timeout_seconds,
                                              TelemetryCallback telemetry_callback) {
    auto request = json::parse(request_body);
    if (endpoint == "/v1/chat/completions" || endpoint == "/v1/completions") {
        utils::JsonUtils::add_legacy_max_tokens_alias(request);
    }
    WrappedServer::forward_streaming_request(endpoint, request.dump(), sink, sse,
                                             timeout_seconds, telemetry_callback);
}

namespace strixite {
std::unique_ptr<WrappedServer> create(const BackendContext& ctx) {
    return make_server<StrixiteServer>(ctx);
}

const BackendSpec* spec() {
    static const BackendSpec value(descriptor.recipe, descriptor.binary);
    return &value;
}

const BackendOps* ops() { return single_ops<StrixiteOps>(); }
}

}  // namespace lemon::backends
