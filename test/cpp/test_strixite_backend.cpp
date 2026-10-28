#include "lemon/backends/strixite/strixite.h"
#include "lemon/backends/strixite/strixite_server.h"
#include "lemon/config_file.h"
#include "lemon/runtime_config.h"
#include "lemon/system_info.h"
#include "lemon/utils/path_utils.h"
#include "lemon/utils/process_manager.h"

#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <pthread.h>

namespace fs = std::filesystem;
using lemon::json;
using lemon::backends::StrixiteServer;
namespace strixite = lemon::backends::strixite;

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

class ScopedEnv {
public:
    explicit ScopedEnv(const std::string& key)
        : key_(key), previous_(lemon::utils::get_environment_variable_utf8(key)) {}
    ~ScopedEnv() {
        if (previous_.empty()) lemon::utils::unset_environment_variable_utf8(key_);
        else lemon::utils::set_environment_variable_utf8(key_, previous_);
    }
    void set(const std::string& value) {
        lemon::utils::set_environment_variable_utf8(key_, value);
    }
private:
    std::string key_;
    std::string previous_;
};

class TestSystemInfo : public lemon::SystemInfo {
public:
    lemon::CPUInfo get_cpu_device() override { return {}; }
    lemon::GPUInfo get_amd_igpu_device() override { return {}; }
    std::vector<lemon::GPUInfo> get_amd_dgpu_devices() override { return {}; }
    std::vector<lemon::GPUInfo> get_nvidia_gpu_devices() override { return {}; }
    lemon::NPUInfo get_npu_device() override { return {}; }
};

class ScopedRuntimeConfig {
public:
    explicit ScopedRuntimeConfig(lemon::RuntimeConfig* config)
        : previous_(lemon::RuntimeConfig::global()) {
        lemon::RuntimeConfig::set_global(config);
    }
    ~ScopedRuntimeConfig() { lemon::RuntimeConfig::set_global(previous_); }
private:
    lemon::RuntimeConfig* previous_;
};

void check_hardware_availability() {
    lemon::RuntimeConfig config(lemon::ConfigFile::base_defaults());
    ScopedRuntimeConfig config_scope(&config);
    TestSystemInfo system_info;
    json devices = {{"cpu", {{"name", "Test CPU"}, {"family", "x86_64"}}}};
    auto recipes = system_info.build_recipes_info(devices);
    check(recipes["strixite"]["backends"]["system"]["state"] == "unsupported",
          "installed Strixite binary is unsupported without an AMD GPU");
    for (const auto& family : {"gfx1150", "gfx1100", "", "gfx1151"}) {
        devices["amd_gpu"] = json::array({{
            {"name", "Test GPU"}, {"family", family}, {"available", true},
        }});
        recipes = system_info.build_recipes_info(devices);
        const bool halo = std::string(family) == "gfx1151";
        check(recipes["strixite"]["backends"]["system"]["state"] ==
                  (halo ? "installed" : "unsupported"),
              "binary availability preserves GPU requirements for '" + std::string(family) + "'");
        check(recipes["strixite"].value("default_backend", "") == (halo ? "system" : ""),
              "only compatible hardware selects Strixite as a default backend");
    }
}

int serve(int argc, char** argv) {
    std::cout << "strixite stdout diagnostic" << std::endl;
    std::cerr << "strixite stderr diagnostic" << std::endl;
    std::map<std::string, std::string> args;
    for (int i = 1; i + 1 < argc; i += 2) args[argv[i]] = argv[i + 1];
    if (args["--host"] != "127.0.0.1" || args["--capacity"] != "8192" ||
        args["--model-id"] != "test-strixite" || args["--thinking"] != "off") {
        return 2;
    }
    for (const auto& flag : {"--weights", "--ngram", "--tokenizer", "--generation-config"}) {
        if (!fs::is_regular_file(args[flag])) return 3;
    }
    sigset_t termination_signals;
    sigemptyset(&termination_signals);
    sigaddset(&termination_signals, SIGTERM);
    if (pthread_sigmask(SIG_BLOCK, &termination_signals, nullptr) != 0) return 4;
    httplib::Server server;
    server.Get("/health", [](const auto&, auto& response) {
        response.set_content("{}", "application/json");
    });
    for (const auto& endpoint : {"/v1/chat/completions", "/v1/completions"}) {
        server.Post(endpoint, [args](const auto& request, auto& response) {
            auto body = json::parse(request.body);
            json result = {
                {"model", args.at("--model-id")}, {"request", body}, {"args", args},
                {"cache_root", lemon::utils::get_environment_variable_utf8("STRIX_MODELS_DIR")},
                {"choices", json::array({{{"index", 0}, {"message", {{"role", "assistant"},
                    {"content", "hello"}}}, {"finish_reason", "stop"}}})},
                {"usage", {{"prompt_tokens", 2}, {"completion_tokens", 1}, {"total_tokens", 3}}},
            };
            if (body.value("stream", false)) {
                response.set_content("data: " + result.dump() + "\n\ndata: [DONE]\n\n",
                                     "text/event-stream");
            } else {
                response.set_content(result.dump(), "application/json");
            }
        });
    }
    std::thread shutdown([&] {
        int signal = 0;
        if (sigwait(&termination_signals, &signal) == 0) {
            std::this_thread::sleep_for(std::chrono::seconds(6));
            std::ofstream(lemon::utils::get_environment_variable_utf8(
                "LEMONADE_STRIXITE_TEST_SHUTDOWN_MARKER")) << "flushed";
            server.stop();
        }
    });
    const bool listening = server.listen(args["--host"], std::stoi(args["--port"]));
    if (!listening) pthread_kill(shutdown.native_handle(), SIGTERM);
    shutdown.join();
    return listening ? 0 : 4;
}

void run(const fs::path& root, const fs::path& executable) {
    ScopedEnv binary("LEMONADE_STRIXITE_SYSTEM_BIN");
    ScopedEnv search_path("PATH");
    binary.set("");
    search_path.set(root.string());
    auto* ops = strixite::ops();
    check(lemon::SystemInfo::backend_supports_arch("strixite", "system", "gfx1151"),
          "Strix Halo is supported");
    check(!lemon::SystemInfo::backend_supports_arch("strixite", "system", "gfx1150"),
          "other AMD architectures are excluded");
    check(!ops->check_install("system", false).installed, "missing binary is unavailable");
    const auto remediation = ops->classify_unavailable("system", "missing", "");
    check(remediation && remediation->state == "action_required", "manual installation is explained");
    fs::create_symlink(executable, root / "strix_server");
    check(ops->check_install("system", false).installed, "binary is discovered through PATH");
    search_path.set("");
    binary.set(root.string());
    check(ops->check_install("system", false).installed, "binary directory override is resolved");
    binary.set(executable.string());
    check(ops->check_install("system", false).installed, "explicit binary works without PATH");
    lemon::utils::set_cache_dir((root / "cache").string());
    check_hardware_availability();

    std::vector<std::string> files = {
        "converted/Qwen3.8-Flash-Next.U-gdn_in-g128/weights.strixw",
        "converted/Qwen3.8-Flash-Next.U-gdn_in-g128/vision.strixw",
        "converted/Qwen3.8-Flash-Next.ngram-q8/ngram.table",
        "Qwen3.8-Flash-Next/tokenizer.json",
        "Qwen3.8-Flash-Next/generation_config.json",
        "Qwen3.8-Flash-Next/config.json",
        "Qwen3.8-Flash-Next/preprocessor_config.json",
    };
    const auto model_dir = root / "model with spaces";
    for (const auto& file : files) {
        fs::create_directories((model_dir / file).parent_path());
        std::ofstream(model_dir / file) << "{}";
    }
    files.push_back("unused-layout/weights.strixw");
    auto selected = ops->select_checkpoint_files("", files);
    check(selected && selected->size() == 7, "download includes companions and excludes other layouts");
    check(ops->validate_checkpoint_file(model_dir.string()).empty(), "complete import is accepted");
    fs::remove(model_dir / "Qwen3.8-Flash-Next/tokenizer.json");
    check(!ops->validate_checkpoint_file(model_dir.string()).empty(), "incomplete import is rejected");
    std::ofstream(model_dir / "Qwen3.8-Flash-Next/tokenizer.json") << "{}";
    try {
        ops->select_checkpoint_files("", {"weights.strixw"});
        check(false, "incomplete repository must fail before downloading");
    } catch (const std::runtime_error&) {}

    lemon::ModelInfo info;
    info.recipe = "strixite";
    info.resolved_paths["main"] = model_dir.string();
    lemon::RecipeOptions defaults("strixite", json::object());
    ops->resolve_runtime_options(info, defaults);
    check(defaults.get_option("ctx_size") == 524288, "automatic context keeps the upstream default");
    ops->populate_metadata(info, {});
    check(info.max_context_window == 524288, "model advertises its context window");
    StrixiteServer backend("error", nullptr, nullptr);
    for (const auto& flag : {"--port 1234", "--host=0.0.0.0", "--weights other", "--capacity 4096"}) {
        try {
            backend.load("test-strixite", info,
                         lemon::RecipeOptions("strixite", {{"strixite_args", flag}}));
            check(false, "managed flag must be rejected: " + std::string(flag));
        } catch (const std::invalid_argument&) {}
    }
    backend.load("test-strixite", info, lemon::RecipeOptions("strixite", {
        {"ctx_size", 8192}, {"strixite_args", "--thinking off"},
    }));
    auto result = backend.chat_completion({{"messages", json::array()}, {"max_completion_tokens", 12}});
    check(result["request"]["max_tokens"] == 12, "chat token alias is forwarded");
    check(result["model"] == "test-strixite", "served model name matches Lemonade");
    check(result["cache_root"] == (root / "cache/strixite").string(), "prompt cache is outside model artifacts");
    result = backend.completion({{"prompt", "hello"}, {"max_completion_tokens", 9}});
    check(result["request"]["max_tokens"] == 9, "completion token alias is forwarded");
    check(backend.responses({}).contains("error"), "unsupported Responses API reports an error");
    for (const auto& endpoint : {"/v1/chat/completions", "/v1/completions"}) {
        std::string output;
        httplib::DataSink sink;
        sink.write = [&output](const char* data, size_t size) {
            output.append(data, size);
            return true;
        };
        sink.is_writable = [] { return true; };
        sink.done = [] {};
        backend.forward_streaming_request(endpoint,
            json({{"stream", true}, {"max_completion_tokens", 7}}).dump(), sink);
        check(output.find("[DONE]") != std::string::npos, "SSE terminator is forwarded");
        check(output.find("\"max_tokens\":7") != std::string::npos, "streaming token alias is forwarded");
    }
    const auto address = backend.get_address();
    backend.unload();
    check(fs::is_regular_file(root / "shutdown-complete"),
          "unload allows a cache flush taking longer than five seconds");
    httplib::Client client(address.substr(0, address.size() - 3));
    check(!client.Get("/health"), "unload terminates the subprocess");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--weights") return serve(argc, argv);
    if (argc == 1) {
        std::string output;
        const int exit_code = lemon::utils::ProcessManager::run_process_with_output(
            fs::canonical(argv[0]).string(), {"--run-tests"},
            [&output](const std::string& line) {
                output += line + '\n';
                return true;
            }, "", 90);
        check(exit_code == 0, "backend regression subprocess exit code " +
                  std::to_string(exit_code) + ":\n" + output);
        check(output.find("strixite stdout diagnostic") != std::string::npos,
              "backend stdout remains visible without debug logging");
        check(output.find("strixite stderr diagnostic") != std::string::npos,
              "backend stderr remains visible without debug logging");
        return failures == 0 ? 0 : 1;
    }
    const auto root = fs::temp_directory_path() / ("lemon-strixite-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root);
    ScopedEnv shutdown_marker("LEMONADE_STRIXITE_TEST_SHUTDOWN_MARKER");
    shutdown_marker.set((root / "shutdown-complete").string());
    try {
        run(root, fs::canonical(argv[0]));
    } catch (const std::exception& e) {
        check(false, e.what());
    }
    fs::remove_all(root);
    return failures == 0 ? 0 : 1;
}
