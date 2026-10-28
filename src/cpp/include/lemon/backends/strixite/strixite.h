#pragma once

#include "lemon/backends/backend_descriptor.h"

namespace lemon::backends::strixite {

inline const BackendDescriptor descriptor = {
    /*recipe*/ "strixite",
    /*display_name*/ "Strixite",
    /*binary*/ "strix_server",
    /*config_section*/ "",
    /*default_device*/ DEVICE_GPU,
    /*slot_policy*/ SlotPolicy::Standard,
    /*selectable_backend*/ false,
    /*uses_ctx_size*/ true,
    /*dynamic_models*/ false,
    /*options*/ {
        {"strixite_args", "--strixite-args", "", "ARGS",
         "Custom arguments to pass to strix_server", "Strixite Options"},
    },
    /*support*/ {
        {"system", BackendTier::Experimental, BackendFormat::Native, {"linux"},
         {{"amd_gpu", {"gfx1151"}}}, "AMD Strix Halo (gfx1151), 128 GB RAM"},
    },
    /*supported_modes*/ {"chat"},
    /*required_checkpoints*/ {"main"},
    /*default_capabilities*/ {"tool-calling", "reasoning", "vision"},
    /*web_display_name*/ "",
    /*rocm_channels*/ {},
    /*exposes_prometheus_metrics*/ true,
    /*rocm_requires_cwsr_fix*/ false,
    /*version_policy*/ VersionPolicy::Exact,
    /*self_manages_downloads*/ false,
    /*takes_args*/ true,
    /*arg_variants*/ {},
    /*bin_variants*/ {"system"},
    /*config_extra*/ nlohmann::json::object(),
    /*streams_model_from_storage*/ true,
};

}  // namespace lemon::backends::strixite
