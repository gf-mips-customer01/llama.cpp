message(STATUS "Using NPX backend")

ggml_add_backend_library(ggml-npx
                         ggml-npx.cpp
                        )

# TODO: link the npx-toolkit library and include the npx-toolkit headers here.
#   target_link_libraries     (ggml-npx PRIVATE ${NPX_TOOLKIT_LIBRARIES})
#   target_include_directories(ggml-npx SYSTEM PRIVATE ${NPX_TOOLKIT_INCLUDE_DIRS})
