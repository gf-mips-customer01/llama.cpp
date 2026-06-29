message(STATUS "Using NPX backend")

ggml_add_backend_library(ggml-npx
                         ggml-npx.cpp
                        )

if (GGML_NPX_TOOLKIT_ROOT)
    if (NOT EXISTS "${GGML_NPX_TOOLKIT_ROOT}/CMakeLists.txt")
        message(FATAL_ERROR "GGML_NPX_TOOLKIT_ROOT does not point to an npx-toolkit source tree: ${GGML_NPX_TOOLKIT_ROOT}")
    endif()

    # Sets the default value to "none" to enable the compilation of the tools, without any runtime components in this moment.
    set(XPLAN_DRIVER none CACHE STRING "Driver backend to use for npx-toolkit" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "Disable npx-toolkit tests when built as a subproject" FORCE)

    add_subdirectory("${GGML_NPX_TOOLKIT_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/npx-toolkit" EXCLUDE_FROM_ALL)

    target_include_directories(ggml-npx PRIVATE "${GGML_NPX_TOOLKIT_ROOT}" "${GGML_NPX_TOOLKIT_ROOT}/toolkit" "${GGML_NPX_TOOLKIT_ROOT}/interface")
    # Add assembler library to the NPX backend library.
    target_link_libraries(ggml-npx PRIVATE npx-asm-core)
else()
    message(STATUS "GGML_NPX_TOOLKIT_ROOT is not set; building the NPX backend skeleton only")
endif()
