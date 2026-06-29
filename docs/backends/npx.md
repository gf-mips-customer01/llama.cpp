# NPX backend

The NPX backend is an experimental integration for the MIPS ARC NPX Neural Processing Unit family. NPX is the name used for this accelerator family in the context of the MIPS ARC NPX line of NPUs.

It is enabled with the CMake option `-DGGML_NPX=ON`.

## Build

```bash
cmake -B build -DGGML_NPX=ON -DGGML_NPX_TOOLKIT_ROOT=/path/to/npx-toolkit
cmake --build build --config Release
```

This backend is implemented as a GGML backend target in [ggml/src/ggml-npx/CMakeLists.txt](../../ggml/src/ggml-npx/CMakeLists.txt). When `GGML_NPX_TOOLKIT_ROOT` is set, the build will add the local npx-toolkit tree as a CMake subproject and link the NPX backend against the toolkit targets that are available.

## Prerequisites

- Make sure the NPX toolkit or an equivalent vendor SDK is available.
- If you are using the local checkout, initialize its submodules first:

  ```bash
  git -C /path/to/npx-toolkit submodule update --init --recursive
  ```

- Some npx-toolkit submodules that build the simulation path (for example the `hlmodel` flow) require `SYSTEMC_HOME` to be set before configuring CMake. If you see an error such as `SYSTEMC_HOME is not set`, provide the SystemC installation prefix in the environment before running CMake.
- Expose the relevant headers and libraries to CMake through the toolchain, `CMAKE_PREFIX_PATH`, or explicit include/link settings when you are not using the local source tree.
- Expect this integration to be experimental and to require some bring-up work on the target platform.

## Notes

- The integration is intended for custom accelerator experimentation and may require extra include or link settings.
- If you are using an NPX SDK or vendor library, make sure its headers and libraries are visible to CMake before configuring the build.
- For general build instructions, see [docs/build.md](../build.md).
