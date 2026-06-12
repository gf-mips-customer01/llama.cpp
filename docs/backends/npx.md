# NPX backend

The NPX backend is an experimental integration for the MIPS ARC NPX Neural Processing Unit family. NPX is the name used for this accelerator family in the context of the MIPS ARC NPX line of NPUs.

It is enabled with the CMake option `-DGGML_NPX=ON`.

## Build

```bash
cmake -B build -DGGML_NPX=ON
cmake --build build --config Release
```

This backend is implemented as a GGML backend target in [ggml/src/ggml-npx/CMakeLists.txt](../../ggml/src/ggml-npx/CMakeLists.txt). In the current tree, the NPX toolkit headers and libraries still need to be provided by the toolchain or build environment.

## Prerequisites

- Make sure the NPX toolkit or an equivalent vendor SDK is available.
- Expose the relevant headers and libraries to CMake through the toolchain, `CMAKE_PREFIX_PATH`, or explicit include/link settings.
- Expect this integration to be experimental and to require some bring-up work on the target platform.

## Notes

- The integration is intended for custom accelerator experimentation and may require extra include or link settings.
- If you are using an NPX SDK or vendor library, make sure its headers and libraries are visible to CMake before configuring the build.
- For general build instructions, see [docs/build.md](../build.md).
