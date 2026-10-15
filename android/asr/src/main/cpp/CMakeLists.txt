# Outer native build for :asr (design.md section 1).
#
# ggml's GGML_CPU_ARM_ARCH is a per-configure cache variable and AGP runs one configure per ABI, so
# each CPU variant of whisper.cpp is built as static libraries by its own ExternalProject. The JNI
# libraries are real targets of this project because AGP only packages .so files it sees through the
# CMake file API.
cmake_minimum_required(VERSION 3.22)
project(habla_asr CXX C)

include(ExternalProject)

set(WHISPER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/whisper.cpp)
get_filename_component(WHISPER_DIR ${WHISPER_DIR} ABSOLUTE)

if (NOT EXISTS ${WHISPER_DIR}/include/whisper.h)
    message(FATAL_ERROR
        "whisper.cpp submodule missing: run \"git submodule update --init android/third_party/whisper.cpp\"")
endif()

# Builds whisper.cpp for one CPU variant into ${CMAKE_BINARY_DIR}/whisper-<NAME> and defines the
# shared JNI library habla_asr_<NAME> linked against it.
#   NAME        variant suffix (v86 | base)
#   ARM_ARCH    value for GGML_CPU_ARM_ARCH, empty on x86_64
#   EXTRA_ARGS  further cache arguments for the whisper.cpp configure (a CMake list)
function(habla_whisper_variant NAME ARM_ARCH EXTRA_ARGS)
    set(prefix ${CMAKE_BINARY_DIR}/whisper-${NAME})
    set(libdir ${prefix}/lib)

    set(arch_args "")
    if (NOT "${ARM_ARCH}" STREQUAL "")
        list(APPEND arch_args -DGGML_CPU_ARM_ARCH=${ARM_ARCH})
    endif()

    # whisper.cpp is always Release: a Debug ggml is 10-50x slower and would make debug builds and
    # instrumented tests unrepresentative. Only the JNI glue below follows the Gradle build type.
    ExternalProject_Add(whisper_${NAME}
        SOURCE_DIR ${WHISPER_DIR}
        BINARY_DIR ${CMAKE_BINARY_DIR}/whisper-${NAME}-build
        INSTALL_DIR ${prefix}
        CMAKE_GENERATOR ${CMAKE_GENERATOR}
        CMAKE_ARGS
            -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
            -DANDROID_ABI=${ANDROID_ABI}
            -DANDROID_PLATFORM=${ANDROID_PLATFORM}
            -DANDROID_STL=c++_static
            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
            -DCMAKE_BUILD_TYPE=Release
            -DBUILD_SHARED_LIBS=OFF
            -DWHISPER_BUILD_EXAMPLES=OFF
            -DWHISPER_BUILD_TESTS=OFF
            -DWHISPER_BUILD_SERVER=OFF
            -DGGML_OPENMP=OFF
            -DGGML_LLAMAFILE=OFF
            -DGGML_NATIVE=OFF
            -DGGML_CPU_KLEIDIAI=OFF
            -DGGML_VULKAN=OFF
            -DGGML_OPENCL=OFF
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DCMAKE_INSTALL_LIBDIR=lib
            ${arch_args}
            ${EXTRA_ARGS}
        BUILD_BYPRODUCTS
            ${libdir}/libwhisper.a
            ${libdir}/libggml.a
            ${libdir}/libggml-cpu.a
            ${libdir}/libggml-base.a
    )

    set(target habla_asr_${NAME})
    add_library(${target} SHARED habla_jni.cpp habla_engine.cpp)
    add_dependencies(${target} whisper_${NAME})
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_include_directories(${target} PRIVATE ${WHISPER_DIR}/include ${WHISPER_DIR}/ggml/include)
    target_compile_options(${target} PRIVATE
        -fvisibility=hidden -fvisibility-inlines-hidden -ffunction-sections -fdata-sections)
    target_link_libraries(${target} PRIVATE
        ${libdir}/libwhisper.a
        ${libdir}/libggml.a
        ${libdir}/libggml-cpu.a
        ${libdir}/libggml-base.a
        log)
    target_link_options(${target} PRIVATE
        -Wl,--gc-sections -Wl,--exclude-libs,ALL -Wl,-z,max-page-size=16384)
endfunction()

if (ANDROID_ABI STREQUAL "arm64-v8a")
    # android_armv8.6_1 feature set: int8 matmul paths of q8_0/q5_x on Cortex-A720/A520.
    habla_whisper_variant(v86 "armv8.6-a+dotprod+i8mm+fp16" "")
    # Any minSdk-29 arm64 phone; no SIGILL risk.
    habla_whisper_variant(base "armv8-a" "")
elseif (ANDROID_ABI STREQUAL "x86_64")
    # Emulator only. SSE4.2 is guaranteed by the Android x86_64 ABI; AVX is not passed through by
    # every emulator host. Cross-compiling turns the AVX family on by default, so it is switched off.
    habla_whisper_variant(base ""
        "-DGGML_SSE42=ON;-DGGML_AVX=OFF;-DGGML_AVX2=OFF;-DGGML_BMI2=OFF;-DGGML_FMA=OFF;-DGGML_F16C=OFF;-DGGML_AVX512=OFF")
else()
    message(FATAL_ERROR "habla_asr supports arm64-v8a and x86_64 only, not ${ANDROID_ABI}")
endif()
