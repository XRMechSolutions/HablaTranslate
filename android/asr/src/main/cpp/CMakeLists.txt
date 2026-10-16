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

# The ExternalProjects build patched copies, never the submodule itself (whisper_patch.cmake, D042).
include(${CMAKE_CURRENT_SOURCE_DIR}/whisper_patch.cmake)

# The copies and their inner builds are keyed by the submodule commit: a bump starts from a clean copy
# (no files left over from the old revision) and a fresh inner cache. Re-configure when HEAD moves.
execute_process(COMMAND git -C ${WHISPER_DIR} rev-parse --short=8 HEAD
    OUTPUT_VARIABLE WHISPER_REV OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE rev_rc ERROR_QUIET)
execute_process(COMMAND git -C ${WHISPER_DIR} rev-parse --path-format=absolute --git-path HEAD
    OUTPUT_VARIABLE WHISPER_HEAD_FILE OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if (NOT rev_rc EQUAL 0 OR WHISPER_REV STREQUAL "")
    set(WHISPER_REV nogit)
endif()
if (EXISTS "${WHISPER_HEAD_FILE}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${WHISPER_HEAD_FILE})
endif()

# Experimental GPU variant (D043): habla_asr_vk = the v86 CPU build plus ggml's Vulkan backend, with
# use_gpu on. Off unless Gradle passes -Phabla.vulkan=true. Needs the LunarG Vulkan SDK (glslc,
# headers, SPIRV-Headers) and, because ggml builds its shader generator for the host, a host C++
# compiler: on Windows run Gradle from an MSVC x64 environment (vcvars64.bat, `gradlew --no-daemon`).
option(HABLA_VULKAN "Build the experimental Vulkan variant habla_asr_vk (arm64 only)" OFF)
set(HABLA_VULKAN_SDK "$ENV{VULKAN_SDK}" CACHE PATH "LunarG Vulkan SDK root for HABLA_VULKAN")

# Builds whisper.cpp for one CPU variant and defines the shared JNI library habla_asr_<NAME> linked
# against the static archives in its build tree.
#   NAME        variant suffix (v86 | base)
#   ARM_ARCH    value for GGML_CPU_ARM_ARCH, empty on x86_64
#   EXTRA_ARGS  further cache arguments for the whisper.cpp configure (a CMake list)
#   ARGV3       optional: VULKAN to add ggml's Vulkan backend and build the JNI glue with use_gpu
function(habla_whisper_variant NAME ARM_ARCH EXTRA_ARGS)
    set(vulkan OFF)
    if (ARGC GREATER 3 AND "${ARGV3}" STREQUAL "VULKAN")
        set(vulkan ON)
    endif()
    # Linked straight from the build tree, with no install step: BUILD_ALWAYS re-runs the build, but
    # ExternalProject keeps the install step's stamp, so installed archives went stale after a patch
    # change (2026-10-16: the engine linked a 15:41 libwhisper.a after a 16:13 rebuild).
    set(bindir ${CMAKE_BINARY_DIR}/whisper-${WHISPER_REV}-${NAME}-build)
    set(lib_whisper ${bindir}/src/libwhisper.a)
    set(lib_ggml ${bindir}/ggml/src/libggml.a)
    set(lib_ggml_cpu ${bindir}/ggml/src/libggml-cpu.a)
    set(lib_ggml_base ${bindir}/ggml/src/libggml-base.a)

    # One copy per variant: whisper.cpp's configure writes into its source dir, and the variants
    # configure in parallel.
    set(WHISPER_SRC ${CMAKE_BINARY_DIR}/whisper.cpp-${WHISPER_REV}-patched-${NAME})
    habla_patched_whisper_source(${WHISPER_DIR} ${WHISPER_SRC})

    set(arch_args "")
    if (NOT "${ARM_ARCH}" STREQUAL "")
        list(APPEND arch_args -DGGML_CPU_ARM_ARCH=${ARM_ARCH})
    endif()

    set(vulkan_args "")
    set(vulkan_libs "")
    set(vulkan_byproducts "") # only files the ExternalProject builds; the NDK libvulkan stub is an input
    if (vulkan)
        set(sdk ${HABLA_VULKAN_SDK})
        # LunarG layout: Bin/Include/Lib on Windows, bin/include/lib (VULKAN_SDK = .../x86_64) on Linux.
        if (CMAKE_HOST_WIN32)
            set(glslc ${sdk}/Bin/glslc.exe)
            set(sdk_include ${sdk}/Include)
            set(sdk_lib ${sdk}/Lib)
        else()
            set(glslc ${sdk}/bin/glslc)
            set(sdk_include ${sdk}/include)
            set(sdk_lib ${sdk}/lib)
        endif()
        set(spirv_dir "")
        foreach(candidate ${sdk_lib}/cmake/SPIRV-Headers ${sdk}/share/cmake/SPIRV-Headers)
            if (spirv_dir STREQUAL "" AND EXISTS ${candidate}/SPIRV-HeadersConfig.cmake)
                set(spirv_dir ${candidate})
            endif()
        endforeach()
        if (NOT EXISTS ${glslc} OR NOT EXISTS ${sdk_include}/vulkan/vulkan.hpp OR spirv_dir STREQUAL "")
            message(FATAL_ERROR "HABLA_VULKAN: no Vulkan SDK at '${sdk}' (set VULKAN_SDK or -Phabla.vulkanSdk)")
        endif()
        if (CMAKE_HOST_WIN32 AND NOT DEFINED ENV{VCINSTALLDIR})
            message(FATAL_ERROR "HABLA_VULKAN on Windows: run Gradle from an MSVC x64 environment "
                "(vcvars64.bat, then gradlew --no-daemon); ggml builds its shader generator with the host compiler")
        endif()
        set(libvulkan ${CMAKE_SYSROOT}/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/${ANDROID_PLATFORM_LEVEL}/libvulkan.so)
        if (NOT EXISTS ${libvulkan})
            message(FATAL_ERROR "HABLA_VULKAN: NDK stub ${libvulkan} not found")
        endif()
        # Only vkGetInstanceProcAddr comes from the loader (vulkan.hpp dynamic dispatch), so the SDK's
        # newer headers work with any Android Vulkan driver.
        list(APPEND vulkan_args
            -DGGML_VULKAN=ON
            -DVulkan_INCLUDE_DIR=${sdk_include}
            -DVulkan_LIBRARY=${libvulkan}
            -DVulkan_GLSLC_EXECUTABLE=${glslc}
            -DSPIRV-Headers_DIR=${spirv_dir}
            -DCMAKE_CXX_FLAGS=-isystem${sdk_include})
        set(vulkan_libs ${bindir}/ggml/src/ggml-vulkan/libggml-vulkan.a ${libvulkan})
        set(vulkan_byproducts ${bindir}/ggml/src/ggml-vulkan/libggml-vulkan.a)
    endif()

    # whisper.cpp is always Release: a Debug ggml is 10-50x slower and would make debug builds and
    # instrumented tests unrepresentative. Only the JNI glue below follows the Gradle build type.
    ExternalProject_Add(whisper_${NAME}
        SOURCE_DIR ${WHISPER_SRC}
        # The inner build is incremental; without this a patched or bumped source is never rebuilt.
        BUILD_ALWAYS ON
        # Named for the patched source: a cache configured from another source path cannot be reused.
        BINARY_DIR ${bindir}
        INSTALL_COMMAND ""
        CMAKE_GENERATOR ${CMAKE_GENERATOR}
        CMAKE_ARGS
            -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
            -DANDROID_ABI=${ANDROID_ABI}
            -DANDROID_PLATFORM=${ANDROID_PLATFORM}
            -DANDROID_STL=c++_static
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
            ${arch_args}
            ${EXTRA_ARGS}
            ${vulkan_args}
        BUILD_BYPRODUCTS
            ${lib_whisper}
            ${lib_ggml}
            ${lib_ggml_cpu}
            ${lib_ggml_base}
            ${vulkan_byproducts}
    )

    set(target habla_asr_${NAME})
    add_library(${target} SHARED habla_jni.cpp habla_engine.cpp)
    add_dependencies(${target} whisper_${NAME})
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_include_directories(${target} PRIVATE ${WHISPER_DIR}/include ${WHISPER_DIR}/ggml/include)
    target_compile_options(${target} PRIVATE
        -fvisibility=hidden -fvisibility-inlines-hidden -ffunction-sections -fdata-sections)
    if (vulkan)
        target_compile_definitions(${target} PRIVATE HABLA_USE_GPU=1)
    endif()
    # libggml.a (the backend registry) references the backends, so they follow it.
    target_link_libraries(${target} PRIVATE
        ${lib_whisper}
        ${lib_ggml}
        ${lib_ggml_cpu}
        ${vulkan_libs}
        ${lib_ggml_base}
        log)
    target_link_options(${target} PRIVATE
        -Wl,--gc-sections -Wl,--exclude-libs,ALL -Wl,-z,max-page-size=16384)
endfunction()

if (ANDROID_ABI STREQUAL "arm64-v8a")
    # android_armv8.6_1 feature set: int8 matmul paths of q8_0/q5_x on Cortex-A720/A520.
    habla_whisper_variant(v86 "armv8.6-a+dotprod+i8mm+fp16" "")
    if (HABLA_VULKAN)
        habla_whisper_variant(vk "armv8.6-a+dotprod+i8mm+fp16" "" VULKAN)
    endif()
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
