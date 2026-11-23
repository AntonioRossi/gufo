gufo_extension_target(
  NAME gufo_hy4_flash
  SOURCES
    extension.cpp
    probe.cpp
    pack.cpp
    tokenizer.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    ICU::uc
    ICU::i18n
)

if(BUILD_TESTING)
  target_sources(gufo_hy4_flash PRIVATE
    "${GUFO_EXTENSION_BUILD_ROOT}/logit_check.cpp")
endif()

gufo_extension_target(
  NAME gufo_hy4_flash_hip
  SOURCES
    provider.cpp
    image_prep.cpp
  LINK_LIBRARIES
    gufo_hy4_flash
    gufo_ext_runtime
    gufo_ext_hip
)

gufo_extension_target(
  NAME hy4_flash_tests
  SOURCES
    tests/probe_test.cpp
    tests/gguf_fixture.cpp
  LINK_LIBRARIES
    gufo_hy4_flash
    gufo_ext_runtime
    gufo_ext_core
)

if(TARGET hy4_flash_tests)
  add_test(NAME hy4_flash.probe COMMAND $<TARGET_FILE:hy4_flash_tests>)
  add_test(NAME hy4_flash.logit COMMAND $<TARGET_FILE:hy4_flash_tests>
           --logit-check)
  set_tests_properties(hy4_flash.probe PROPERTIES
    LABELS "cpu;hy4_flash"
    ENVIRONMENT "HY4_FLASH_MODEL=")
  set_tests_properties(hy4_flash.logit PROPERTIES LABELS "cpu;hy4_flash")
  # The CPU compatibility build uses the upstream core. Select the image
  # checks from the availability of the required image API.
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_INCLUDES
    "${PROJECT_SOURCE_DIR}/include;${PROJECT_SOURCE_DIR}")
  check_cxx_source_compiles("
    #include <gufo/ext/image.hpp>
    using HybridResize = decltype(gufo::core::ResizeRgb8);
    int main() {
      gufo::core::ImageDecodeOptions options{};
      (void)options;
      return 0;
    }" GUFO_HY4_IMAGE_API)
  unset(CMAKE_REQUIRED_INCLUDES)
  if(GUFO_HY4_IMAGE_API)
    target_sources(hy4_flash_tests PRIVATE
      "${GUFO_EXTENSION_BUILD_ROOT}/image_prep.cpp")
    target_compile_definitions(hy4_flash_tests PRIVATE HY4_IMAGE_CHECKS=1)
  endif()
endif()

if(TARGET gufo_hy4_flash_hip)
  set(HY4_MAYA "$ENV{HY4_MAYA}" CACHE PATH "Path to the selected project-maya source")
  set(STRATA_GGML_DIR "$ENV{STRATA_GGML_DIR}" CACHE PATH "Path to the selected llama.cpp source")
  if(NOT EXISTS "${HY4_MAYA}/src/hy4/hy4_model.cu")
    message(FATAL_ERROR "hy4_flash: set HY4_MAYA to the project-maya source directory")
  endif()
  if(NOT EXISTS "${STRATA_GGML_DIR}/ggml/CMakeLists.txt")
    message(FATAL_ERROR "hy4_flash: set STRATA_GGML_DIR to the llama.cpp source directory")
  endif()

  set(_hy4_ggml_vars
    BUILD_SHARED_LIBS GGML_STATIC GGML_CUDA GGML_OPENMP GGML_BACKEND_DL
    GGML_BUILD_TESTS GGML_BUILD_EXAMPLES GGML_NATIVE)
  foreach(_hy4_var IN LISTS _hy4_ggml_vars)
    get_property(_hy4_defined CACHE ${_hy4_var} PROPERTY TYPE SET)
    if(_hy4_defined)
      get_property(_hy4_saved_${_hy4_var} CACHE ${_hy4_var} PROPERTY VALUE)
      get_property(_hy4_saved_type_${_hy4_var} CACHE ${_hy4_var} PROPERTY TYPE)
    endif()
  endforeach()
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(GGML_STATIC ON CACHE BOOL "" FORCE)
  set(GGML_CUDA OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
  set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  if(NOT DEFINED HY4_GGML_COMMIT AND STRATA_GGML_DIR STREQUAL "$ENV{STRATA_GGML_DIR}")
    set(HY4_GGML_COMMIT "$ENV{HY4_GGML_COMMIT}")
  endif()
  if(HY4_GGML_COMMIT AND NOT EXISTS "${STRATA_GGML_DIR}/.git")
    # Nix supplies the revision for its immutable source tree.
    set(GIT_EXE OFF)
    set(GGML_BUILD_COMMIT "${HY4_GGML_COMMIT}")
  endif()
  add_subdirectory("${STRATA_GGML_DIR}/ggml"
                   "${CMAKE_BINARY_DIR}/hy4-ggml" EXCLUDE_FROM_ALL)
  foreach(_hy4_var IN LISTS _hy4_ggml_vars)
    if(DEFINED _hy4_saved_${_hy4_var})
      set(${_hy4_var} "${_hy4_saved_${_hy4_var}}" CACHE
          ${_hy4_saved_type_${_hy4_var}} "" FORCE)
    else()
      unset(${_hy4_var} CACHE)
    endif()
  endforeach()

  add_library(hy4_hip_config INTERFACE)
  target_include_directories(hy4_hip_config INTERFACE
    "${HY4_MAYA}/include"
    "${HY4_MAYA}/include/strata/hip_compat"
    "${HY4_MAYA}/src/hy4"
    "${HY4_MAYA}/src/mm3"
    "${HY4_MAYA}/third_party/ggml"
    "${ROCWMMA_INCLUDE_DIR}"
  )
  target_compile_definitions(hy4_hip_config INTERFACE
    STRATA_USE_HIP=1 STRATA_HAS_WMMA=1 MAYA_DS4_IQ_ONLY DS4_IQ_MIN_FORMS)
  target_compile_options(hy4_hip_config INTERFACE
    "$<$<COMPILE_LANGUAGE:CXX>:-include>"
    "$<$<COMPILE_LANGUAGE:CXX>:${HY4_MAYA}/include/strata/hip_compat/cuda_runtime.h>"
    "$<$<COMPILE_LANGUAGE:HIP>:-include>"
    "$<$<COMPILE_LANGUAGE:HIP>:${HY4_MAYA}/include/strata/hip_compat/cuda_runtime.h>"
  )

  set(_hy4_cpu_sources
    "${HY4_MAYA}/src/kernels/cpu/pool.cpp"
    "${HY4_MAYA}/src/kernels/cpu/expert_layout.cpp"
    "${HY4_MAYA}/src/kernels/cpu/exl3_ref.cpp"
    "${HY4_MAYA}/src/kernels/cpu/expert.cpp"
    "${HY4_MAYA}/src/kernels/cpu/q2_avx2.cpp"
    "${HY4_MAYA}/src/kernels/cpu/exl3_cpu.cpp"
    "${HY4_MAYA}/src/kernels/cpu/native_expert.cpp"
    "${HY4_MAYA}/src/kernels/cpu/iq_avx512.cpp"
    "${HY4_MAYA}/src/kernels/cpu/iq_avx2.cpp"
    "${HY4_MAYA}/src/kernels/cpu/kq_avx512.cpp"
    "${HY4_MAYA}/src/kernels/cpu/iq_avx512_one.cpp"
    "${HY4_MAYA}/src/kernels/cpu/iq2v_rows.cpp"
    "${HY4_MAYA}/src/kernels/cpu/iq3v_rows.cpp"
    "${HY4_MAYA}/src/kernels/cpu/glm_lane_row4.cpp"
    "${HY4_MAYA}/src/kernels/cpu/glm_lane_row4_avx2.cpp"
  )
  add_library(hy4_cpu STATIC ${_hy4_cpu_sources})
  target_include_directories(hy4_cpu PUBLIC "${HY4_MAYA}/include"
    PRIVATE "${HY4_MAYA}/third_party/ggml")
  target_link_libraries(hy4_cpu PUBLIC ggml-cpu ggml-base Threads::Threads)
  target_compile_definitions(hy4_cpu PUBLIC STRATA_NATIVE_EXPERTS=1)

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties("${HY4_MAYA}/src/kernels/cpu/expert.cpp"
      PROPERTIES COMPILE_OPTIONS
        "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
    set_source_files_properties("${HY4_MAYA}/src/kernels/cpu/q2_avx2.cpp"
      PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties("${HY4_MAYA}/src/kernels/cpu/iq_avx512.cpp"
      PROPERTIES COMPILE_OPTIONS
        "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mf16c;-mfma;-ffp-contract=off")
    set_source_files_properties("${HY4_MAYA}/src/kernels/cpu/iq_avx2.cpp"
      PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c;-ffp-contract=off")
    set_source_files_properties("${HY4_MAYA}/src/kernels/cpu/kq_avx512.cpp"
      PROPERTIES COMPILE_OPTIONS
        "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mf16c;-mfma")
    set_source_files_properties("${HY4_MAYA}/src/kernels/cpu/iq_avx512_one.cpp"
      PROPERTIES COMPILE_OPTIONS
        "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vbmi;-mbmi2;-mf16c;-mfma;-ffp-contract=off")
    set_source_files_properties("${HY4_MAYA}/src/kernels/cpu/iq2v_rows.cpp"
      PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
    set_source_files_properties("${HY4_MAYA}/src/kernels/cpu/iq3v_rows.cpp"
      PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
    set_source_files_properties("${HY4_MAYA}/src/kernels/cpu/glm_lane_row4.cpp"
      PROPERTIES COMPILE_OPTIONS "-ffp-contract=off"
                 COMPILE_DEFINITIONS "STRATA_GLM_LANE_ROW4_KERNELS=1")
    set_source_files_properties("${HY4_MAYA}/src/kernels/cpu/glm_lane_row4_avx2.cpp"
      PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c;-ffp-contract=off"
                 COMPILE_DEFINITIONS "STRATA_GLM_LANE_ROW4_KERNELS=1")
  endif()

  file(GLOB _hy4_lane_sources
    "${HY4_MAYA}/src/hy4/hy4_lane.cpp"
    "${HY4_MAYA}/src/hy4/hy4_lane.hpp"
    "${HY4_MAYA}/src/ds4/ds4_iq_cpu.*"
    "${HY4_MAYA}/src/ds4/ds4_iq512_kern.inc"
    "${HY4_MAYA}/src/kernels/cpu/iq*.cpp"
    "${HY4_MAYA}/src/kernels/cpu/iq*.inc"
    "${HY4_MAYA}/include/strata/kernels/cpu/iq*.hpp"
  )
  list(SORT _hy4_lane_sources)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
               ${_hy4_lane_sources})
  set(_hy4_lane_sum "")
  foreach(_hy4_file IN LISTS _hy4_lane_sources)
    file(MD5 "${_hy4_file}" _hy4_lane_hash)
    string(APPEND _hy4_lane_sum "${_hy4_lane_hash}")
  endforeach()
  string(MD5 _hy4_lane_sum "${_hy4_lane_sum}")
  string(SUBSTRING "${_hy4_lane_sum}" 0 12 _hy4_lane_sum)
  set_source_files_properties("${HY4_MAYA}/src/hy4/hy4_lane.cpp"
    PROPERTIES COMPILE_DEFINITIONS "HY4_LANE_KERNELS=k${_hy4_lane_sum}")

  target_sources(gufo_hy4_flash_hip PRIVATE
    "${HY4_MAYA}/src/hy4/hy4_lane.cpp"
    "${HY4_MAYA}/src/hy4/hy4_tp.cpp"
    "${HY4_MAYA}/src/ds4/ds4_iq_cpu.cpp"
    "${HY4_MAYA}/src/ds4/ds4_gguf_cpu.cpp"
  )
  gufo_extension_hip_sources(gufo_hy4_flash_hip
    SOURCES
      "${HY4_MAYA}/src/hy4/hy4_model.cu"
      "${HY4_MAYA}/src/hy4/hy4_vision.cu"
      "${HY4_MAYA}/src/hy4/hy4_kernels.cu"
      "${HY4_MAYA}/src/ds4/ds4_iq.cu"
      "${HY4_MAYA}/src/ds4/ds4_lt.cu")
  target_link_libraries(gufo_hy4_flash_hip PRIVATE hy4_cpu hy4_hip_config)
  install(FILES "${HY4_MAYA}/LICENSE"
    DESTINATION share/licenses/gufo/third-party RENAME LICENSE.project-maya)
  install(FILES "${HY4_MAYA}/third_party/ggml/LICENSE"
    DESTINATION share/licenses/gufo/third-party RENAME LICENSE.maya-ggml)
  install(FILES "${STRATA_GGML_DIR}/LICENSE"
    DESTINATION share/licenses/gufo/third-party RENAME LICENSE.hy4-llama-cpp)
  if(BUILD_TESTING)
    target_compile_definitions(gufo_hy4_flash_hip PRIVATE
      HY4_TEST_DIAGNOSTICS=1)
  endif()
  if(TARGET roc::hipblaslt)
    set_source_files_properties("${HY4_MAYA}/src/ds4/ds4_lt.cu"
      PROPERTIES COMPILE_DEFINITIONS STRATA_HIPBLASLT_AVAILABLE=1)
    target_link_libraries(gufo_hy4_flash_hip PRIVATE roc::hipblaslt)
  endif()
endif()

gufo_extension_target(
  NAME hy4_flash_kernels_test
  SOURCES
    "${HY4_MAYA}/tests/hy4_kernels_test.cu"
  LINK_LIBRARIES
    gufo_ext_hip
)

if(TARGET hy4_flash_kernels_test)
  set_source_files_properties("${HY4_MAYA}/tests/hy4_kernels_test.cu"
    PROPERTIES LANGUAGE HIP)
  gufo_extension_hip_sources(hy4_flash_kernels_test
    SOURCES "${HY4_MAYA}/src/hy4/hy4_kernels.cu")
  target_link_libraries(hy4_flash_kernels_test PRIVATE hy4_hip_config)
  add_test(NAME hy4_flash.kernels
           COMMAND $<TARGET_FILE:hy4_flash_kernels_test>)
  set_tests_properties(hy4_flash.kernels PROPERTIES
    LABELS "gpu;hy4_flash"
    SKIP_REGULAR_EXPRESSION "no GPU")
endif()
