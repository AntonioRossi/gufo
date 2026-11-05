# DeepSeek V4.1 Flash model extension.

gufo_extension_target(
  NAME gufo_deepseek_v41_flash
  SOURCES
    chat_template.cpp
    tool_calls.cpp
    runtime/sampling.cpp
    probe.cpp
    extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_core
)

if(TARGET gufo_deepseek_v41_flash)
  target_include_directories(gufo_deepseek_v41_flash PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT})
  target_include_directories(gufo_deepseek_v41_flash SYSTEM PUBLIC
    ${GUFO_EXTENSION_BUILD_ROOT}/runtime)
  target_compile_definitions(gufo_deepseek_v41_flash PRIVATE
    _GNU_SOURCE
    DS41_ROCM_BUILD=1)
  if(ENGINE_ENABLE_HIP)
    target_compile_options(gufo_deepseek_v41_flash PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-O3;-ffast-math;-fno-finite-math-only>)
  endif()
endif()

if(ENGINE_ENABLE_HIP)
  set(DEEPSEEK_V41_FLASH_ROCM_DIR "${GUFO_EXTENSION_BUILD_ROOT}/kernels/rocm")
  set(DEEPSEEK_V41_FLASH_MMQ_DIR "${DEEPSEEK_V41_FLASH_ROCM_DIR}/mmq")
  set(DEEPSEEK_V41_FLASH_MMQ_SOURCES
    ${DEEPSEEK_V41_FLASH_MMQ_DIR}/ds41_ggml_stubs.hip.cpp
    ${DEEPSEEK_V41_FLASH_MMQ_DIR}/ds41_mmq.hip.cpp
    ${DEEPSEEK_V41_FLASH_MMQ_DIR}/ds41_mmq_d2r.hip.cpp
    ${DEEPSEEK_V41_FLASH_MMQ_DIR}/mmid.hip.cpp
    ${DEEPSEEK_V41_FLASH_MMQ_DIR}/mmvq.hip.cpp
    ${DEEPSEEK_V41_FLASH_MMQ_DIR}/quantize.hip.cpp)

  add_library(gufo_deepseek_v41_flash_mmq STATIC
    ${DEEPSEEK_V41_FLASH_MMQ_SOURCES})

  foreach(ds41_mmq_type Q5_K Q6_K IQ1_S IQ1_M IQ2_XS IQ2_S)
    set(ds41_mmq_type_source
      "${CMAKE_CURRENT_BINARY_DIR}/ds41_mmq_v41_${ds41_mmq_type}.hip.cpp")
    file(GENERATE OUTPUT ${ds41_mmq_type_source} CONTENT
      "#define DS41_MMQ_TYPE GGML_TYPE_${ds41_mmq_type}\n#include \"ds41_mmq_v41_types.hip.cpp\"\n")
    target_sources(gufo_deepseek_v41_flash_mmq PRIVATE ${ds41_mmq_type_source})
    list(APPEND DEEPSEEK_V41_FLASH_MMQ_SOURCES ${ds41_mmq_type_source})
  endforeach()

  set_source_files_properties(${DEEPSEEK_V41_FLASH_MMQ_SOURCES}
    PROPERTIES LANGUAGE HIP)
  set_target_properties(gufo_deepseek_v41_flash_mmq PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(gufo_deepseek_v41_flash_mmq PRIVATE
    GGML_USE_HIP
    GGML_HIP_NO_VMM
    DS41_ROCM_WMMA_MMQ_Y=64)
  target_compile_options(gufo_deepseek_v41_flash_mmq PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-value;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_deepseek_v41_flash_mmq SYSTEM PUBLIC
    ${DEEPSEEK_V41_FLASH_MMQ_DIR})
  target_link_libraries(gufo_deepseek_v41_flash_mmq PUBLIC
    hip::host
    roc::hipblas)
endif()

gufo_extension_target(
  NAME gufo_deepseek_v41_flash_hip
  SOURCES
    engine.cpp
    runtime/expert_store.cpp
    runtime/model_data.cpp
    runtime/tokenizer.cpp
    runtime/v41_decode.cpp
    runtime/v41_dspark.cpp
    runtime/v41_session.cpp
    runner.cpp
    drivers.cpp
    bench.cpp
    provider.cpp
  LINK_LIBRARIES
    gufo_deepseek_v41_flash
    gufo_deepseek_v41_flash_mmq
    gufo_core
    gufo_logging
    gufo_ext_cache
    hip::host
    roc::hipblas
    roc::hipblaslt
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

if(TARGET gufo_deepseek_v41_flash_hip)
  gufo_extension_hip_sources(gufo_deepseek_v41_flash_hip
    SOURCES
      kernels/rocm/backend.hip.cpp
      kernels/rocm/ds41_v41.hip.cpp
      kernels/rocm/ds41_gemv.hip.cpp
      kernels/rocm/ds41_expert_gemv.hip.cpp
      kernels/rocm/ds41_routed_f16.hip.cpp
      kernels/rocm/ds41_dense_f16.hip.cpp)
  # Preserve the source flags for the routed and dense F16 kernels only.
  set_source_files_properties(
    ${DEEPSEEK_V41_FLASH_ROCM_DIR}/ds41_routed_f16.hip.cpp
    ${DEEPSEEK_V41_FLASH_ROCM_DIR}/ds41_dense_f16.hip.cpp
    PROPERTIES COMPILE_OPTIONS "-Xclang;-target-feature;-Xclang;-real-true16")
  target_compile_definitions(gufo_deepseek_v41_flash_hip PRIVATE
    _GNU_SOURCE
    DS41_ROCM_BUILD=1)
  target_compile_options(gufo_deepseek_v41_flash_hip PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-ffast-math;-fno-finite-math-only>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_deepseek_v41_flash_hip PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT})
  target_include_directories(gufo_deepseek_v41_flash_hip SYSTEM PUBLIC
    ${GUFO_EXTENSION_BUILD_ROOT}/runtime
    ${DEEPSEEK_V41_FLASH_ROCM_DIR}
    ${ROCWMMA_INCLUDE_DIR})
endif()

install(FILES "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE.ds4"
  DESTINATION share/licenses/gufo/third-party)
