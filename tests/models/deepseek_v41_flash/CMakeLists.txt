add_executable(ds41_tool_calls_test tool_calls_test.cpp)
target_link_libraries(ds41_tool_calls_test PRIVATE gufo_deepseek_v41_flash)
target_compile_definitions(ds41_tool_calls_test PRIVATE
  GUFO_DS41_CHAT_GOLDENS="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/encoding-1017.json")
add_test(NAME ds41.tool_calls COMMAND ds41_tool_calls_test)
set_tests_properties(ds41.tool_calls PROPERTIES LABELS "cpu;ds41;tools")

add_executable(ds41_chat_template_test
  chat_template_test.cpp
  ${PROJECT_SOURCE_DIR}/src/models/deepseek_v41_flash/chat_template.cpp)
target_link_libraries(ds41_chat_template_test PRIVATE gufo_core)
target_compile_definitions(ds41_chat_template_test PRIVATE
  GUFO_DS41_CHAT_GOLDENS="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/encoding-1017.json")
add_test(NAME ds41.template COMMAND ds41_chat_template_test)
set_tests_properties(ds41.template PROPERTIES LABELS "cpu;ds41;tokenization")

add_executable(ds41_probe_status_test probe_status_test.cpp)
target_link_libraries(ds41_probe_status_test PRIVATE gufo_core)
add_test(NAME ds41.probe_status COMMAND ds41_probe_status_test)
set_tests_properties(ds41.probe_status PROPERTIES LABELS "cpu;ds41")

add_executable(ds41_continuation_layout_test
  continuation_layout_test.cpp
  ${PROJECT_SOURCE_DIR}/src/models/deepseek_v41_flash/continuation_layout.cpp)
target_link_libraries(ds41_continuation_layout_test PRIVATE gufo_core)
target_compile_definitions(ds41_continuation_layout_test PRIVATE
  DS41_VARIANT_CODER=1
  DS41_VARIANT_NAMESPACE=gufo::models::deepseek_v41_flash::coder)
add_test(NAME ds41.continuation_layout COMMAND ds41_continuation_layout_test)
set_tests_properties(ds41.continuation_layout PROPERTIES LABELS "cpu;ds41;cache")

if(TARGET gufo_deepseek_v41_flash_hip)
  add_executable(ds41_continuation_adapter_test continuation_adapter_test.cpp)
  target_link_libraries(ds41_continuation_adapter_test PRIVATE
    gufo_deepseek_v41_flash_hip gufo_cache_hip hip::host)
  target_compile_definitions(ds41_continuation_adapter_test PRIVATE
    DS41_VARIANT_CODER=1
    DS41_VARIANT_NAMESPACE=gufo::models::deepseek_v41_flash::coder)
  add_test(NAME ds41.continuation_adapter COMMAND ds41_continuation_adapter_test)
  set_tests_properties(ds41.continuation_adapter PROPERTIES
    LABELS "gpu;hip;ds41;cache;external-model;slow"
    RESOURCE_LOCK ds41_model SKIP_RETURN_CODE 77)

  foreach(ds41_test mmq_typed attention hc_state_pass candidate_mask)
    add_executable(ds41_${ds41_test}_test ${ds41_test}_test.cpp)
    target_link_libraries(ds41_${ds41_test}_test PRIVATE
      gufo_deepseek_v41_flash_hip hip::host)
    add_test(NAME ds41.${ds41_test} COMMAND ds41_${ds41_test}_test)
    set_tests_properties(ds41.${ds41_test} PROPERTIES
      LABELS "gpu;hip;gfx1151;ds41;quality" SKIP_RETURN_CODE 77)
  endforeach()
  target_link_libraries(ds41_mmq_typed_test PRIVATE gufo_deepseek_v41_flash_mmq)

  add_executable(ds41_load_probe EXCLUDE_FROM_ALL load_probe.cpp)
  target_link_libraries(ds41_load_probe PRIVATE gufo_deepseek_v41_flash_hip hip::host)

  add_executable(ds41_load_probe_coder EXCLUDE_FROM_ALL load_probe.cpp)
  target_link_libraries(ds41_load_probe_coder PRIVATE
    gufo_deepseek_v41_flash_hip hip::host)
  target_compile_definitions(ds41_load_probe_coder PRIVATE
    DS41_VARIANT_CODER=1
    DS41_VARIANT_NAMESPACE=gufo::models::deepseek_v41_flash::coder)

  add_executable(ds41_probe_check EXCLUDE_FROM_ALL probe_check.cpp)
  target_link_libraries(ds41_probe_check PRIVATE
    gufo_deepseek_v41_flash gufo_core)
endif()
