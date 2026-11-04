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

if(TARGET gufo_deepseek_v41_flash_hip)
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
endif()
