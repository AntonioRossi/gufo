gufo_extension_target(
  NAME gufo_hy4_flash
  SOURCES
    extension.cpp
    probe.cpp
    pack.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
)

gufo_extension_target(
  NAME gufo_hy4_flash_hip
  SOURCES
    provider.cpp
  LINK_LIBRARIES
    gufo_hy4_flash
    gufo_ext_runtime
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
  set_tests_properties(hy4_flash.probe PROPERTIES
    LABELS "cpu;hy4_flash"
    ENVIRONMENT "HY4_FLASH_MODEL=")
endif()
