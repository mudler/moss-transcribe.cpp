# Test helpers for moss-transcribe.cpp. Individual tests are registered from
# Task 4 onwards via mt_add_test(); none are registered yet.
macro(mt_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE moss-transcribe)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src)
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endmacro()

mt_add_test(test_audio_io)

mt_add_test(test_mel)
set_tests_properties(test_mel PROPERTIES LABELS "model")

mt_add_test(test_whisper_encoder)
set_tests_properties(test_whisper_encoder PROPERTIES LABELS "model")

mt_add_test(test_audio_adaptor)
set_tests_properties(test_audio_adaptor PROPERTIES LABELS "model")

mt_add_test(test_audio_path_e2e)
set_tests_properties(test_audio_path_e2e PROPERTIES LABELS "model")

mt_add_test(test_tokenizer)
set_tests_properties(test_tokenizer PROPERTIES LABELS "model")

mt_add_test(test_audio_span)
set_tests_properties(test_audio_span PROPERTIES LABELS "model")

mt_add_test(test_qwen3_decoder)
set_tests_properties(test_qwen3_decoder PROPERTIES LABELS "model")

mt_add_test(test_fusion)
set_tests_properties(test_fusion PROPERTIES LABELS "model")

mt_add_test(test_generate_e2e)
set_tests_properties(test_generate_e2e PROPERTIES LABELS "model")
