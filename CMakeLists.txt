cmake_minimum_required(VERSION 3.18)
project(moss_transcribe CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(MT_BUILD_TESTS "Build tests" OFF)
option(MT_BUILD_CLI "Build CLI" ON)
option(MT_SHARED "Build shared lib" OFF)
foreach(be CUDA METAL VULKAN HIP)
  option(MT_GGML_${be} "Enable ggml ${be}" OFF)
  if(MT_GGML_${be})
    set(GGML_${be} ON CACHE BOOL "" FORCE)
  endif()
endforeach()
set(GGML_NATIVE ON CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE ON CACHE BOOL "" FORCE)

add_subdirectory(third_party/ggml)

set(MT_SRC
  src/common.cpp
  src/backend.cpp
  src/model_loader.cpp
  src/audio_io.cpp
  src/fft.cpp
  src/mel.cpp
  src/whisper_encoder.cpp
  src/audio_adaptor.cpp
  src/tokenizer.cpp
  src/audio_span.cpp
  src/qwen3.cpp
  src/qwen3_decoder.cpp
  src/generate.cpp
  src/audio_encoder.cpp
  src/transcribe.cpp
)
if(MT_SHARED)
  add_library(moss-transcribe SHARED ${MT_SRC})
  set_target_properties(moss-transcribe PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
  add_library(moss-transcribe STATIC ${MT_SRC})
endif()
target_include_directories(moss-transcribe PUBLIC include PRIVATE src third_party)
target_link_libraries(moss-transcribe PUBLIC ggml)

if(MT_BUILD_CLI)
  add_executable(moss-transcribe-cli src/cli.cpp)
  set_target_properties(moss-transcribe-cli PROPERTIES OUTPUT_NAME moss-transcribe)
  target_link_libraries(moss-transcribe-cli PRIVATE moss-transcribe)
endif()

if(MT_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
