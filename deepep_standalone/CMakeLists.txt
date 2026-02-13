cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(deepep-standalone LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELEASE)
endif()

add_compile_options(-Wno-unused-parameter -Wno-unused-function -Wunused-value -Wcast-align)
add_compile_options(-Wcast-qual -Winvalid-pch -Wwrite-strings -Wsign-compare -Wextra)

if (${CMAKE_BUILD_TYPE} MATCHES "RELEASE")
    add_compile_options(-O3)
    add_compile_options(-fvisibility=hidden -fvisibility-inlines-hidden)
    add_compile_options(-fstack-protector-strong)
    add_compile_options(-Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now -fPIE -fPIC -ftrapv -s)
    message(STATUS "Build type set to RELEASE")
else ()
    add_compile_options(-g -rdynamic)
endif ()

if(DEFINED ENV{DEBUG_MODE})
    if("$ENV{DEBUG_MODE}" STREQUAL "ON")
        add_compile_definitions(DEBUG_MODE)
        message(STATUS "Debug logging enabled from environment")
    endif()
endif()

# Include config files
set(PROJECT_OP_SRC_BASE ${PROJECT_SOURCE_DIR}/csrc)
set(PROJECT_BUILD_PATH ${PROJECT_SOURCE_DIR}/build)
set(PROJECT_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/output)

include(${PROJECT_SOURCE_DIR}/cmake/config_git_last_commit.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/config_envs.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/config_ascend.cmake)

# Build deepep module
add_subdirectory(csrc/deepep)
