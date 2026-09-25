cmake_minimum_required(VERSION 3.22...4.2)
project(spatial_tensix_probe)

add_executable(spatial_tensix_probe placement_probe.cpp)
add_executable(spatial_tensix_chain_probe two_core_chain.cpp)

set(SPATIAL_TENSIX_TARGETS
    spatial_tensix_probe
    spatial_tensix_chain_probe
)

if(TT_METAL_SOURCE_DIR)
    set(WITH_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
    set(ENABLE_DISTRIBUTED OFF CACHE BOOL "" FORCE)
    set(TT_METAL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(TTNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_PROGRAMMING_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(TT_USE_SYSTEM_SFPI ON CACHE BOOL "" FORCE)
    set(CMAKE_COMPILE_WARNING_AS_ERROR OFF CACHE BOOL "" FORCE)
    add_subdirectory("${TT_METAL_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/tt-metal" EXCLUDE_FROM_ALL)
    # The pinned source's HAL archive can race its object libraries when built
    # from this external project. Finish both before assembling the archive.
    add_dependencies(hal_1xx wh_hal bh_hal)
    foreach(target IN LISTS SPATIAL_TENSIX_TARGETS)
        target_link_libraries(${target} PUBLIC tt_metal)
    endforeach()
else()
    find_package(TT-Metalium REQUIRED)
    foreach(target IN LISTS SPATIAL_TENSIX_TARGETS)
        target_link_libraries(${target} PUBLIC TT::Metalium)
        if(TARGET TT::CommonPCH)
            tt_reuse_precompile_headers(${target} TT::CommonPCH)
        endif()
    endforeach()
endif()
