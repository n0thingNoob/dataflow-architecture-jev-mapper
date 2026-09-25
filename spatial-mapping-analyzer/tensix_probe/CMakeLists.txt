cmake_minimum_required(VERSION 3.22...4.2)
project(spatial_tensix_probe)

add_executable(spatial_tensix_probe placement_probe.cpp)

find_package(TT-Metalium REQUIRED)
target_link_libraries(spatial_tensix_probe PUBLIC TT::Metalium)
if(TARGET TT::CommonPCH)
    tt_reuse_precompile_headers(spatial_tensix_probe TT::CommonPCH)
endif()
