cmake_minimum_required(VERSION 2.8)

project (HW-VTK)

find_package (ITK REQUIRED)
include (${ITK_USE_FILE})

find_package (VTK REQUIRED)
include (${VTK_USE_FILE})

add_executable (HW-VTK HW-VTK.cxx)

target_link_libraries (HW-VTK ${ITK_LIBRARIES} ${VTK_LIBRARIES})
