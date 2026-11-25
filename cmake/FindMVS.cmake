# FindMVS.cmake - 自动查找海康威视工业相机 SDK (MVS)，兼容 x86_64 与 aarch64
# 导出变量与目标：
#   MVS_FOUND
#   MVS_INCLUDE_DIRS
#   MVS_LIBRARIES
#   MVS::MVS (IMPORTED 目标)

set(_mvs_root_hints
    "$ENV{MVS_ROOT}"
    "/opt/MVS"
    "/usr/local/MVS"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/MVS"
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    set(_mvs_lib_subdir "lib/aarch64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(_mvs_lib_subdir "lib/64")
else()
    set(_mvs_lib_subdir "lib")
endif()

find_path(MVS_INCLUDE_DIR
    NAMES MvCameraControl.h
    HINTS ${_mvs_root_hints}
    PATH_SUFFIXES include
)

find_library(MVS_LIBRARY
    NAMES MvCameraControl
    HINTS ${_mvs_root_hints}
    PATH_SUFFIXES ${_mvs_lib_subdir} lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MVS
    REQUIRED_VARS MVS_LIBRARY MVS_INCLUDE_DIR
)

if(MVS_FOUND)
    set(MVS_INCLUDE_DIRS "${MVS_INCLUDE_DIR}")
    set(MVS_LIBRARIES "${MVS_LIBRARY}")
    get_filename_component(_mvs_lib_dir "${MVS_LIBRARY}" DIRECTORY)

    if(NOT TARGET MVS::MVS)
        add_library(MVS::MVS UNKNOWN IMPORTED)
        set_target_properties(MVS::MVS PROPERTIES
            IMPORTED_LOCATION "${MVS_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${MVS_INCLUDE_DIR}"
            INTERFACE_LINK_OPTIONS "-Wl,-rpath,${_mvs_lib_dir}"
        )
    endif()
endif()

mark_as_advanced(MVS_INCLUDE_DIR MVS_LIBRARY)
