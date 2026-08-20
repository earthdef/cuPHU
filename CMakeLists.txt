cmake_minimum_required(VERSION 3.18)

project(cuPHU LANGUAGES C CXX CUDA)

# ── options ──────────────────────────────────────────────────────────────────
option(BUILD_PYTHON_BINDINGS "Build pybind11 Python extension" ON)
option(BUILD_TESTS           "Build C++ unit tests"             OFF)
option(CUPHU_PROFILE          "Print per-stage TICK/TOCK timers to stderr" OFF)

set(CMAKE_C_STANDARD    11)   # _Thread_local in patched CS2
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)

# ── CUDA ─────────────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# Default to all architectures; override with -DCMAKE_CUDA_ARCHITECTURES="80;86"
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "70;80;86;89;90" CACHE STRING "CUDA architectures")
endif()

# ── SNAPHU sources (everything except snaphu.c which defines main()) ──────────
set(SNAPHU_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ext/snaphu/src")
if(NOT IS_DIRECTORY "${SNAPHU_SRC_DIR}")
  message(
    FATAL_ERROR
    "SNAPHU source submodule not found at ${SNAPHU_SRC_DIR}.\n"
    "Run: git submodule update --init --recursive"
  )
endif()

# snaphu_cs2.c is replaced by a patched copy (src/cxx/snaphu_tls) whose
# file-scope solver state is _Thread_local, allowing concurrent per-tile
# MCF initialization without a global mutex.
set(SNAPHU_LIB_SOURCES
  ${SNAPHU_SRC_DIR}/snaphu_cost.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cxx/snaphu_tls/snaphu_cs2.c
  ${SNAPHU_SRC_DIR}/snaphu_io.c
  ${SNAPHU_SRC_DIR}/snaphu_solver.c
  ${SNAPHU_SRC_DIR}/snaphu_tile.c
  ${SNAPHU_SRC_DIR}/snaphu_util.c
)

add_library(snaphu_lib STATIC ${SNAPHU_LIB_SOURCES})
target_include_directories(snaphu_lib PUBLIC "${SNAPHU_SRC_DIR}")
target_compile_definitions(snaphu_lib PRIVATE _GNU_SOURCE)
set_target_properties(snaphu_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
find_library(C_MATH_LIBRARY m REQUIRED)
target_link_libraries(snaphu_lib PUBLIC ${C_MATH_LIBRARY})

# ── cuPHU CUDA library ─────────────────────────────────────────────────────
set(CUDA_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/cxx/cuphu")

set(CUPHU_CUDA_SOURCES
  ${CUDA_SRC_DIR}/cuphu_util.cu
  ${CUDA_SRC_DIR}/cuphu_cost.cu
  ${CUDA_SRC_DIR}/cuphu_phase.cu
  ${CUDA_SRC_DIR}/cuphu_conncomp.cu
  ${CUDA_SRC_DIR}/cuphu_bridge_gpu.cu
  ${CUDA_SRC_DIR}/cuphu_incrcost.cu
  ${CUDA_SRC_DIR}/cuphu_negcycle.cu
  ${CUDA_SRC_DIR}/cuphu_laplace.cu
)

set(CUPHU_CXX_SOURCES
  ${CUDA_SRC_DIR}/cuphu_solver.cpp
)

add_library(cuphu_core STATIC
  ${CUPHU_CUDA_SOURCES}
  ${CUPHU_CXX_SOURCES}
)

target_include_directories(cuphu_core PUBLIC
  "${CUDA_SRC_DIR}"
  "${SNAPHU_SRC_DIR}"
)

set_target_properties(cuphu_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE  ON
)

target_compile_options(cuphu_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
)

if(CUPHU_PROFILE)
  target_compile_definitions(cuphu_core PRIVATE CUPHU_PROFILE)
endif()

target_link_libraries(cuphu_core PUBLIC
  snaphu_lib
  CUDA::cudart
  CUDA::cublas
)

# ── Python bindings ───────────────────────────────────────────────────────────
if(BUILD_PYTHON_BINDINGS)
  # scikit-build-core sets SKBUILD; standalone CMake builds also work via FindPython
  if(DEFINED SKBUILD_PROJECT_NAME)
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 CONFIG REQUIRED)
    set(_py_dest "${SKBUILD_PROJECT_NAME}")
  else()
    find_package(pybind11 CONFIG REQUIRED)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

    # Where to install the Python package for standalone (non-scikit-build-core)
    # builds. If CMAKE_INSTALL_PREFIX is the active conda environment, install
    # straight into its real site-packages (Python3_SITEARCH, from
    # FindPython3).
    # To install cuphu with a custom isce3 installation, e.g.:
    #   cmake -G Ninja -B build -DCMAKE_INSTALL_PREFIX=/opt/isce3
    #   ninja -C build install
    get_filename_component(_install_prefix_real "${CMAKE_INSTALL_PREFIX}" REALPATH)
    set(_conda_prefix_real "")
    if(DEFINED ENV{CONDA_PREFIX})
      get_filename_component(_conda_prefix_real "$ENV{CONDA_PREFIX}" REALPATH)
    endif()

    if(_conda_prefix_real AND _install_prefix_real STREQUAL _conda_prefix_real)
      set(_py_dest "${Python3_SITEARCH}/cuphu")
    else()
      set(_py_dest "${CMAKE_INSTALL_PREFIX}/packages/cuphu")
    endif()

    # Standalone (non-scikit-build-core) builds: pip normally packages the
    # pure-Python sources itself, so a plain `cmake --install` needs its own
    # rule to place them alongside the compiled extension.
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/cuphu/"
      DESTINATION "${_py_dest}"
      FILES_MATCHING PATTERN "*.py"
    )
  endif()

  pybind11_add_module(_cuphu_ext MODULE
    ${CUDA_SRC_DIR}/cuphu_py.cu
  )

  set_target_properties(_cuphu_ext PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )

  target_link_libraries(_cuphu_ext PRIVATE cuphu_core)

  install(TARGETS _cuphu_ext LIBRARY DESTINATION "${_py_dest}")
endif()
