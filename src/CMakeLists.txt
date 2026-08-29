cmake_minimum_required(VERSION 3.12)
project(pixelforge)

set(CMAKE_BUILD_TYPE Release)

# -------------- 빌드 버전 정보 (콘솔 배너/-h에 표시, 구버전 exe 혼동 방지용) --------------#
# git 커밋 해시를 구성 시점에 잡아둠. git이 없거나 .git이 없는 환경(예: 소스만 압축해서
# 받은 경우)에서도 빌드가 실패하지 않도록 실패 시 "unknown"으로 대체.
find_package(Git QUIET)
if (GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE PIXELFORGE_GIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE PIXELFORGE_GIT_RESULT
    )
endif ()
if (NOT PIXELFORGE_GIT_HASH OR PIXELFORGE_GIT_RESULT)
    set(PIXELFORGE_GIT_HASH "unknown")
endif ()
string(TIMESTAMP PIXELFORGE_BUILD_DATE "%Y-%m-%d %H:%M UTC" UTC)
message(STATUS "PixelForge build version: git=${PIXELFORGE_GIT_HASH} date=${PIXELFORGE_BUILD_DATE}")
add_compile_definitions(PIXELFORGE_GIT_HASH="${PIXELFORGE_GIT_HASH}")
add_compile_definitions(PIXELFORGE_BUILD_DATE="${PIXELFORGE_BUILD_DATE}")

# -------------- Support C++17 for using filesystem  ------------------#
# https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# set(CMAKE_CXX_CLANG_TIDY
#         clang-tidy;
#         -header-filter=.;
#         -checks=*;
#         -warnings-as-errors=*;)

# set_target_properties(${target} PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_PROGRAM}")

option(USE_SYSTEM_NCNN "build with system libncnn" OFF)
option(USE_SYSTEM_WEBP "build with system libwebp" OFF)



find_package(Threads)
find_package(OpenMP REQUIRED)
find_package(Vulkan REQUIRED)

if (Vulkan_FOUND)
    message(STATUS "Vulkan_INCLUDE_DIRS: ${Vulkan_INCLUDE_DIRS}")
    message(STATUS "Vulkan_LIBRARIES: ${Vulkan_LIBRARIES}")
    message(STATUS "Vulkan_LIBRARY: ${Vulkan_LIBRARY}")
    message(STATUS "Vulkan_FOUND: ${Vulkan_FOUND}")
else ()
    message(FATAL_ERROR "Vulkan Not Found!")
endif ()

find_program(GLSLANGVALIDATOR_EXECUTABLE NAMES glslangValidator PATHS $ENV{VULKAN_SDK}/bin NO_CMAKE_FIND_ROOT_PATH)
message(STATUS "Found glslangValidator: ${GLSLANGVALIDATOR_EXECUTABLE}")

# ---- RealESRGAN 배경 업스케일용 Vulkan compute shader 컴파일 ----
# realesrgan-ncnn-vulkan 공식 배포판과 동일하게, preproc/postproc 셰이더를
# fp32 / fp16-storage / fp16+int8-storage 세 가지 변형으로 미리 컴파일해서
# .spv.hex.h 헤더로 임베드합니다. (realesrgan.cpp가 include)
macro(compile_shader SHADER_SRC)
    set(SHADER_SRC_FULLPATH ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SRC})

    get_filename_component(SHADER_SRC_NAME_WE ${SHADER_SRC} NAME_WE)
    set(SHADER_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -V -s -x -o ${SHADER_SPV_HEX_FILE} ${SHADER_SRC_FULLPATH}
        DEPENDS ${SHADER_SRC_FULLPATH}
        COMMENT "Building SPIR-V module ${SHADER_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)
    list(APPEND SHADER_SPV_HEX_FILES ${SHADER_SPV_HEX_FILE})

    set(SHADER_fp16s_SRC_NAME_WE "${SHADER_SRC_NAME_WE}_fp16s")
    set(SHADER_fp16s_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_fp16s_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_fp16s_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -DNCNN_fp16_storage=1 -V -s -x -o ${SHADER_fp16s_SPV_HEX_FILE} ${SHADER_SRC_FULLPATH}
        DEPENDS ${SHADER_SRC_FULLPATH}
        COMMENT "Building SPIR-V module ${SHADER_fp16s_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_fp16s_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)
    list(APPEND SHADER_SPV_HEX_FILES ${SHADER_fp16s_SPV_HEX_FILE})

    set(SHADER_int8s_SRC_NAME_WE "${SHADER_SRC_NAME_WE}_int8s")
    set(SHADER_int8s_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_int8s_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_int8s_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -DNCNN_fp16_storage=1 -DNCNN_int8_storage=1 -V -s -x -o ${SHADER_int8s_SPV_HEX_FILE} ${SHADER_SRC_FULLPATH}
        DEPENDS ${SHADER_SRC_FULLPATH}
        COMMENT "Building SPIR-V module ${SHADER_int8s_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_int8s_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)
    list(APPEND SHADER_SPV_HEX_FILES ${SHADER_int8s_SPV_HEX_FILE})
endmacro()

include_directories(${CMAKE_CURRENT_BINARY_DIR})

find_package(OpenCV REQUIRED)

if (OpenCV_FOUND)
    message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
    message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "OpenCV Not Found!")
endif ()

# -------------- Auto-detect opencv_contrib's xphoto module ------------------#
# The -wb (auto white balance) option uses cv::xphoto::GrayworldWB, which lives
# in opencv_contrib and is NOT always installed alongside the main OpenCV
# modules (e.g. a vcpkg "opencv4" install without the "contrib" feature won't
# have it). Rather than hard-coding an assumption and breaking the build on
# environments without it, we check at configure time and compile -wb as a
# no-op if the header isn't available, so the build always succeeds either way.
include(CheckIncludeFileCXX)
set(CMAKE_REQUIRED_INCLUDES ${OpenCV_INCLUDE_DIRS})
check_include_file_cxx("opencv2/xphoto.hpp" HAVE_XPHOTO_HEADER)
unset(CMAKE_REQUIRED_INCLUDES)

if (HAVE_XPHOTO_HEADER)
    message(STATUS "opencv2/xphoto.hpp found - auto white balance (-wb) will be enabled")
else ()
    message(STATUS "opencv2/xphoto.hpp NOT found (opencv_contrib not installed) - "
                    "auto white balance (-wb) will be compiled as a no-op")
endif ()

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message(FATAL_ERROR "OpenMP Not Found!")
endif ()


if(USE_SYSTEM_NCNN)
    find_package(ncnn)
    if(NOT TARGET ncnn)
        message(WARNING "ncnn target not found! USE_SYSTEM_NCNN will be turned off.")
        set(USE_SYSTEM_NCNN OFF)
    endif()
endif()

if(NOT USE_SYSTEM_NCNN)
    # build ncnn library
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/ncnn/CMakeLists.txt")
        message(FATAL_ERROR "The submodules were not downloaded! Please update submodules with \"git submodule update --init --recursive\" and try again.")
    endif()

    option(NCNN_VULKAN "vulkan compute support" ON)
    option(NCNN_THREADS "" ON)
    option(NCNN_OPENMP "" ON)
    option(NCNN_VULKAN_ONLINE_SPIRV "" ON)

    option(NCNN_PIXEL "convert and resize from/to image pixel" ON)
    option(NCNN_PIXEL_ROTATE "rotate image pixel orientation" ON)
    option(NCNN_PIXEL_AFFINE "warp affine image pixel" ON)
    option(NCNN_PIXEL_DRAWING "draw basic figure and text" ON)
    option(NCNN_INT8 "int8 inference" ON)
    option(NCNN_BF16 "bf16 inference" ON)


    option(NCNN_INSTALL_SDK "" OFF)
    option(NCNN_PIXEL_ROTATE "" OFF)
    option(NCNN_BUILD_BENCHMARK "" OFF)
    option(NCNN_BUILD_TESTS "" OFF)
    option(NCNN_BUILD_TOOLS "" OFF)
    option(NCNN_BUILD_EXAMPLES "" OFF)
    option(NCNN_DISABLE_RTTI "" OFF)
    option(NCNN_DISABLE_EXCEPTION "" OFF)
    option(NCNN_SYSTEM_GLSLANG "" OFF)
    option(NCNN_SHARED_LIB "" OFF)



    add_subdirectory(external/ncnn)
endif()

if(USE_SYSTEM_WEBP)
    set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
    find_package(WebP)
    if(NOT TARGET webp)
        message(WARNING "webp target not found! USE_SYSTEM_WEBP will be turned off.")
        set(USE_SYSTEM_WEBP OFF)
    endif()
endif()

if(NOT USE_SYSTEM_WEBP)
    # build libwebp library
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/libwebp/CMakeLists.txt")
        message(FATAL_ERROR "The submodules were not downloaded! Please update submodules with \"git submodule update --init --recursive\" and try again.")
    endif()

    option(WEBP_ENABLE_SIMD "" ON)
    option(WEBP_BUILD_ANIM_UTILS "" OFF)
    option(WEBP_BUILD_CWEBP "" OFF)
    option(WEBP_BUILD_DWEBP "" OFF)
    option(WEBP_BUILD_GIF2WEBP "" OFF)
    option(WEBP_BUILD_IMG2WEBP "" OFF)
    option(WEBP_BUILD_VWEBP "" OFF)
    option(WEBP_BUILD_WEBPINFO "" OFF)
    option(WEBP_BUILD_WEBPMUX "" OFF)
    option(WEBP_BUILD_EXTRAS "" OFF)
    option(WEBP_BUILD_WEBP_JS "" OFF)
    option(WEBP_NEAR_LOSSLESS "" OFF)
    option(WEBP_ENABLE_SWAP_16BIT_CSP "" OFF)

    add_subdirectory(external/libwebp)

    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/libwebp/src)
endif()

set(NCNN_GFPGAN_LINK_LIBRARIES ncnn webp ${Vulkan_LIBRARY})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# RealESRGAN 배경 업스케일 셰이더 (preproc/postproc) 컴파일
set(SHADER_SPV_HEX_FILES)
compile_shader(realesrgan_preproc.comp)
compile_shader(realesrgan_postproc.comp)
# waifu2x cunet 노이즈 제거(-nn) 셰이더 (preproc/postproc) 컴파일
compile_shader(waifu2x_preproc.comp)
compile_shader(waifu2x_postproc.comp)
# RealCUGAN 배경 업스케일(애니메이션/일러스트 전용) 셰이더 컴파일
compile_shader(realcugan_preproc.comp)
compile_shader(realcugan_postproc.comp)
compile_shader(realcugan_4x_postproc.comp)
add_custom_target(generate-spirv DEPENDS ${SHADER_SPV_HEX_FILES})

add_executable(pixelforge gfpgan-ncnn-vulkan-demo.cpp gfpgan.cpp face.cpp realesrgan.cpp waifu2x_denoise.cpp realcugan.cpp)

# Waifu2xDenoise(-nn) 클래스 단독 검증용 테스트 도구 (GFPGAN/RealESRGAN 모델 불필요)
add_executable(test_waifu2x_denoise test_waifu2x_denoise.cpp waifu2x_denoise.cpp)
add_dependencies(pixelforge generate-spirv)
target_link_libraries(pixelforge ${NCNN_GFPGAN_LINK_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(test_waifu2x_denoise ${NCNN_GFPGAN_LINK_LIBRARIES} ${OpenCV_LIBS})

if (HAVE_XPHOTO_HEADER)
    target_compile_definitions(pixelforge PRIVATE HAVE_XPHOTO=1)
else ()
    target_compile_definitions(pixelforge PRIVATE HAVE_XPHOTO=0)
endif ()
