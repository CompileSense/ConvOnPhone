# Sets the minimum version of CMake required to build the native
# library. You should either keep the default value or only pass a
# value of 3.4.0 or lower.

cmake_minimum_required(VERSION 3.4.1)



# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_SRC 变量
aux_source_directory(./src/main/cpp DIR_SRC)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds it for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
             native-lib

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             # Associated headers in the same location as their source
             # file are automatically included.
             ${DIR_SRC})

# Specifies a path to native header files.
include_directories(./src/main/cpp/openblas/arm64-v8ainclude)
include_directories(/Users/shenjingyuan002/work/opencl/include)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because system libraries are included in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in the
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
                       native-lib

                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib}
                       /Users/shenjingyuan002/Documents/AndroidStudioProjects/android-ndk/ConvOnPhone/app/src/main/JniLibs/libopenblas_armv8-r0.2.20.dev.a
                       /Users/shenjingyuan002/work/opencl/lib64/libOpenCL.so
                       )