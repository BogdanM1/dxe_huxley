CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

project(huxpinn)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Bogdan surro interface 
include_directories(/home/bogdan/tensorflow/include)
link_directories(/home/bogdan/tensorflow/lib)


add_executable(huxpinn "src/huxpinn_interface.cpp" "src/tf_utils.cpp")
target_link_libraries(huxpinn tensorflow)
