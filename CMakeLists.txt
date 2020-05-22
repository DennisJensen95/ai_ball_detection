cmake_minimum_required(VERSION 3.16)
project(YOLO LANGUAGES C CXX VERSION 0.0.1)

set(YOLO_SRC
    src/main.cpp
    )

find_package(OpenCV REQUIRED PATHS /usr/local/share/OpenCV NO_DEFAULT_PATH)

add_executable(yolo
    ${YOLO_SRC}
)

target_link_libraries(yolo
    ${OpenCV_LIBRARIES}
)