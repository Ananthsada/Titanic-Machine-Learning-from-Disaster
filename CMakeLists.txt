cmake_minimum_required(VERSION 3.16)

project(titanic)
set(CSV_CXX_STANDARD 17)

set(SOURCES src/main.cpp)
include_directories(inc)

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} -lpthread)