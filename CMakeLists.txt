cmake_minimum_required(VERSION 3.15)
include(${CMAKE_CURRENT_SOURCE_DIR}/Resources/pd.cmake/pd.cmake)
project(perceptron-object)

#╭──────────────────────────────────────╮
#│              Static GSL              │
#╰──────────────────────────────────────╯
set(GSL_DISABLE_TESTS ON)
add_subdirectory(Libraries/gsl)
set_target_properties(gsl PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(gsl PROPERTIES POSITION_INDEPENDENT_CODE ON)

#╭──────────────────────────────────────╮
#│             Pd External              │
#╰──────────────────────────────────────╯
# Get all cpp files inside ${CMAKE_SOURCE_DIR}/Sources/ and add them to the project
file(GLOB EXTERNAL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Sources/*.c")
add_pd_external(perceptron-object perceptron ${EXTERNAL_SOURCES})
target_link_libraries(perceptron-object gsl)
target_include_directories(perceptron-object PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/gsl/")

if(${MSVC})
    set_property(TARGET ${PROJECT_NAME} APPEND_STRING PROPERTY COMPILE_FLAGS "/wd4091 /wd4996")
endif()


target_compile_options(perceptron-object PUBLIC
                "-O3" 
                "-DPD"
)
