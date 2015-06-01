#/*****************************************************
# **  PIDX Parallel I/O Library                      **
# **  Copyright (c) 2010-2014 University of Utah     **
# **  Scientific Computing and Imaging Institute     **
# **  72 S Central Campus Drive, Room 3750           **
# **  Salt Lake City, UT 84112                       **
# **                                                 **
# **  PIDX is licensed under the Creative Commons    **
# **  Attribution-NonCommercial-NoDerivatives 4.0    **
# **  International License. See LICENSE.md.         **
# **                                                 **
# **  For information about this project see:        **
# **  http://www.cedmav.com/pidx                     **
# **  or contact: pascucci@sci.utah.edu              **
# **  For support: PIDX-support@visus.net            **
# **                                                 **
# *****************************************************/

IF (PIDX_BUILD_TEST)

  # ////////////////////////////////////////
  # sources
  # ////////////////////////////////////////

  #FILE(GLOB EXAMPLES_SOURCES *.h *.c *.cpp)
  
  # ////////////////////////////////////////
  # includes and link libraries
  # ////////////////////////////////////////

  INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/pidx)
  SET(EXAMPLES_LINK_LIBS pidx) 
  IF (MPI_CXX_FOUND)
    MESSAGE("Configuring examples with MPI support")
    INCLUDE_DIRECTORIES(${MPI_CXX_INCLUDE_PATH})
    SET(EXAMPLES_LINK_LIBS ${EXAMPLES_LINK_LIBS} ${MPI_CXX_LIBRARIES})
  ENDIF()


  # ////////////////////////////////////////
  # executable
  # ////////////////////////////////////////

  SET(S3D_CHECKPOINT_SOURCES checkpoint-restart/PIDX-s3d-checkpoint.c)
  PIDX_ADD_EXECUTABLE(pidx-s3d-checkpoint "${S3D_CHECKPOINT_SOURCES}")
  ADD_DEPENDENCIES(pidx-s3d-checkpoint pidx)
  TARGET_LINK_LIBRARIES(pidx-s3d-checkpoint ${EXAMPLES_LINK_LIBS})

  SET(S3D_RESTART_SOURCES checkpoint-restart/PIDX-s3d-restart.c)
  PIDX_ADD_EXECUTABLE(pidx-s3d-restart "${S3D_RESTART_SOURCES}")
  ADD_DEPENDENCIES(pidx-s3d-restart pidx)
  TARGET_LINK_LIBRARIES(pidx-s3d-restart ${EXAMPLES_LINK_LIBS})

  SET(UINTAH_CHECKPOINT_SOURCES checkpoint-restart/PIDX-uintah-checkpoint.c)
  PIDX_ADD_EXECUTABLE(pidx-uintah-checkpoint "${UINTAH_CHECKPOINT_SOURCES}")
  ADD_DEPENDENCIES(pidx-uintah-checkpoint pidx)
  TARGET_LINK_LIBRARIES(pidx-uintah-checkpoint ${EXAMPLES_LINK_LIBS})

  SET(UINTAH_RESTART_SOURCES checkpoint-restart/PIDX-uintah-restart.c)
  PIDX_ADD_EXECUTABLE(pidx-uintah-restart "${UINTAH_RESTART_SOURCES}")
  ADD_DEPENDENCIES(pidx-uintah-restart pidx)
  TARGET_LINK_LIBRARIES(pidx-uintah-restart ${EXAMPLES_LINK_LIBS})

  SET(PIDX_ROI_SOURCES ROI/PIDX-roi.c)
  PIDX_ADD_EXECUTABLE(pidx-roi "${PIDX_ROI_SOURCES}")
  ADD_DEPENDENCIES(pidx-roi pidx)
  TARGET_LINK_LIBRARIES(pidx-roi ${EXAMPLES_LINK_LIBS})

  SET(PIDX_REDUCED_RESOLUTION_WRITE_SOURCES ROI/PIDX-reduced-resolution-write.c)
  PIDX_ADD_EXECUTABLE(pidx-reduced-resolution-write "${PIDX_REDUCED_RESOLUTION_WRITE_SOURCES}")
  ADD_DEPENDENCIES(pidx-reduced-resolution-write pidx)
  TARGET_LINK_LIBRARIES(pidx-reduced-resolution-write ${EXAMPLES_LINK_LIBS})

  SET(PIDX_REDUCED_RESOLUTION_READ_SOURCES ROI/PIDX-reduced-resolution-read.c)
  PIDX_ADD_EXECUTABLE(pidx-reduced-resolution-read "${PIDX_REDUCED_RESOLUTION_READ_SOURCES}")
  ADD_DEPENDENCIES(pidx-reduced-resolution-read pidx)
  TARGET_LINK_LIBRARIES(pidx-reduced-resolution-read ${EXAMPLES_LINK_LIBS})

  SET(PIDX_ADAPTIVE_ROI_SOURCES ROI/PIDX-adaptive-roi.c)
  PIDX_ADD_EXECUTABLE(pidx-adaptive-roi "${PIDX_ADAPTIVE_ROI_SOURCES}")
  ADD_DEPENDENCIES(pidx-adaptive-roi pidx)
  TARGET_LINK_LIBRARIES(pidx-adaptive-roi ${EXAMPLES_LINK_LIBS})

  SET(PIDX_WRITE_COMPRESSED_SOURCES compression/PIDX-write-compressed.c)
  PIDX_ADD_EXECUTABLE(pidx-write-compressed "${PIDX_WRITE_COMPRESSED_SOURCES}")
  ADD_DEPENDENCIES(pidx-write-compressed pidx)
  TARGET_LINK_LIBRARIES(pidx-write-compressed ${EXAMPLES_LINK_LIBS})

  SET(PIDX_READ_COMPRESSED_SOURCES compression/PIDX-read-compressed.c)
  PIDX_ADD_EXECUTABLE(pidx-read-compressed "${PIDX_READ_COMPRESSED_SOURCES}")
  ADD_DEPENDENCIES(pidx-read-compressed pidx)
  TARGET_LINK_LIBRARIES(pidx-read-compressed ${EXAMPLES_LINK_LIBS})

  SET(PIDX_CHANGE_FILE_COUNT_SOURCES file-variations/PIDX-change-file-count.c)
  PIDX_ADD_EXECUTABLE(pidx-change-file-count "${PIDX_CHANGE_FILE_COUNT_SOURCES}")
  ADD_DEPENDENCIES(pidx-change-file-count pidx)
  TARGET_LINK_LIBRARIES(pidx-change-file-count ${EXAMPLES_LINK_LIBS})

  SET(PIDX_WRITE_BY_FIELD_SOURCES field-data-access/PIDX-write-by-field.c)
  PIDX_ADD_EXECUTABLE(pidx-write-by-field "${PIDX_WRITE_BY_FIELD_SOURCES}")
  ADD_DEPENDENCIES(pidx-write-by-field pidx)
  TARGET_LINK_LIBRARIES(pidx-write-by-field ${EXAMPLES_LINK_LIBS})

  SET(PIDX_READ_BY_FIELD_SOURCES field-data-access/PIDX-read-by-field.c)
  PIDX_ADD_EXECUTABLE(pidx-read-by-field "${PIDX_READ_BY_FIELD_SOURCES}")
  ADD_DEPENDENCIES(pidx-read-by-field pidx)
  TARGET_LINK_LIBRARIES(pidx-read-by-field ${EXAMPLES_LINK_LIBS})

  IF (PIDX_BUILD_CONVERT)

    SET(PNETCDF_TO_IDX_SOURCES convert/pnetcdf-to-idx.c)  
    SET(HDF_TO_IDX_SOURCES convert/hdf5-to-idx.c)
    SET(FILE_PER_PROCESS_SOURCES convert/file-per-process-to-idx.c)
    SET(RAW_TO_IDX_SOURCES convert/raw-to-idx.c)

    INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/pidx)
    IF (MPI_C_FOUND)
      INCLUDE_DIRECTORIES(${MPI_C_INCLUDE_PATH})
    ENDIF ()

    PIDX_ADD_EXECUTABLE(file-per-process-to-idx "${FILE_PER_PROCESS_SOURCES}")
    ADD_DEPENDENCIES(file-per-process-to-idx pidx)
    TARGET_LINK_LIBRARIES(file-per-process-to-idx ${EXAMPLES_LINK_LIBS})

    PIDX_ADD_EXECUTABLE(raw-to-idx "${RAW_TO_IDX_SOURCES}")
    ADD_DEPENDENCIES(raw-to-idx pidx)
    TARGET_LINK_LIBRARIES(raw-to-idx ${EXAMPLES_LINK_LIBS})

    if (PIDX_OPTION_HDF5)
      INCLUDE_DIRECTORIES(${HDF5_INCLUDE_DIR})

      PIDX_ADD_EXECUTABLE(hdf5_to_idx "${HDF_TO_IDX_SOURCES}")
      ADD_DEPENDENCIES(hdf5_to_idx pidx)
      TARGET_LINK_LIBRARIES(hdf5_to_idx pidx)
  
      IF (MPI_C_FOUND)
        TARGET_LINK_LIBRARIES(hdf5_to_idx ${MPI_C_LIBRARIES})
      ENDIF()

      ADD_DEPENDENCIES(hdf5_to_idx ${HDF5_LIBRARIES})
      TARGET_LINK_LIBRARIES(hdf5_to_idx ${HDF5_LIBRARIES})
    
    ENDIF()

    if (PIDX_OPTION_PNETCDF)
      INCLUDE_DIRECTORIES(${PNETCDF_INCLUDE_DIRS})

      PIDX_ADD_EXECUTABLE(pnetcdf_to_idx "${PNETCDF_TO_IDX_SOURCES}")
      ADD_DEPENDENCIES(pnetcdf_to_idx pidx)
      TARGET_LINK_LIBRARIES(pnetcdf_to_idx pidx)  

      IF (MPI_C_FOUND)
        TARGET_LINK_LIBRARIES(pnetcdf_to_idx ${MPI_C_LIBRARIES})
      ENDIF()

      ADD_DEPENDENCIES(pnetcdf_to_idx ${PNETCDF_LIBRARIES})
      TARGET_LINK_LIBRARIES(pnetcdf_to_idx ${PNETCDF_LIBRARIES})
    ENDIF()  

  ENDIF()


ENDIF()