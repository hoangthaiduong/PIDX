/*****************************************************
 **  PIDX Parallel I/O Library                      **
 **  Copyright (c) 2010-2014 University of Utah     **
 **  Scientific Computing and Imaging Institute     **
 **  72 S Central Campus Drive, Room 3750           **
 **  Salt Lake City, UT 84112                       **
 **                                                 **
 **  PIDX is licensed under the Creative Commons    **
 **  Attribution-NonCommercial-NoDerivatives 4.0    **
 **  International License. See LICENSE.md.         **
 **                                                 **
 **  For information about this project see:        **
 **  http://www.cedmav.com/pidx                     **
 **  or contact: pascucci@sci.utah.edu              **
 **  For support: PIDX-support@visus.net            **
 **                                                 **
 *****************************************************/

// This program reads a raw file and converts it into a compressed IDX file
#include "pidxtest.h"

#ifdef __cplusplus
extern "C" {
#endif

int test_converter(struct Args args, int rank)
{
  /// The command line arguments are shared by all processes
  MPI_Bcast(args.extents, 5, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(args.count_local, 5, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&args.output_file_template, 512, MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast(args.restructured_box_size, 5, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&args.compression_type, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&args.compression_bit_rate, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&args.perform_brst, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&args.perform_hz, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&args.perform_compression, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&args.perform_agg, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&args.perform_io, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&args.dump_agg, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&args.blocks_per_file, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&args.bits_per_block, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&args.aggregation_factor, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Creating the filenames
  char *input_file_name = (char *)malloc(sizeof(char) * 512);
  sprintf(input_file_name, "%s", args.output_file_template);
  args.output_file_name = (char *)malloc(sizeof(char) * 512);
  sprintf(args.output_file_name, "%s-bricked%s", args.output_file_template, ".idx");

  // Calculating every process's offset and count
  unsigned int sub_div[3], local_offset[3];
  sub_div[0] = (args.extents[0] / args.count_local[0]);
  sub_div[1] = (args.extents[1] / args.count_local[1]);
  sub_div[2] = (args.extents[2] / args.count_local[2]);
  int slice = rank % (sub_div[0] * sub_div[1]);
  local_offset[2] = (rank / (sub_div[0] * sub_div[1])) * args.count_local[2];
  local_offset[1] = (slice / sub_div[0]) * args.count_local[1];
  local_offset[0] = (slice % sub_div[0]) * args.count_local[0];

  unsigned int rank_x = 0, rank_y = 0, rank_z = 0, rank_slice = 0;
  rank_z = rank / (sub_div[0] * sub_div[1]);
  rank_slice = rank % (sub_div[0] * sub_div[1]);
  rank_y = (rank_slice / sub_div[0]);
  rank_x = (rank_slice % sub_div[0]);

  PIDX_point global_bounding_box, local_offset_point, local_box_count_point, restructured_box_size_point;
  PIDX_set_point_5D(global_bounding_box, args.extents[0], args.extents[1], args.extents[2], 1, 1);
  PIDX_set_point_5D(local_offset_point, local_offset[0], local_offset[1], local_offset[2], 0, 0);
  PIDX_set_point_5D(local_box_count_point, args.count_local[0], args.count_local[1], args.count_local[2], 1, 1);
  PIDX_set_point_5D(restructured_box_size_point, args.restructured_box_size[0], args.restructured_box_size[1], args.restructured_box_size[2], 1, 1);

  PIDX_time_step_caching_ON();
  int time_step = 0;
  for (time_step = 0; time_step < 1; time_step++)
  {
    PIDX_access access;
    PIDX_create_access(&access);
    PIDX_set_mpi_access(access, MPI_COMM_WORLD);
    PIDX_set_global_indexing_order(access, 0);
    PIDX_set_process_extent(access, sub_div[0], sub_div[1], sub_div[2]);
    PIDX_set_process_rank_decomposition(access, rank_x, rank_y, rank_z);

    int variable_count = 1;
    int var = 0;

    double **double_data = (double **)malloc(sizeof(*double_data) * variable_count); // DUONG_HARDCODE?
    memset(double_data, 0, sizeof(*double_data) * variable_count);
    PIDX_variable *write_variable;
    
    var = 0;
    double_data[var] = (double*)malloc(sizeof (double) * args.count_local[0] * args.count_local[1] * args.count_local[2]  * 1);
    int fp = open(input_file_name, O_RDONLY);
    read(fp, double_data[0], 512*256*256*sizeof(double));
    close(fp);

    // set write parameters
    PIDX_file output_file;
    PIDX_file_create(args.output_file_name, PIDX_file_trunc, access, &output_file);
    PIDX_set_dims(output_file, global_bounding_box);
    PIDX_set_current_time_step(output_file, time_step);
    PIDX_set_block_size(output_file, 16);
    PIDX_set_block_count(output_file, 512);
    PIDX_set_variable_count(output_file, variable_count);
    PIDX_set_aggregation_factor(output_file, args.aggregation_factor);
    PIDX_set_compression_type(output_file, args.compression_type);
    PIDX_set_lossy_compression_bit_rate(output_file, args.compression_bit_rate);
    PIDX_set_restructuring_box(output_file, restructured_box_size_point);

    write_variable = (PIDX_variable *)malloc(sizeof(*write_variable) * variable_count);
    memset(write_variable, 0, sizeof(*write_variable) * variable_count);

    // write
    for (var = 0; var < variable_count; var++)
    {
      PIDX_variable_create("var_name", 1 * sizeof(uint64_t) * 8, "1*float64", &write_variable[var]);
      PIDX_variable_data_layout(write_variable[var], local_offset_point, local_box_count_point, double_data[var], PIDX_row_major);
      PIDX_append_and_write_variable(output_file, write_variable[var]);
    }
    PIDX_close(output_file);
    PIDX_close_access(access);

    // free stuffs
    for (int i = 0; i < variable_count; ++i)
    {
      free(double_data[i]);
    }
    free(double_data);
    double_data = 0;

    free(write_variable);
    write_variable = 0;
  }
  PIDX_time_step_caching_OFF();
  free(args.output_file_name);
  free(input_file_name);
  return 0;
}

#ifdef __cplusplus
}
#endif
