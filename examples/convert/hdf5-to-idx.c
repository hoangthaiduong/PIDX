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

#include <PIDX.h>
#include <hdf5.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#if PIDX_HAVE_MPI
#include <mpi.h>
#endif

#define HDF5_IO 1
#define USE_DOUBLE
#define PIDX_ROW_OR_COLUMN_MAJOR PIDX_row_major

enum { X, Y, Z, NUM_DIMS };
enum AtomicType { CHAR = 1, INT = 4, FLOAT = 4, DOUBLE = 8, INVALID };
struct Type
{
  enum AtomicType atomic_type;
  unsigned long long num_values;
};
static unsigned long long global_box_size[NUM_DIMS] = { 0 }; ///< global dimensions of 3D volume
static unsigned long long local_box_size[NUM_DIMS] = { 0 }; ///< local dimensions of the per-process block
static unsigned long long local_box_offset[NUM_DIMS] = { 0 };
static PIDX_point pidx_global_box_size = { 0 };
static PIDX_point pidx_local_box_offset = { 0 };
static PIDX_point pidx_local_box_size = { 0 };
static int process_count = 1; ///< Number of processes
static int rank = 0;
static int time_step_count = 1; ///< Number of time-steps
static int var_count = 1; ///< Number of fields
static char output_file_template[512] = "test"; ///< output IDX file Name Template
static char output_file_name[512] = "test.idx";
static char var_file[512];
static char hdf5_file_list[512];
static char **var_names = 0;
static char **hdf5_file_names = 0;
static struct Type *var_types = 0;
static PIDX_variable *pidx_vars = 0;
static void *var_data = 0;

static char *usage = "Serial Usage: ./hdf5-to-idx -g 4x4x4 -l 4x4x4 -v var_list -i hdf5_file_names_list -f output_idx_file_name\n"
                     "Parallel Usage: mpirun -n 8 ./hdf5-to-idx -g 4x4x4 -l 2x2x2 -f Filename_ -v var_list -i hdf5_file_names_list\n"
                     "  -g: global dimensions\n"
                     "  -l: local (per-process) dimensions\n"
                     "  -f: IDX filename\n"
                     "  -i: file containing list of input hdf5 files\n"
                     "  -v: file containing list of input fields\n";

//----------------------------------------------------------------
// TODO: write another function to release resources before terminating
static void terminate()
{
#if PIDX_HAVE_MPI
  MPI_Abort(MPI_COMM_WORLD, -1);
#else
  exit(-1);
#endif
}

//----------------------------------------------------------------
static void terminate_with_error_msg(const char *format, ...)
{
  va_list arg_ptr;
  va_start(arg_ptr, format);
  vfprintf(stderr, format, arg_ptr);
  va_end(arg_ptr);
  terminate();
}

//----------------------------------------------------------------
static void rank_0_print(const char *format, ...)
{
  if (rank != 0) return;
  va_list arg_ptr;
  va_start(arg_ptr, format);
  vfprintf(stderr, format, arg_ptr);
  va_end(arg_ptr);
}

//----------------------------------------------------------------
static void init_mpi(int argc, char **argv)
{
#if PIDX_HAVE_MPI
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
    terminate_with_error_msg("ERROR: MPI_Init error\n");
  if (MPI_Comm_size(MPI_COMM_WORLD, process_count) != MPI_SUCCESS)
    terminate_with_error_msg("ERROR: MPI_Comm_size error\n");
  if (MPI_Comm_rank(MPI_COMM_WORLD, rank) != MPI_SUCCESS)
    terminate_with_error_msg("ERROR: MPI_Comm_rank error\n");
#endif
}

//----------------------------------------------------------------
static void shutdown_mpi()
{
#if PIDX_HAVE_MPI
  MPI_Finalize();
#endif
}

//----------------------------------------------------------------
// Read all the lines in a file into a list.
// Return the number of lines read.
// A line starting with // is treated as a comment and does not increase the count.
static int read_list_in_file(const char *file_name, char **list)
{
  assert(file_name != 0);

  FILE *fp = 0;
  if (fopen(file_name, "r") == 0)
    return 0;
  
  // first pass, count the number of non-comment lines
  int line_count = 0;
  char line[512]; // a line cannot be longer than 512 characters
  while (fgets(line, sizeof(line), fp) != 0)
  {
    if (line[0] != '/' || line[1] != '/')
      ++line_count;
  }
  
  // second pass, actually read the data
  list = (char **)calloc(line_count, sizeof(*list));
  if (list == 0)
    terminate_with_error_msg("ERROR: Failed to allocate memory for a list of names. Bytes requested = %d (items) * %u (bytes)\n", line_count, sizeof(*list));
  rewind(fp);
  int i = 0;
  while (fgets(line, sizeof(line), fp) != 0)
  {
    if (line[0] != '/' || line[1] != '/')
    {
      line[strcspn(line, "\r\n")] = 0; // trim the newline character at the end if any
      list[i] = strdup(line);
      ++i;
    }
  }
  
  fclose(fp);
  return line_count;
}

//----------------------------------------------------------------
static void free_memory()
{
  free(var_data);
  var_data = 0;
  
  int i = 0;
  for (i = 0; i < var_count; i++)
  {
    free(var_names[i]);
    var_names[i] = 0;
  }
  free(var_names);
  var_names = 0;
  free(var_types);
  free(pidx_vars);
  
  for (i = 0; i < time_step_count; i++)
  {
    free(hdf5_file_names[i]);
    hdf5_file_names[i] = 0;
  }
  free(hdf5_file_names);
  hdf5_file_names = 0;
}

//----------------------------------------------------------------
///< Parse the input arguments
static void parse_args(int argc, char **argv)
{
  char flags[] = "g:l:f:i:v:";
  int opt = 0;
  while ((opt = getopt(argc, argv, flags)) != -1)
  {
    switch (opt)
    {
      case('g'): // global dimension
        if ((sscanf(optarg, "%lldx%lldx%lld", &global_box_size[0], &global_box_size[1], &global_box_size[2]) == EOF) ||
            (global_box_size[0] < 1 || global_box_size[1] < 1 || global_box_size[2] < 1))
          terminate_with_error_msg("Invalid global dimensions\n%s", usage);
        break;
      case('l'): // local dimension
        if ((sscanf(optarg, "%lldx%lldx%lld", &local_box_size[0], &local_box_size[1], &local_box_size[2]) == EOF) ||
            (local_box_size[0] < 1 || local_box_size[1] < 1 || local_box_size[2] < 1))
          terminate_with_error_msg("Invalid local dimension\n%s", usage);
        break;
      case('f'): // output file name
        if (sprintf(output_file_template, "%s", optarg) < 0)
          terminate_with_error_msg("Invalid output file name template\n%s", usage);
        sprintf(output_file_name, "%s%s", output_file_template, ".idx");
        break;
      case('i'): // a file with a list of hdf5 files
        if (sprintf(hdf5_file_list, "%s", optarg) < 0)
          terminate_with_error_msg("Invalid input file\n%s", usage);
        break;
      case('v'): // a file with a list of variables
        if (sprintf(var_file, "%s", optarg) < 0)
          terminate_with_error_msg("Invalid variable file\n%s", usage);
        break;
      default:
        terminate_with_error_msg("Wrong arguments\n%s", usage);
    }
  }
}

//----------------------------------------------------------------
static void check_args()
{
  if (global_box_size[X] < local_box_size[X] || global_box_size[Y] < local_box_size[Y] || global_box_size[Z] < local_box_size[Z])
    terminate_with_error_msg("ERROR: Global box is smaller than local box in one of the dimensions\n");
  
  // check if the number of processes given by the user is consistent with the actual number of procesess needed
  int brick_count = (int)((global_box_size[X] + local_box_size[X] - 1) / local_box_size[X]) *
                    (int)((global_box_size[Y] + local_box_size[Y] - 1) / local_box_size[Y]) *
                    (int)((global_box_size[Z] + local_box_size[Z] - 1) / local_box_size[Z]);
  if(brick_count != process_count)
    terminate_with_error_msg("ERROR: Number of sub-blocks (%d) doesn't match number of processes (%d)\n", brick_count, process_count);
}

//----------------------------------------------------------------
static void calculate_per_process_offsets()
{
  int sub_div[NUM_DIMS];
  sub_div[X] = (global_box_size[X] / local_box_size[X]);
  sub_div[Y] = (global_box_size[Y] / local_box_size[Y]);
  sub_div[Z] = (global_box_size[Z] / local_box_size[Z]);
  local_box_offset[Z] = (rank / (sub_div[X] * sub_div[Y])) * local_box_size[Z];
  int slice = rank % (sub_div[X] * sub_div[Y]);
  local_box_offset[Y] = (slice / sub_div[X]) * local_box_size[Y];
  local_box_offset[X] = (slice % sub_div[X]) * local_box_size[X];
}

//----------------------------------------------------------------
// Convert an atomic HDF5 datatype to native C datatype
static enum AtomicType from_hdf5_atomic_type(hid_t atomic_type_id)
{
  if (H5Tequal(atomic_type_id, H5T_NATIVE_CHAR))
    return CHAR;
  if (H5Tequal(atomic_type_id, H5T_NATIVE_INT))
    return INT;
  if (H5Tequal(atomic_type_id, H5T_NATIVE_FLOAT))
    return FLOAT;
  if (H5Tequal(atomic_type_id, H5T_NATIVE_DOUBLE))
    return DOUBLE;
  return INVALID;
}

//----------------------------------------------------------------
// Convert a HDF5 datatype to C native or array datatype
static struct Type from_hdf5_type(hid_t type_id)
{
  struct Type type;
  H5T_class_t type_class = H5Tget_class(type_id);
  if (type_class == H5T_ARRAY)
  {
    int num_dims = H5Tget_array_ndims(type_id);
    if (num_dims != 1)
    {
      type.atomic_type = INVALID; // we don't support arrays of more than 1 dimension
      return type;
    }
    if (H5Tget_array_dims2(type_id, &type.num_values) < 0)
    {
      type.atomic_type = INVALID;
      return type;
    }
  }
  else if (type_class == H5T_FLOAT || H5T_INTEGER)
  {
    type.num_values = 1;
  }
  else // we don't support HD5_COMPOUND datatype for example
  {
    type.atomic_type = INVALID;
  }
  
  hid_t atomic_type_id = H5Tget_native_type(type_id, H5T_DIR_DESCEND);
  type.atomic_type = from_hdf5_atomic_type(atomic_type_id);
  H5Tclose(atomic_type_id);
  return type;
}

//----------------------------------------------------------------
// Open the first HDF5 file and query all the types for all the variables
static void determine_var_types(hid_t plist_id)
{
  assert(hdf5_file_names != 0);
  assert(var_names != 0);
  
  hid_t file_id = H5Fopen(hdf5_file_names[0], H5F_ACC_RDONLY, plist_id);
  if (file_id < 0)
    terminate_with_error_msg("ERROR: Cannot open file %s\n", hdf5_file_names[0]);
  
  var_types = (struct Type *)calloc(var_count, sizeof(*var_types));
  
  int i = 0;
  for (i = 0; i < var_count; ++i)
  {
    hid_t dataset_id = H5Dopen2(file_id, var_names[i], H5P_DEFAULT);
    if (dataset_id < 0)
      terminate_with_error_msg("ERROR: Variable %s does not exist in file %s\n", var_names[i], hdf5_file_names[0]);
    hid_t type_id = H5Dget_type(dataset_id);
    if (type_id < 0)
      terminate_with_error_msg("ERROR: Failed to query the type of variable %s\n", var_names[i]);
    var_types[i] = from_hdf5_type(type_id);
    if (var_types[i].atomic_type == INVALID)
      terminate_with_error_msg("ERROR: The datatype of the %s variable is not supported\n");
    H5Tclose(type_id);
  }
  
  H5Fclose(file_id);
}

//----------------------------------------------------------------
// Return a negative value when failed, otherwise return 0
int read_var_from_hdf5(hid_t file_id, const char *var_name, struct Type type)
{
  assert(var_name != 0);
  assert(var_data != 0);
  
  hid_t dataset_id = H5Dopen2(file_id, var_name, H5P_DEFAULT);
  if (dataset_id < 0)
    terminate_with_error_msg("ERROR: Failed to open HDF5 dataset for variable %s\n", var_name);
  hid_t mem_dataspace = H5Screate_simple(3, local_box_size, 0);
  if (mem_dataspace < 0)
    terminate_with_error_msg("ERROR: Failed to create memory dataspace for variable %s\n", var_name);
  hid_t file_dataspace = H5Dget_space(dataset_id);
  if (file_dataspace < 0)
    terminate_with_error_msg("ERROR: Failed to create file dataspace for variable %s\n", var_name);
  if (H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, local_box_offset, 0, local_box_size, 0) < 0)
    terminate_with_error_msg("ERROR: Failed to create a hyperslab for variable %s\n", var_name);
  herr_t read_error = 0;
  if (type.atomic_type == DOUBLE)
    read_error = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, mem_dataspace, file_dataspace, H5P_DEFAULT, var_data);
  else if (type.atomic_type == FLOAT)
    read_error = H5Dread(dataset_id, H5T_NATIVE_FLOAT, mem_dataspace, file_dataspace, H5P_DEFAULT, var_data);
  else if (type.atomic_type == INT)
    read_error = H5Dread(dataset_id, H5T_NATIVE_INT, mem_dataspace, file_dataspace, H5P_DEFAULT, var_data);
  else if (type.atomic_type == CHAR)
    read_error = H5Dread(dataset_id, H5T_NATIVE_CHAR, mem_dataspace, file_dataspace, H5P_DEFAULT, var_data);
  else
    terminate_with_error_msg("ERROR: Unsupported type. Type = %d\n", type.atomic_type);
  H5Sclose(mem_dataspace);
  H5Sclose(file_dataspace);
  H5Dclose(dataset_id);
  
  if (read_error < 0)
    return -1;
  return 0;
}

//----------------------------------------------------------------
static void to_idx_type_string(struct Type type, char *type_string)
{
  assert(type_string != 0);
  
  if (type.atomic_type == DOUBLE)
    sprintf(type_string, "%lld*float64", type.num_values);
  else if (type.atomic_type == FLOAT)
      sprintf(type_string, "%lld*float32", type.num_values);
  else if (type.atomic_type == INT)
    sprintf(type_string, "%lld*int32", type.num_values);
  else if (type.atomic_type == CHAR)
    sprintf(type_string, "%lld*int8", type.num_values);
  else
    terminate_with_error_msg("ERROR: Unsupported type. Type = %d\n", type.atomic_type);
}

//----------------------------------------------------------------
static void create_pidx_vars(PIDX_file pidx_file)
{
  assert(var_names != 0);
  assert(var_types != 0);
  
  pidx_vars = (PIDX_variable *)calloc(var_count, sizeof(*pidx_vars));
  int i = 0;
  for (i = 0; i < var_count; ++i)
  {
    char type_string[32] = { 0 };
    to_idx_type_string(var_types[i], type_string);
    int ret = PIDX_variable_create(var_names[i], var_types[i].atomic_type * 8, type_string, &pidx_vars[i]);
    if (ret != PIDX_success)
      terminate_with_error_msg("ERROR: PIDX failed to create variable %s\n", var_names[i]);
    rank_0_print("Successfully create variable %s, type = %s", var_names[i], type_string);
  }
}

//----------------------------------------------------------------
static void write_var_to_idx(PIDX_file pidx_file, const char *var_name, struct Type type, PIDX_variable pidx_var)
{
  assert(var_name != 0);
  assert(var_data != 0);
  
  PIDX_set_point_5D(pidx_local_box_offset, local_box_offset[X], local_box_offset[Y], local_box_offset[Z], 0, 0);
  PIDX_set_point_5D(pidx_local_box_size, local_box_size[X], local_box_size[Y], local_box_size[Z], 1, 1);
  int ret = PIDX_success;
  ret = PIDX_variable_data_layout(pidx_var, pidx_local_box_offset, pidx_local_box_size, var_data, PIDX_ROW_OR_COLUMN_MAJOR);
  if (ret != PIDX_success)
    terminate_with_error_msg("ERROR: PIDX failed to specify variable data layout for %s\n", var_name);
  ret = PIDX_append_and_write_variable(pidx_file, pidx_var);
  if (ret != PIDX_success)
    terminate_with_error_msg("ERROR: PIDX failed to append and write variable %s\n", var_name);
}

//----------------------------------------------------------------
static void create_pidx_file_and_access(PIDX_file *pidx_file, PIDX_access *pidx_access)
{
  int ret = PIDX_create_access(pidx_access);
  if (ret != PIDX_success)
    terminate_with_error_msg("ERROR: Failed to create PIDX access\n");
#if PIDX_HAVE_MPI
  ret = PIDX_set_mpi_access(*pidx_access, MPI_COMM_WORLD);
  if (ret != PIDX_success)
    terminate_with_error_msg("ERROR: Failed to create PIDX MPI access\n");
#endif
}

//----------------------------------------------------------------
static void set_pidx_params(PIDX_file pidx_file)
{
  int ret = PIDX_set_point_5D(pidx_global_box_size, global_box_size[X], global_box_size[Y], global_box_size[Z], 1, 1);
  ret = PIDX_set_dims(pidx_file, pidx_global_box_size);
  if (ret != PIDX_success)
    terminate_with_error_msg("ERROR: Failed to set PIDX global dimensions\n");
  ret = PIDX_set_variable_count(pidx_file, var_count);
  if (ret != 0)
    terminate_with_error_msg("ERROR: Failed to set PIDX variable count\n");
  PIDX_time_step_caching_ON();
}

//----------------------------------------------------------------
static void shutdown_pidx(PIDX_file pidx_file, PIDX_access pidx_access)
{
  PIDX_time_step_caching_OFF();  
  PIDX_close_access(pidx_access);
}

//----------------------------------------------------------------
static hid_t init_hdf5()
{
  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  if (plist_id < 0)
    terminate_with_error_msg("ERROR: Failed to create HDF5 file access\n");
#if PIDX_HAVE_MPI
  if (H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL) < 0)
    terminate_with_error_msg("ERROR: Failed to enable MPI for HDF5\n");
#endif
  return plist_id;
}

//----------------------------------------------------------------
int main(int argc, char **argv)
{
  init_mpi(argc, argv);
  parse_args(argc, argv);
  check_args();
  calculate_per_process_offsets();
  var_count = read_list_in_file(var_file, var_names);
  rank_0_print("Number of variables = %d\n", var_count);
  time_step_count = read_list_in_file(hdf5_file_list, hdf5_file_names);
  rank_0_print("Number of timesteps = %d\n", time_step_count);
  
  PIDX_file pidx_file;
  PIDX_access pidx_access;
  create_pidx_file_and_access(&pidx_file, &pidx_access);
  set_pidx_params(pidx_file);
  
  hid_t plist_id = init_hdf5();
  determine_var_types(plist_id);
  create_pidx_vars(pidx_file);
  
  int t = 0;
  for (t = 0; t < time_step_count; ++t)
  {
    ret = PIDX_file_create(output_file_name, PIDX_file_trunc, *pidx_access, pidx_file);
    if (ret != PIDX_success)
      terminate_with_error_msg("ERROR: Failed to create PIDX file\n");
      
    rank_0_print("Processing time step %d\n", t);
    PIDX_set_current_time_step(pidx_file, t);
    hid_t file_id = H5Fopen(hdf5_file_names[t], H5F_ACC_RDONLY, plist_id);
    if (file_id < 0)
      terminate_with_error_msg("ERROR: Failed to open file %s\n", hdf5_file_names[t]);
    int v = 0;
    for (v = 0; v < var_count; ++v)
    {
      rank_0_print("Processing variable %s\n", var_names[v]);
      if (read_var_from_hdf5(file_id, var_names[v], var_types[v]) < 0)
        terminate_with_error_msg("ERROR: Failed to read variable %s from file %s\n", var_names[v], hdf5_file_names[t]);
      write_var_to_idx(pidx_file, var_names[v], var_types[v], pidx_vars[v]);
      if (PIDX_flush(pidx_file) != PIDX_success)
        terminate_with_error_msg("ERROR: Failed to flush variable %s, time step %d\n", var_names[v], t);
    }
    H5Fclose(file_id);
    PIDX_close(pidx_file);
  }
  
  shutdown_pidx(pidx_file, pidx_access);
  H5Pclose(plist_id);
  
  free_memory();
  shutdown_mpi();
}
