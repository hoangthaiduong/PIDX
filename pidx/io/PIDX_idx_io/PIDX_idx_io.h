//#include "../../PIDX_inc.h"
//#include "../PIDX_io.h"

#ifndef __PIDX_IDX_IO_H
#define __PIDX_IDX_IO_H


struct PIDX_idx_io_descriptor;
typedef struct PIDX_idx_io_descriptor* PIDX_idx_io;


///
PIDX_idx_io PIDX_idx_io_init( idx_dataset idx_meta_data, idx_dataset_derived_metadata idx_derived_ptr, idx_debug idx_dbg);


#if PIDX_HAVE_MPI
/// Attach the communicator wit the ID.
/// \param id restructuring id
/// \param comm the communicator
/// \return error code
PIDX_return_code PIDX_idx_io_set_communicator(PIDX_idx_io id, MPI_Comm comm);
#endif


///
PIDX_return_code PIDX_idx_write(PIDX_idx_io file, int start_var_index, int end_var_index);


///
PIDX_return_code PIDX_idx_read(PIDX_idx_io file, int start_var_index, int end_var_index);


///
PIDX_return_code PIDX_idx_io_finalize(PIDX_idx_io file);

#endif
