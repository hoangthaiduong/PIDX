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

#ifndef __PIDX_BOX_RST_H
#define __PIDX_BOX_RST_H 


struct PIDX_block_rst_id_struct;
typedef struct PIDX_block_rst_id_struct* PIDX_block_rst_id;


PIDX_block_rst_id PIDX_block_rst_init(idx_dataset idx_meta_data, idx_dataset_derived_metadata idx_derived_ptr, int start_var_index, int end_var_index);

#if PIDX_HAVE_MPI
int PIDX_block_rst_set_communicator(PIDX_block_rst_id id, MPI_Comm comm);
#endif

int PIDX_block_rst_prepare(PIDX_block_rst_id id, PIDX_variable* variable);

int PIDX_block_rst_compress(PIDX_block_rst_id id, PIDX_variable* variable, int MODE);
  
int PIDX_block_rst_buf_destroy(PIDX_block_rst_id id);

int PIDX_block_rst_finalize(PIDX_block_rst_id id);  

#endif