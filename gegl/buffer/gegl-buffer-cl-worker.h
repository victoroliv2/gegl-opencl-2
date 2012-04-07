#ifndef __GEGL_BUFFER_CL_WORKER_H__
#define __GEGL_BUFFER_CL_WORKER_H__

void gegl_buffer_cl_worker_transf (GeglBuffer *buffer, gpointer data, size_t pixel_size, GeglRectangle roi, gboolean write);

#endif