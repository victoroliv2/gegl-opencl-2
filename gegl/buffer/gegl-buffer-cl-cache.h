#ifndef __GEGL_BUFFER_CL_CACHE_H__
#define __GEGL_BUFFER_CL_CACHE_H__

#include "gegl.h"
#include "gegl-types-internal.h"
#include "gegl-buffer-types.h"
#include "gegl-buffer.h"
#include "gegl-buffer-private.h"
#include "opencl/gegl-cl.h"

typedef enum
{
  GEGL_CL_BUFFER_NO_CACHE    = 0,
  GEGL_CL_BUFFER_CACHE_CLEAN = 1,
  GEGL_CL_BUFFER_CACHE_DIRTY = 2,
} GeglBufferClCacheMode;

/* Cache Entry */
typedef struct
{
  GeglBuffer           *buffer;

  cl_mem                tex;
  GeglRectangle         roi;
  GeglBufferClCacheMode mode;
  gboolean              locked;
} GeglBufferClCacheEntry;

cl_mem
gegl_buffer_cl_cache_get (GeglBuffer          *buffer,
                          const GeglRectangle *roi,
                          cl_int              *err);

void
gegl_buffer_cl_cache_set (GeglBuffer            *buffer,
                          cl_mem                 tex,
                          const GeglRectangle  *roi,
                          GeglBufferClCacheMode  mode);

gboolean
gegl_buffer_cl_cache_dispose (cl_mem tex);

cl_mem
gegl_buffer_cl_cache_request (GeglBuffer            *buffer,
                              cl_mem_flags           flags,
                              cl_image_format       *image_format,
                              const GeglRectangle   *roi,
                              GeglBufferClCacheMode  mode,
                              cl_int                *errcode_ret);

void
gegl_buffer_cl_cache_invalidate (GeglBuffer          *buffer,
                                 const GeglRectangle *roi);

gboolean
gegl_buffer_cl_cache_from (GeglBuffer          *buffer,
                           const GeglRectangle *roi,
                           gpointer             dest_buf,
                           const Babl          *format,
                           gint                 rowstride);
void
gegl_buffer_cl_cache_clear (GeglBuffer          *buffer,
                            const GeglRectangle *roi);

void
gegl_buffer_cl_cache_remove (GeglBuffer *buffer);

#endif
