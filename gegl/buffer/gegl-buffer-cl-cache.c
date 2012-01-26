#include <glib.h>

#include "gegl.h"
#include "gegl-utils.h"
#include "gegl-types-internal.h"
#include "gegl-buffer-types.h"
#include "gegl-buffer.h"
#include "gegl-buffer-private.h"
#include "gegl-buffer-cl-cache.h"
#include "opencl/gegl-cl.h"

/* This is a write-back cache with write allocate */
static GQueue cache_entries = G_QUEUE_INIT;

static gint
find_roi_equal (gconstpointer a, gconstpointer b)
{
  GeglBufferClCacheEntry *entry = (GeglBufferClCacheEntry *) a;
  GeglRectangle *roi            = (GeglRectangle*)           b;

  if (entry->mode == GEGL_CL_BUFFER_NO_CACHE)
    return 0;

  if (gegl_rectangle_equal (&entry->roi, roi))
    return 0;
  else
    return 1;
}

static gint
find_tex (gconstpointer a, gconstpointer b)
{
  GeglBufferClCacheEntry *entry = (GeglBufferClCacheEntry *) a;
  cl_mem tex                    = (cl_mem)                   b;

  if (entry->mode == GEGL_CL_BUFFER_NO_CACHE)
    return 0;

  if (entry->tex == tex)
    return 0;
  else
    return 1;
}

static void
bump_entry (GeglBufferClCacheEntry *entry)
{
  g_queue_remove    (&cache_entries, entry);
  g_queue_push_head (&cache_entries, entry);

  g_queue_remove    (entry->buffer->cl_cache, entry);
  g_queue_push_head (entry->buffer->cl_cache, entry);
}

cl_mem
gegl_buffer_cl_cache_get (GeglBuffer          *buffer,
                          const GeglRectangle *roi,
                          cl_int              *err)
{
  GList *result = g_queue_find_custom (buffer->cl_cache, roi, find_roi_equal);

  if (result)
    {
      GeglBufferClCacheEntry *entry = g_list_first(result)->data;

      bump_entry (entry);

      return entry->tex;
    }
  else
    {
      return NULL;
    }
}

void
gegl_buffer_cl_cache_set (GeglBuffer            *buffer,
                          cl_mem                 tex,
                          const GeglRectangle   *roi,
                          GeglBufferClCacheMode  mode)
{
  GeglBufferClCacheEntry *new_entry = g_slice_new0 (GeglBufferClCacheEntry);

  new_entry->buffer = buffer;
  new_entry->tex    = tex;
  new_entry->roi    = *roi;
  new_entry->mode   = mode;
  new_entry->locked = FALSE;

  g_queue_push_head (&cache_entries, new_entry);
  g_queue_push_head (buffer->cl_cache, new_entry);
}

static gboolean
merge_entry (GeglBufferClCacheEntry *entry)
{
  gpointer data;
  size_t pitch;
  const size_t origin_zero[3] = {0, 0, 0};
  const size_t region[3] = {entry->roi.width, entry->roi.height, 1};
  cl_int cl_err;

  if (entry->mode == GEGL_CL_BUFFER_CACHE_DIRTY)
    {
      entry->locked = TRUE;

      g_printf ("[OpenCL] merge texture %p {%d,%d,%d,%d} tex:%p\n", entry->buffer, entry->roi.x, entry->roi.y, entry->roi.width, entry->roi.height, entry->tex);

      data = gegl_clEnqueueMapImage(gegl_cl_get_command_queue(), entry->tex, CL_TRUE,
                                    CL_MAP_READ,
                                    origin_zero, region, &pitch, NULL,
                                    0, NULL, NULL, &cl_err);
      if (cl_err != CL_SUCCESS) return FALSE;

      /* tile-ize */
      gegl_buffer_set (entry->buffer, &entry->roi, entry->buffer->format, data, pitch);

      cl_err = gegl_clEnqueueUnmapMemObject (gegl_cl_get_command_queue(), entry->tex, data,
                                             0, NULL, NULL);
      if (cl_err != CL_SUCCESS) return FALSE;

      entry->mode = GEGL_CL_BUFFER_CACHE_CLEAN;

      entry->locked = FALSE;
    }

    return TRUE;
}

/* static method */
gboolean
gegl_buffer_cl_cache_dispose (cl_mem tex)
{

  GList *result = g_queue_find_custom (&cache_entries, tex, find_tex);

  g_printf ("[OpenCL] gegl_buffer_cl_cache_dispose tex:%p\n", tex);

  if (result)
    {
      GeglBufferClCacheEntry *entry = g_list_first(result)->data;
      if (entry->locked) /* this texture is being used right now */
        {
          g_warning ("[OpenCL] Error: Trying to Release locked texture");
          return FALSE;
        }

      gboolean ok = merge_entry (entry);
      if (!ok)
        g_warning ("[OpenCL] Error: Releasing unmerged texture");

      gegl_clReleaseMemObject (tex);

      g_queue_remove (&cache_entries, entry);
      g_queue_remove (entry->buffer->cl_cache, entry);

      g_slice_free (GeglBufferClCacheEntry, entry);

      if (!ok)
        return FALSE;

      return TRUE;
    }
  else
    {
      g_warning ("[OpenCL] Tried to dispose texture not present in cache");
      return FALSE;
    }
}

cl_mem
gegl_buffer_cl_cache_request (GeglBuffer            *buffer,
                              cl_mem_flags           flags,
                              cl_image_format       *image_format,
                              const GeglRectangle   *roi,
                              GeglBufferClCacheMode  mode,
                              cl_int                *errcode_ret)
{
  cl_mem tex = NULL;
  cl_int cl_err;

  gboolean allocd = FALSE;

  while (!allocd)
    {
      tex = gegl_clCreateImage2D (gegl_cl_get_context (),
                                   flags,
                                   image_format,
                                   roi->width, roi->height,
                                   0, NULL, &cl_err);

      if      (cl_err == CL_SUCCESS)
        allocd = TRUE;
      else if ((cl_err == CL_OUT_OF_RESOURCES || cl_err == CL_MEM_OBJECT_ALLOCATION_FAILURE)
               && !g_queue_is_empty (&cache_entries))
        {
          GeglBufferClCacheEntry *entry = g_queue_peek_tail (&cache_entries);
          merge_entry (entry);
          gegl_clFinish (gegl_cl_get_command_queue ());
          gegl_buffer_cl_cache_dispose (entry->tex);
        }
      else
        break;
    }

  if (tex)
    gegl_buffer_cl_cache_set (buffer, tex, roi, mode);

  *errcode_ret = cl_err;

  g_printf ("[OpenCL] gegl_buffer_cl_cache_request buffer:%p {%d,%d,%d,%d} mode:%d tex:%p\n", buffer, roi->x, roi->y, roi->width, roi->height, mode, tex);

  return tex;
}

void
gegl_buffer_cl_cache_invalidate (GeglBuffer    *buffer,
                                 const GeglRectangle *roi)
{
  GeglRectangle tmp;
  GList *iter;
  gboolean found = FALSE;

  if (g_queue_is_empty (buffer->cl_cache)) return;

  g_printf ("[OpenCL] gegl_buffer_cl_cache_invalidate %p {%d,%d,%d,%d}\n", buffer, roi->x, roi->y, roi->width, roi->height);

  for (iter=g_queue_peek_head_link (buffer->cl_cache); iter; iter=iter->next)
    {
      GeglBufferClCacheEntry *entry = iter->data;
      if (gegl_rectangle_intersect (&tmp, &entry->roi, roi))
        {
          merge_entry (entry);
          found = TRUE;
        }
    }

  if (found) gegl_clFinish (gegl_cl_get_command_queue ());
}

#define CL_ERROR {g_printf("[OpenCL] Error in %s:%d@%s - %s\n", __FILE__, __LINE__, __func__, gegl_cl_errstring(cl_err)); goto error;}
#define verify_cl_error {if (cl_err != CL_SUCCESS) CL_ERROR;}

gboolean
gegl_buffer_cl_cache_from (GeglBuffer          *buffer,
                           const GeglRectangle *roi,
                           gpointer             dest_buf,
                           const Babl          *format,
                           gint                 rowstride)
{
  GList *iter;

  cl_mem tex_buf = NULL;
  cl_mem tex_aux = NULL;

  if (roi->width >= 256 && roi->height >= 256) /* no point in using the GPU to get small textures */
    for (iter=g_queue_peek_head_link (buffer->cl_cache); iter; iter=iter->next)
      {
        GeglBufferClCacheEntry *entry = iter->data;
        /* why bringing data from the GPU if it's already there */
        if (gegl_rectangle_contains (&entry->roi, roi) && entry->mode == GEGL_CL_BUFFER_CACHE_DIRTY)
          {
            cl_int cl_err;
            const size_t origin[3] = {roi->x - entry->roi.x, roi->y - entry->roi.y, 0};
            const size_t region[3] = {roi->width, roi->height, 1};
            const size_t size[2]   = {roi->width, roi->height};

            const size_t origin_zero[3] = {0, 0, 0};

            gegl_cl_color_op conv = gegl_cl_color_supported (buffer->format, format);

            cl_image_format buf_format;
            cl_image_format cl_format;

            gegl_cl_color_babl (buffer->format, &buf_format, NULL);
            gegl_cl_color_babl (format,         &cl_format,  NULL);

            if (conv == GEGL_CL_COLOR_NOT_SUPPORTED)
              {
                gegl_buffer_cl_cache_invalidate (buffer, roi);
                return FALSE;
              }
            else if (conv == GEGL_CL_COLOR_EQUAL)
              {
                cl_err = gegl_clEnqueueReadImage (gegl_cl_get_command_queue (),
                                                  entry->tex, TRUE,
                                                  origin, region,
                                                  (rowstride == GEGL_AUTO_ROWSTRIDE)? 0 : rowstride, 0,
                                                  dest_buf,
                                                  0, NULL, NULL);
              }
            else if (conv == GEGL_CL_COLOR_CONVERT)
              {
                tex_aux = gegl_buffer_cl_cache_request (buffer,
                                                        CL_MEM_READ_WRITE,
                                                        &cl_format,
                                                        roi,
                                                        GEGL_CL_BUFFER_NO_CACHE,
                                                        &cl_err);
                verify_cl_error;

                if (entry->roi.width == roi->width && entry->roi.height == roi->height)
                  {
                    cl_err = gegl_cl_color_conv (entry->tex, tex_aux, size, buffer->format, format);
                    if (cl_err == FALSE) return FALSE;
                  }
                else
                  {
                    tex_buf = gegl_buffer_cl_cache_request (buffer,
                                                            CL_MEM_READ_WRITE,
                                                            &buf_format,
                                                            roi,
                                                            GEGL_CL_BUFFER_NO_CACHE,
                                                            &cl_err);
                    verify_cl_error;

                    cl_err =  gegl_clEnqueueCopyImage (gegl_cl_get_command_queue (),
                                                       entry->tex,
                                                       tex_buf,
                                                       origin,
                                                       origin_zero,
                                                       region,
                                                       0, NULL, NULL);
                    verify_cl_error;

                    cl_err = gegl_clEnqueueBarrier(gegl_cl_get_command_queue ());
                    verify_cl_error;

                    cl_err = gegl_cl_color_conv (tex_buf, tex_aux, size, buffer->format, format);
                    if (cl_err == FALSE) return FALSE;

                    cl_err = gegl_clEnqueueReadImage (gegl_cl_get_command_queue (),
                                                      tex_aux, TRUE,
                                                      origin_zero, region,
                                                      (rowstride == GEGL_AUTO_ROWSTRIDE)? 0 : rowstride, 0,
                                                      dest_buf,
                                                      0, NULL, NULL);
                    verify_cl_error;
                  }
              }

            if (tex_buf) gegl_buffer_cl_cache_dispose (tex_buf);
            if (tex_aux) gegl_buffer_cl_cache_dispose (tex_aux);

            bump_entry (entry);

            g_printf ("[OpenCL] cache hit! buffer:%p {%d,%d,%d,%d}\n", entry->buffer, roi->x, roi->y, roi->width, roi->height);

            return TRUE;
          }
      }

  /* we have to merge entries that intersect ROI */
  //g_printf ("[OpenCL] cache miss! buffer:%p {%d,%d,%d,%d}\n", buffer, roi->x, roi->y, roi->width, roi->height);
  gegl_buffer_cl_cache_invalidate (buffer, roi);

  return FALSE;

error:
  if (tex_buf) gegl_buffer_cl_cache_dispose (tex_buf);
  if (tex_aux) gegl_buffer_cl_cache_dispose (tex_aux);

  gegl_buffer_cl_cache_invalidate (buffer, roi);

  return FALSE;
}

#undef CL_ERROR
#undef verify_cl_error

void
gegl_buffer_cl_cache_clear (GeglBuffer          *buffer,
                            const GeglRectangle *roi)
{
  GeglRectangle tmp;
  GList *iter;
  gboolean finish = FALSE;

  g_printf ("[OpenCL] gegl_buffer_cl_cache_clear buffer:%p {%d,%d,%d,%d}\n", buffer, roi->x, roi->y, roi->width, roi->height);

  while (!finish)
    {
      finish = TRUE;
      for (iter=g_queue_peek_head_link (buffer->cl_cache); iter; iter=iter->next)
        {
          GeglBufferClCacheEntry *entry = iter->data;
          if (!entry->locked)
            {
              if (gegl_rectangle_contains (roi, &entry->roi))
                {
                  gegl_buffer_cl_cache_dispose (entry->tex);
                  finish = FALSE;
                }
              else if (gegl_rectangle_intersect (&tmp, &entry->roi, roi))
                {
                  merge_entry (entry);
                  gegl_clFinish (gegl_cl_get_command_queue ());

                  gegl_buffer_cl_cache_dispose (entry->tex);
                  finish = FALSE;
                }
            }
        }
    }
}

void
gegl_buffer_cl_cache_remove (GeglBuffer *buffer)
{
  GList *iter;

  g_printf ("[OpenCL] gegl_buffer_cl_cache_remove buffer:%p\n", buffer);

  while (iter = g_queue_peek_head_link (buffer->cl_cache))
    {
      GeglBufferClCacheEntry *entry = iter->data;
      gegl_buffer_cl_cache_dispose (entry->tex);
    }
}
