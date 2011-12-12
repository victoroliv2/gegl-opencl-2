/* This file is part of GEGL
 *
 * GEGL is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * GEGL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GEGL; if not, see <http://www.gnu.org/licenses/>.
 *
 * Copyright 2006 Øyvind Kolås
 */


#include "config.h"

#include <glib-object.h>

#include "gegl.h"
#include "gegl-types-internal.h"
#include "gegl-operation-point-filter.h"
#include "graph/gegl-pad.h"
#include "graph/gegl-node.h"
#include "gegl-utils.h"
#include <string.h>

#include "gegl-buffer-private.h"
#include "gegl-tile-storage.h"

#include "buffer/gegl-buffer-iterator.h"
#include "opencl/gegl-cl.h"

static gboolean gegl_operation_point_filter_process
                              (GeglOperation       *operation,
                               GeglBuffer          *input,
                               GeglBuffer          *output,
                               const GeglRectangle *result);

static gboolean gegl_operation_point_filter_op_process
                              (GeglOperation       *operation,
                               GeglOperationContext *context,
                               const gchar          *output_pad,
                               const GeglRectangle  *roi);

G_DEFINE_TYPE (GeglOperationPointFilter, gegl_operation_point_filter, GEGL_TYPE_OPERATION_FILTER)

static void prepare (GeglOperation *operation)
{
  gegl_operation_set_format (operation, "input", babl_format ("RGBA float"));
  gegl_operation_set_format (operation, "output", babl_format ("RGBA float"));
}

static void
gegl_operation_point_filter_class_init (GeglOperationPointFilterClass *klass)
{
  GeglOperationClass *operation_class = GEGL_OPERATION_CLASS (klass);

  operation_class->process = gegl_operation_point_filter_op_process;
  operation_class->prepare = prepare;
  operation_class->no_cache = TRUE;

  klass->process = NULL;
  klass->cl_process = NULL;
}

static void
gegl_operation_point_filter_init (GeglOperationPointFilter *self)
{
}

struct buf_tex
{
  GeglBuffer *buf;
  cl_mem *tex;
};

//#define CL_ERROR {g_assert(0);}
#define CL_ERROR {g_printf("[OpenCL] Error in %s:%d@%s - %s\n", __FILE__, __LINE__, __func__, gegl_cl_errstring(errcode)); goto error;}

static gboolean
gegl_operation_point_filter_cl_process_full (GeglOperation       *operation,
                                             GeglBuffer          *input,
                                             GeglBuffer          *output,
                                             const GeglRectangle *result_)
{
  const Babl *in_format  = gegl_operation_get_format (operation, "input");
  const Babl *out_format = gegl_operation_get_format (operation, "output");

  GeglOperationPointFilterClass *point_filter_class = GEGL_OPERATION_POINT_FILTER_GET_CLASS (operation);

  GeglBufferTileIterator in_tile_iterator;
  GeglBufferTileIterator out_tile_iterator;

  GeglRectangle result = *result_;

  int i;
  int errcode;

  int ntex = 0;
  struct buf_tex input_tex;
  struct buf_tex output_tex;

  gboolean in_iter = TRUE, out_iter = TRUE;

  /* XXX: this is only working for color conversion in the GPU */

  cl_image_format format;
  format.image_channel_order = CL_RGBA;
  format.image_channel_data_type = CL_FLOAT;

  ntex = ((result.width  / input->tile_storage->tile_width)  + 1) *
         ((result.height / input->tile_storage->tile_height) + 1);

  g_printf("[OpenCL] BABL formats: (%s,%s:%d) (%s,%s:%d)\n \t Tile Size:(%d, %d)\n", babl_get_name(gegl_buffer_get_format(input)),  babl_get_name(in_format),
                                                             gegl_cl_color_supported (gegl_buffer_get_format(input), in_format),
                                                             babl_get_name(out_format), babl_get_name(gegl_buffer_get_format(output)),
                                                             gegl_cl_color_supported (out_format, gegl_buffer_get_format(output)),
                                                             input->tile_storage->tile_width,
                                                             input->tile_storage->tile_height);

  input_tex.tex  = (cl_mem *) gegl_malloc(ntex * sizeof(cl_mem));
  output_tex.tex = (cl_mem *) gegl_malloc(ntex * sizeof(cl_mem));

  if (input_tex.tex == NULL || output_tex.tex == NULL)
    CL_ERROR;

  for (i=0; i<ntex; i++)
    {
      input_tex.tex [i] = NULL;
      output_tex.tex[i] = NULL;
    }

  gegl_buffer_tile_iterator_init (&in_tile_iterator,  input,  result, FALSE);
  in_iter  = gegl_buffer_tile_iterator_next (&in_tile_iterator);

  i = 0;
  while (in_iter)
    {
      input_tex.tex[i]  = gegl_clCreateImage2D (gegl_cl_get_context(),
                                                CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, &format,
                                                in_tile_iterator.subrect.width, in_tile_iterator.subrect.height,
                                                input->tile_storage->tile_width * sizeof (cl_float4), in_tile_iterator.sub_data, &errcode);
      if (errcode != CL_SUCCESS) CL_ERROR;

      output_tex.tex[i]  = gegl_clCreateImage2D (gegl_cl_get_context(),
                                                CL_MEM_READ_WRITE, &format,
                                                in_tile_iterator.subrect.width, in_tile_iterator.subrect.height,
                                                0,  NULL, &errcode);
      if (errcode != CL_SUCCESS) CL_ERROR;

      i++;
      in_iter  = gegl_buffer_tile_iterator_next (&in_tile_iterator);
    }

  errcode = gegl_clEnqueueBarrier(gegl_cl_get_command_queue());
  if (errcode != CL_SUCCESS) CL_ERROR;

  /* color conversion in the GPU (input) */
  gegl_buffer_tile_iterator_init (&in_tile_iterator,  input, result, FALSE);
  in_iter  = gegl_buffer_tile_iterator_next (&in_tile_iterator);

  i = 0;
  if (gegl_cl_color_supported (gegl_buffer_get_format(input), in_format))
    while (in_iter)
      {
        if (!input_tex.tex[i]) continue;

        cl_mem swap;
        const size_t size[2] = {in_tile_iterator.subrect.width, in_tile_iterator.subrect.height};
        errcode = gegl_cl_color_conv (input_tex.tex[i], output_tex.tex[i], size, gegl_buffer_get_format(input), in_format);

        if (errcode == FALSE) CL_ERROR;

        swap = input_tex.tex[i];
        input_tex.tex[i]  = output_tex.tex[i];
        output_tex.tex[i] = swap;

        i++;
        in_iter  = gegl_buffer_tile_iterator_next (&in_tile_iterator);
      }

  /* Process */
  gegl_buffer_tile_iterator_init (&in_tile_iterator,  input,  result, FALSE);
  in_iter  = gegl_buffer_tile_iterator_next (&in_tile_iterator);

  i = 0;
  while (in_iter)
    {
      if (!input_tex.tex[i]) continue;

      const size_t size[2] = {in_tile_iterator.subrect.width, in_tile_iterator.subrect.height};

      errcode = point_filter_class->cl_process(operation, input_tex.tex[i], output_tex.tex[i], size, &in_tile_iterator.subrect);
      if (errcode != CL_SUCCESS) CL_ERROR;

      i++;
      in_iter  = gegl_buffer_tile_iterator_next (&in_tile_iterator);
    }

  /* Wait Processing */
  errcode = gegl_clEnqueueBarrier(gegl_cl_get_command_queue());
  if (errcode != CL_SUCCESS) CL_ERROR;

  /* color conversion in the GPU (output) */
  gegl_buffer_tile_iterator_init (&out_tile_iterator, output, result, FALSE);
  out_iter = gegl_buffer_tile_iterator_next (&out_tile_iterator);

  i = 0;
  if (gegl_cl_color_supported (out_format, gegl_buffer_get_format(output)))
    while (out_iter)
      {
        if (!output_tex.tex[i]) continue;

        cl_mem swap;
        const size_t size[2] = {out_tile_iterator.subrect.width, out_tile_iterator.subrect.height};
        errcode = gegl_cl_color_conv (output_tex.tex[i], input_tex.tex[i], size, out_format, gegl_buffer_get_format(output));

        if (errcode == FALSE) CL_ERROR;

        swap = input_tex.tex[i];
        input_tex.tex[i]  = output_tex.tex[i];
        output_tex.tex[i] = swap;

        i++;
        out_iter  = gegl_buffer_tile_iterator_next (&out_tile_iterator);
      }

  /* GPU -> CPU */
  gegl_buffer_tile_iterator_init (&out_tile_iterator, output, result, FALSE); /* XXX: we are writing here */
  out_iter = gegl_buffer_tile_iterator_next (&out_tile_iterator);

  i = 0;
  while (out_iter)
    {
      if (!output_tex.tex[i]) continue;

      const size_t origin[3] = {0, 0, 0};
      const size_t region[3] = {out_tile_iterator.subrect.width, out_tile_iterator.subrect.height, 1};

      errcode = gegl_clEnqueueReadImage(gegl_cl_get_command_queue(), output_tex.tex[i], CL_FALSE,
                                         origin, region, output->tile_storage->tile_width * sizeof (cl_float4), 0, out_tile_iterator.sub_data,
                                         0, NULL, NULL);
      if (errcode != CL_SUCCESS) CL_ERROR;

      i++;
      out_iter  = gegl_buffer_tile_iterator_next (&out_tile_iterator);
    }

  /* Wait */
  errcode = gegl_clEnqueueBarrier(gegl_cl_get_command_queue());
  if (errcode != CL_SUCCESS) CL_ERROR;

  /* Run! */
  errcode = gegl_clFinish(gegl_cl_get_command_queue());
  if (errcode != CL_SUCCESS) CL_ERROR;

  for (i=0; i < ntex; i++)
    {
      if (input_tex.tex[i])  gegl_clReleaseMemObject (input_tex.tex[i]);
      if (output_tex.tex[i]) gegl_clReleaseMemObject (output_tex.tex[i]);
    }

  gegl_free(input_tex.tex);
  gegl_free(output_tex.tex);

  return TRUE;

error:

    for (i=0; i < ntex; i++)
      {
        if (input_tex.tex[i])  gegl_clReleaseMemObject (input_tex.tex[i]);
        if (output_tex.tex[i]) gegl_clReleaseMemObject (output_tex.tex[i]);
      }

  if (input_tex.tex)     gegl_free(input_tex.tex);
  if (output_tex.tex)    gegl_free(output_tex.tex);

  return FALSE;
}

#undef CL_ERROR

static gboolean
gegl_operation_point_filter_process (GeglOperation       *operation,
                                     GeglBuffer          *input,
                                     GeglBuffer          *output,
                                     const GeglRectangle *result)
{
  const Babl *in_format  = gegl_operation_get_format (operation, "input");
  const Babl *out_format = gegl_operation_get_format (operation, "output");
  GeglOperationPointFilterClass *point_filter_class;

  point_filter_class = GEGL_OPERATION_POINT_FILTER_GET_CLASS (operation);

  if ((result->width > 0) && (result->height > 0))
    {
      if (cl_state.is_accelerated && point_filter_class->cl_process)
        {
          if (gegl_operation_point_filter_cl_process_full (operation, input, output, result))
            return TRUE;
        }

      {
        GeglBufferIterator *i = gegl_buffer_iterator_new (output, result, out_format, GEGL_BUFFER_WRITE);
        gint read = /*output == input ? 0 :*/ gegl_buffer_iterator_add (i, input,  result, in_format, GEGL_BUFFER_READ);
        /* using separate read and write iterators for in-place ideally a single
         * readwrite indice would be sufficient
         */
          while (gegl_buffer_iterator_next (i))
            point_filter_class->process (operation, i->data[read], i->data[0], i->length, &i->roi[0]);
      }
    }
  return TRUE;
}

gboolean gegl_can_do_inplace_processing (GeglOperation       *operation,
                                         GeglBuffer          *input,
                                         const GeglRectangle *result);

gboolean gegl_can_do_inplace_processing (GeglOperation       *operation,
                                         GeglBuffer          *input,
                                         const GeglRectangle *result)
{
  if (!input ||
      GEGL_IS_CACHE (input))
    return FALSE;
  if (gegl_object_get_has_forked (input))
    return FALSE;

  if (input->format == gegl_operation_get_format (operation, "output") &&
      gegl_rectangle_contains (gegl_buffer_get_extent (input), result))
    return TRUE;
  return FALSE;
}


static gboolean gegl_operation_point_filter_op_process
                              (GeglOperation       *operation,
                               GeglOperationContext *context,
                               const gchar          *output_pad,
                               const GeglRectangle  *roi)
{
  GeglBuffer               *input;
  GeglBuffer               *output;
  gboolean                  success = FALSE;

  input = gegl_operation_context_get_source (context, "input");

  if (gegl_can_do_inplace_processing (operation, input, roi))
    {
      output = g_object_ref (input);
      gegl_operation_context_take_object (context, "output", G_OBJECT (output));
    }
  else
    {
      output = gegl_operation_context_get_target (context, "output");
    }

  success = gegl_operation_point_filter_process (operation, input, output, roi);
  if (output == GEGL_BUFFER (operation->node->cache))
    gegl_cache_computed (operation->node->cache, roi);

  if (input != NULL)
    g_object_unref (input);
  return success;
}
