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

  GeglBufferTileIterator in_iter;
  GeglBufferTileIterator out_iter;

  GeglRectangle result = *result_;

  int i;
  int errcode;

  int ntex = 0;
  struct buf_tex input_tex;
  struct buf_tex output_tex;

  /* supported babl formats up to now:
     RGBA u8
     All formats with four floating-point channels
     (I suppose others formats would be hard to put on GPU)
  */

  cl_image_format rgbaf_format;
  cl_image_format rgbau8_format;

  rgbaf_format.image_channel_order      = CL_RGBA;
  rgbaf_format.image_channel_data_type  = CL_FLOAT;

  rgbau8_format.image_channel_order     = CL_RGBA;
  rgbau8_format.image_channel_data_type = CL_UNORM_INT8;

  g_printf("[OpenCL] BABL formats: (%s,%s:%d) (%s,%s:%d)\n \t Tile Size:(%d, %d)\n", babl_get_name(input->format),  babl_get_name(in_format),
                                                             gegl_cl_color_supported (input->format, in_format),
                                                             babl_get_name(out_format), babl_get_name(output->format),
                                                             gegl_cl_color_supported (out_format, output->format),
                                                             input->tile_storage->tile_width,
                                                             input->tile_storage->tile_height);

  ntex = 0;
  gegl_buffer_tile_iterator_init (&in_iter,  input, result, FALSE);
  while (gegl_buffer_tile_iterator_next (&in_iter))
    ntex++;

  input_tex.tex  = (cl_mem *) gegl_malloc(ntex * sizeof(cl_mem));
  output_tex.tex = (cl_mem *) gegl_malloc(ntex * sizeof(cl_mem));

  if (input_tex.tex == NULL || output_tex.tex == NULL)
    CL_ERROR;

  i = 0;
  gegl_buffer_tile_iterator_init (&in_iter,  input,  result, FALSE);
  while (gegl_buffer_tile_iterator_next (&in_iter))
    {
      input_tex.tex[i]  = gegl_clCreateImage2D (gegl_cl_get_context(),
                                                CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE,
                                                (input->format == babl_format ("RGBA u8"))? &rgbau8_format : &rgbaf_format,
                                                in_iter.subrect.width, in_iter.subrect.height,
                                                (input->format == babl_format ("RGBA u8"))? input->tile_storage->tile_width * sizeof (cl_uchar4) :
                                                                                            input->tile_storage->tile_width * sizeof (cl_float4),
                                                in_iter.sub_data, &errcode);
      if (errcode != CL_SUCCESS) CL_ERROR;

      output_tex.tex[i]  = gegl_clCreateImage2D (gegl_cl_get_context(),
                                                 CL_MEM_READ_WRITE,
                                                 (output->format == babl_format ("RGBA u8"))? &rgbau8_format : &rgbaf_format,
                                                 in_iter.subrect.width, in_iter.subrect.height,
                                                 0, NULL, &errcode);
      if (errcode != CL_SUCCESS) CL_ERROR;

      i++;
    }

  errcode = gegl_clEnqueueBarrier(gegl_cl_get_command_queue());
  if (errcode != CL_SUCCESS) CL_ERROR;

  /* color conversion in the GPU (input) */

  if (gegl_cl_color_supported (input->format, in_format) == CL_COLOR_CONVERT)
    {
      i = 0;
      gegl_buffer_tile_iterator_init (&in_iter,  input, result, FALSE);
      while(gegl_buffer_tile_iterator_next (&in_iter))
        {
          const size_t size[2] = {in_iter.subrect.width, in_iter.subrect.height};
          errcode = gegl_cl_color_conv (&input_tex.tex[i], &output_tex.tex[i], size, input->format, in_format);

          if (errcode == FALSE) CL_ERROR;

          i++;
        }
    }

  /* Process */

  i = 0;
  gegl_buffer_tile_iterator_init (&in_iter,  input,  result, FALSE);
  while (gegl_buffer_tile_iterator_next (&in_iter))
    {
      const size_t size[2] = {in_iter.subrect.width, in_iter.subrect.height};

      errcode = point_filter_class->cl_process(operation, input_tex.tex[i], output_tex.tex[i], size, &in_iter.subrect);
      if (errcode != CL_SUCCESS) CL_ERROR;

      i++;
    }

  /* Wait Processing */
  errcode = gegl_clEnqueueBarrier(gegl_cl_get_command_queue());
  if (errcode != CL_SUCCESS) CL_ERROR;

  /* color conversion in the GPU (output) */

  if (gegl_cl_color_supported (out_format, output->format) == CL_COLOR_CONVERT)
    {
      i = 0;
      gegl_buffer_tile_iterator_init (&out_iter, output, result, FALSE);
      while (gegl_buffer_tile_iterator_next (&out_iter))
        {
          const size_t size[2] = {out_iter.subrect.width, out_iter.subrect.height};
          errcode = gegl_cl_color_conv (&output_tex.tex[i], &input_tex.tex[i], size, out_format, output->format);

          if (errcode == FALSE) CL_ERROR;

          i++;
        }
    }

  /* GPU -> CPU */

  i = 0;
  gegl_buffer_tile_iterator_init (&out_iter, output, result, FALSE); /* XXX: we are writing here */
  while (gegl_buffer_tile_iterator_next (&out_iter))
    {
      const size_t origin[3] = {0, 0, 0};
      const size_t region[3] = {out_iter.subrect.width, out_iter.subrect.height, 1};

      errcode = gegl_clEnqueueReadImage(gegl_cl_get_command_queue(), output_tex.tex[i], CL_FALSE,
                                         origin, region,
                                         (output->format == babl_format ("RGBA u8"))? output->tile_storage->tile_width * sizeof (cl_uchar4) :
                                                                                      output->tile_storage->tile_width * sizeof (cl_float4),
                                         0, out_iter.sub_data,
                                         0, NULL, NULL);
      if (errcode != CL_SUCCESS) CL_ERROR;

      i++;
    }

  /* Wait */
  errcode = gegl_clEnqueueBarrier(gegl_cl_get_command_queue());
  if (errcode != CL_SUCCESS) CL_ERROR;

  /* Run! */
  errcode = gegl_clFinish(gegl_cl_get_command_queue());
  if (errcode != CL_SUCCESS) CL_ERROR;

  for (i=0; i < ntex; i++)
    {
      gegl_clReleaseMemObject (input_tex.tex[i]);
      gegl_clReleaseMemObject (output_tex.tex[i]);
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
