/* This file is an image processing operation for GEGL
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
 * Copyright 2006 Øyvind Kolås <pippin@gimp.org>
 */

#include "config.h"
#include <glib/gi18n-lib.h>


#ifdef GEGL_CHANT_PROPERTIES

#else

#define GEGL_CHANT_TYPE_FILTER
#define GEGL_CHANT_C_FILE       "stretch-contrast.c"

#include "gegl-chant.h"

static gboolean
inner_process (gdouble  min,
               gdouble  max,
               gfloat  *buf,
               gint     n_pixels)
{
  gint o;

  for (o=0; o<n_pixels; o++)
    {
      buf[0] = (buf[0] - min) / (max-min);
      buf[1] = (buf[1] - min) / (max-min);
      buf[2] = (buf[2] - min) / (max-min);
      /* FIXME: really stretch the alpha channel?? */
      buf[3] = (buf[3] - min) / (max-min);

      buf += 4;
    }
  return TRUE;
}

static void
buffer_get_min_max (GeglBuffer *buffer,
                    gdouble    *min,
                    gdouble    *max)
{
  gfloat tmin = 9000000.0;
  gfloat tmax =-9000000.0;

  gfloat *buf = g_new0 (gfloat, 4 * gegl_buffer_get_pixel_count (buffer));
  gint i;
  gegl_buffer_get (buffer, 1.0, NULL, babl_format ("RGBA float"), buf, GEGL_AUTO_ROWSTRIDE);
  for (i=0;i< gegl_buffer_get_pixel_count (buffer);i++)
    {
      gint component;
      for (component=0; component<3; component++)
        {
          gfloat val = buf[i*4+component];

          if (val<tmin)
            tmin=val;
          if (val>tmax)
            tmax=val;
        }
    }
  g_free (buf);
  if (min)
    *min = tmin;
  if (max)
    *max = tmax;
}

static void prepare (GeglOperation *operation)
{
  gegl_operation_set_format (operation, "input", babl_format ("RGBA float"));
  gegl_operation_set_format (operation, "output", babl_format ("RGBA float"));
}

static GeglRectangle
get_required_for_output (GeglOperation        *operation,
                         const gchar         *input_pad,
                         const GeglRectangle *roi)
{
  GeglRectangle result = *gegl_operation_source_get_bounding_box (operation, "input");
  return result;
}

#include "opencl/gegl-cl.h"
#include "buffer/gegl-buffer-cl-iterator.h"

static const char* kernel_source =
"__kernel void kernel_StretchContrast(__global float4 * in,     \n"
"                                     __global float4 * out,    \n"
"                                     float           min,      \n"
"                                     float           max)      \n"     
"{                                                              \n"
"  int gid = get_global_id(0);                                  \n"
"  float4 in_v = in[gid];                                       \n"
"  out[gid] = ( in_v - min ) / ( max - min );                   \n"
"}                                                              \n";

static gegl_cl_run_data * cl_data = NULL;

static cl_int
cl_stretch_contrast (cl_mem                in_tex,
                     cl_mem                out_tex,
                     size_t                global_worksize,
                     const GeglRectangle  *roi,
                     gdouble               min,
                     gdouble               max)
{  
  int i = 0 , stride = 16;/*RGBA float*/
  size_t size = roi->width * roi->height * stride;
  cl_int cl_err = 0;   

  if (!cl_data)
  {
    const char *kernel_name[] ={"kernel_StretchContrast", NULL};
    cl_data = gegl_cl_compile_and_build(kernel_source, kernel_name);
  }
  if (!cl_data)  return 0;

  cl_float cl_min = (cl_float)min;
  cl_float cl_max = (cl_float)max;

  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 0, sizeof(cl_mem), (void*)&in_tex);  
  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 1, sizeof(cl_mem), (void*)&out_tex);
  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 2, sizeof(cl_float), (void*)&cl_min);
  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 3, sizeof(cl_float), (void*)&cl_max);
  if (cl_err != CL_SUCCESS) return cl_err;

  cl_err = gegl_clEnqueueNDRangeKernel(
    gegl_cl_get_command_queue(), cl_data->kernel[0],
    1, NULL,
    &global_worksize, NULL,
    0, NULL, NULL);

  cl_err = gegl_clEnqueueBarrier(gegl_cl_get_command_queue());
  if (CL_SUCCESS != cl_err) return cl_err;

  return cl_err;
}


static gboolean
cl_process (GeglOperation       *operation,
      GeglBuffer          *input,
      GeglBuffer          *output,
      const GeglRectangle *result,
      gdouble              min,
      gdouble              max)
{
  const Babl *in_format  = gegl_operation_get_format (operation, "input");
  const Babl *out_format = gegl_operation_get_format (operation, "output");
  gint err;
  gint j;
  cl_int cl_err;

  GeglBufferClIterator *i = gegl_buffer_cl_iterator_new (output,result, out_format, GEGL_CL_BUFFER_WRITE);
  gint read = gegl_buffer_cl_iterator_add (i, input, result, in_format,  GEGL_CL_BUFFER_READ);

  while (gegl_buffer_cl_iterator_next (i, &err))
  {
    if (err) return FALSE;
    for (j=0; j < i->n; j++)

    {
      cl_err=cl_stretch_contrast(i->tex[read][j],i->tex[0][j],i->size[0][j],&i->roi[0][j],min,max);
      if (cl_err != CL_SUCCESS)
      {
        g_warning("[OpenCL] Error in %s [GeglOperationFilter:Edge-sobel] Kernel\n");
        return FALSE;
      }
    }
  }  
  return TRUE;
}


static gboolean
process (GeglOperation       *operation,
         GeglBuffer          *input,
         GeglBuffer          *output,
         const GeglRectangle *result)
{
  gdouble  min, max;

  buffer_get_min_max (input, &min, &max);

  if (cl_state.is_accelerated)
    if(cl_process(operation, input, output, result, min, max))
      return TRUE;

  {
    gint row;
    gfloat *buf;
    gint chunk_size=128;
    gint consumed=0;

    buf = g_new0 (gfloat, 4 * result->width  * chunk_size);

    for (row = 0; row < result->height; row = consumed)
      {
        gint chunk = consumed+chunk_size<result->height?chunk_size:result->height-consumed;
        GeglRectangle line;

        line.x = result->x;
        line.y = result->y + row;
        line.width = result->width;
        line.height = chunk;

        gegl_buffer_get (input, 1.0, &line, babl_format ("RGBA float"), buf, GEGL_AUTO_ROWSTRIDE);
        inner_process (min, max, buf, result->width  * chunk);
        gegl_buffer_set (output, &line, babl_format ("RGBA float"), buf,
                         GEGL_AUTO_ROWSTRIDE);
        consumed+=chunk;
      }
    g_free (buf);
  }

  return TRUE;
}

/* This is called at the end of the gobject class_init function.
 *
 * Here we override the standard passthrough options for the rect
 * computations.
 */
static void
gegl_chant_class_init (GeglChantClass *klass)
{
  GeglOperationClass       *operation_class;
  GeglOperationFilterClass *filter_class;

  operation_class = GEGL_OPERATION_CLASS (klass);
  filter_class    = GEGL_OPERATION_FILTER_CLASS (klass);

  filter_class->process = process;
  operation_class->prepare = prepare;
  operation_class->opencl_support = TRUE;
  operation_class->get_required_for_output = get_required_for_output;

  operation_class->name        = "gegl:stretch-contrast";
  operation_class->categories  = "color:enhance";
  operation_class->description =
        _("Scales the components of the buffer to be in the 0.0-1.0 range. "
          "This improves images that make poor use of the available contrast "
          "(little contrast, very dark, or very bright images).");
}

#endif
