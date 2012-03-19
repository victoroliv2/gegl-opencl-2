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
 * Copyright 2010 Danny Robson      <danny@blubinc.net>
 * (pfstmo)  2007 Grzegorz Krawczyk <krawczyk@mpi-sb.mpg.de>
 */

#include "config.h"
#include <glib/gi18n-lib.h>
#include <math.h>


#ifdef GEGL_CHANT_PROPERTIES

gegl_chant_double (brightness, _("Brightness"),
                  -100.0, 100.0, 0.0,
                  _("Overall brightness of the image"))
gegl_chant_double (chromatic, _("Chromatic Adaptation"),
                  0.0, 1.0, 0.0,
                  _("Adapation to colour variation across the image"))
gegl_chant_double (light, _("Light Adaptation"),
                  0.0, 1.0, 1.0,
                  _("Adapation to light variation across the image"))


#else

#define GEGL_CHANT_TYPE_FILTER
#define GEGL_CHANT_C_FILE       "reinhard05.c"

#include "gegl-chant.h"


typedef struct {
  gfloat min, max, avg, range;
  guint  num;
} stats;


static const gchar *OUTPUT_FORMAT = "RGBA float";


static void
reinhard05_prepare (GeglOperation *operation)
{
  gegl_operation_set_format (operation, "input",  babl_format (OUTPUT_FORMAT));
  gegl_operation_set_format (operation, "output", babl_format (OUTPUT_FORMAT));
}

static GeglRectangle
reinhard05_get_required_for_output (GeglOperation       *operation,
                                    const gchar         *input_pad,
                                    const GeglRectangle *roi)
{
  GeglRectangle result = *gegl_operation_source_get_bounding_box (operation,
                                                                  "input");
  return result;
}


static GeglRectangle
reinhard05_get_cached_region (GeglOperation       *operation,
                              const GeglRectangle *roi)
{
  return *gegl_operation_source_get_bounding_box (operation, "input");
}

static void
reinhard05_stats_start (stats *s)
{
  g_return_if_fail (s);

  s->min   = G_MAXFLOAT;
  s->max   = G_MINFLOAT;
  s->avg   = 0.0;
  s->range = NAN;
  s->num   = 0;
};


static void
reinhard05_stats_update (stats *s,
                         gfloat value)
{
  g_return_if_fail (s);
  g_return_if_fail (!isinf (value));
  g_return_if_fail (!isnan (value));

  s->min  = MIN (s->min, value);
  s->max  = MAX (s->max, value);
  s->avg += value;
  s->num += 1;
}


static void
reinhard05_stats_finish (stats *s)
{
  g_return_if_fail (s->num !=    0.0);
  g_return_if_fail (s->max >= s->min);

  s->avg   /= s->num;
  s->range  = s->max - s->min;
}

#include "opencl/gegl-cl.h"
#include "buffer/gegl-buffer-cl-iterator.h"

static const char* kernel_source =
"__kernel void reinhard05_1 (__global const float4 * pix,        \n"
"                            __global       float4 * pix_out,    \n"
"                            __global const float  * lum,        \n"
"                            float chrom,                        \n"
"                            float light,                        \n"
"                            float intensity,                    \n"
"                            float contrast,                     \n"
"                            float4 channel_avg,                 \n"
"                            float world_lin_avg)                \n"
"{                                                               \n"
" int gid = get_global_id(0);                                    \n"
" float4 pix_v = pix[gid];                                       \n"
" float  lum_v = lum[gid];                                       \n"
" float3 local_;                                                 \n"
" float3 global_;                                                \n"
" float3 adapt;                                                  \n"
"                                                                \n"
" if (lum_v == 0.0f) return;                                     \n"
"                                                                \n"
" local_  = chrom * pix_v.xyz       + (1.0f - chrom) * lum_v;         \n"
" global_ = chrom * channel_avg.xyz + (1.0f - chrom) * world_lin_avg; \n"
" adapt   = light * local_ + (1.0f - light) * global_;                \n"
" pix_v.xyz /= pix_v.xyz + pow (intensity * adapt, contrast);    \n"
"                                                                \n"
" pix_out[gid] = pix_v;                                          \n"
"}                                                               \n"
"                                                                \n"
"__kernel void reinhard05_2 (__global float4 * src,              \n"
"                            __global float4 * dst,              \n"
"                            float min,                          \n"
"                            float range)                        \n"
"{                                                               \n"
" int gid = get_global_id(0);                                    \n"
" dst[gid] = (src[gid]-min) / range;                             \n"
"}                                                               \n";

static gegl_cl_run_data * cl_data = NULL;

static cl_int
cl_reinhard05_1 (cl_mem               in_tex,
                 cl_mem               lum_tex,
                 cl_mem               out_tex,
                 size_t               global_worksize,
                 const GeglRectangle *roi,
                 gfloat               chrom,
                 gfloat               light,
                 gfloat               intensity,
                 gfloat               contrast,
                 stats                world_lin,
                 stats                channel[])
{
  cl_int cl_err = 0;
  cl_float4 channel_avg = {channel[0].avg, channel[1].avg, channel[2].avg, 1.0f};

  if (!cl_data)
    {
      const char *kernel_name[] = {"reinhard05_1", "reinhard05_2", NULL};
      cl_data = gegl_cl_compile_and_build (kernel_source, kernel_name);
    }

  if (!cl_data) return 1;

  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 0, sizeof(cl_mem),    (void*)&in_tex);
  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 1, sizeof(cl_mem),    (void*)&out_tex);
  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 2, sizeof(cl_mem),    (void*)&lum_tex);

  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 3, sizeof(cl_float),  (void*)&chrom);
  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 4, sizeof(cl_float),  (void*)&light);
  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 5, sizeof(cl_float),  (void*)&intensity);
  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 6, sizeof(cl_float),  (void*)&contrast);

  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 7, sizeof(cl_float4), (void*)&channel_avg);
  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 8, sizeof(cl_float),  (void*)&world_lin.avg);
  if (cl_err != CL_SUCCESS) return cl_err;

  cl_err = gegl_clEnqueueNDRangeKernel(gegl_cl_get_command_queue (),
                                        cl_data->kernel[0], 1,
                                        NULL, &global_worksize, NULL,
                                        0, NULL, NULL);
  if (cl_err != CL_SUCCESS) return cl_err;
  return cl_err;
}

static cl_int
cl_reinhard05_2 (cl_mem               in_tex,
                 cl_mem               out_tex,
                 size_t               global_worksize,
                 const GeglRectangle *roi,
                 gfloat               min,
                 gfloat               range)
{
  cl_int cl_err = 0;

  if (!cl_data)
    {
      const char *kernel_name[] = {"reinhard05_1", "reinhard05_2", NULL};
      cl_data = gegl_cl_compile_and_build (kernel_source, kernel_name);
    }

  if (!cl_data) return 1;

  cl_err |= gegl_clSetKernelArg(cl_data->kernel[1], 0, sizeof(cl_mem),    (void*)&in_tex);
  cl_err |= gegl_clSetKernelArg(cl_data->kernel[1], 1, sizeof(cl_mem),    (void*)&out_tex);
  cl_err |= gegl_clSetKernelArg(cl_data->kernel[1], 2, sizeof(cl_float),  (void*)&min);
  cl_err |= gegl_clSetKernelArg(cl_data->kernel[1], 3, sizeof(cl_float),  (void*)&range);
  if (cl_err != CL_SUCCESS) return cl_err;

  cl_err = gegl_clEnqueueNDRangeKernel(gegl_cl_get_command_queue (),
                                        cl_data->kernel[1], 1,
                                        NULL, &global_worksize, NULL,
                                        0, NULL, NULL);
  if (cl_err != CL_SUCCESS) return cl_err;
  return cl_err;
}



static gboolean
reinhard05_process (GeglOperation       *operation,
                    GeglBuffer          *input,
                    GeglBuffer          *output,
                    const GeglRectangle *result)
{
  const GeglChantO *o = GEGL_CHANT_PROPERTIES (operation);

  const gint  pix_stride = 4, /* RGBA */
              RGB        = 3;

  gfloat *lum,
         *pix;
  gfloat  key, contrast, intensity,
          chrom      =       o->chromatic,
          chrom_comp = 1.0 - o->chromatic,
          light      =       o->light,
          light_comp = 1.0 - o->light;

  stats   world_lin,
          world_log,
          channel [RGB],
          normalise;

  gint    i, c;

  g_return_val_if_fail (operation, FALSE);
  g_return_val_if_fail (input, FALSE);
  g_return_val_if_fail (output, FALSE);
  g_return_val_if_fail (result, FALSE);

  g_return_val_if_fail (babl_format_get_n_components (babl_format (OUTPUT_FORMAT)) == pix_stride, FALSE);

  g_return_val_if_fail (chrom      >= 0.0 && chrom      <= 1.0, FALSE);
  g_return_val_if_fail (chrom_comp >= 0.0 && chrom_comp <= 1.0, FALSE);
  g_return_val_if_fail (light      >= 0.0 && light      <= 1.0, FALSE);
  g_return_val_if_fail (light_comp >= 0.0 && light_comp <= 1.0, FALSE);


  /* Obtain the pixel data */
  lum = g_new (gfloat, result->width * result->height),
  gegl_buffer_get (input, 1.0, result, babl_format ("Y float"),
                   lum, GEGL_AUTO_ROWSTRIDE);

  pix = g_new (gfloat, result->width * result->height * pix_stride);
  gegl_buffer_get (input, 1.0, result, babl_format (OUTPUT_FORMAT),
                   pix, GEGL_AUTO_ROWSTRIDE);

  /* Collect the image stats, averages, etc */
  reinhard05_stats_start (&world_lin);
  reinhard05_stats_start (&world_log);
  reinhard05_stats_start (&normalise);
  for (i = 0; i < RGB; ++i)
    {
      reinhard05_stats_start (channel + i);
    }

  for (i = 0; i < result->width * result->height; ++i)
    {
      reinhard05_stats_update (&world_lin,                 lum[i] );
      reinhard05_stats_update (&world_log, logf (2.3e-5f + lum[i]));

      for (c = 0; c < RGB; ++c)
        {
          reinhard05_stats_update (channel + c, pix[i * pix_stride + c]);
        }
    }

  g_return_val_if_fail (world_lin.min >= 0.0, FALSE);

  reinhard05_stats_finish (&world_lin);
  reinhard05_stats_finish (&world_log);
  for (i = 0; i < RGB; ++i)
    {
      reinhard05_stats_finish (channel + i);
    }

  /* Calculate key parameters */
  key       = (logf (world_lin.max) -                 world_log.avg) /
              (logf (world_lin.max) - logf (2.3e-5f + world_lin.min));
  contrast  = 0.3 + 0.7 * powf (key, 1.4);
  intensity = expf (-o->brightness);

  g_return_val_if_fail (contrast >= 0.3 && contrast <= 1.0, FALSE);

  if (cl_state.is_accelerated)
    {
      const Babl *in_format  = gegl_operation_get_format (operation, "input");
      const Babl *out_format = gegl_operation_get_format (operation, "output");
      const Babl *lum_format = babl_format("Y float");

      GeglBuffer *pix_out = gegl_buffer_new (result, in_format);
      gint j, k;
      cl_int err, cl_err;

      {
      gfloat *pix_map;

      GeglBufferClIterator *i = gegl_buffer_cl_iterator_new (pix_out, result, in_format, GEGL_CL_BUFFER_WRITE);
      gint read = gegl_buffer_cl_iterator_add (i, input, result, in_format,  GEGL_CL_BUFFER_READ);
      gint lum_ = gegl_buffer_cl_iterator_add (i, input, result, lum_format, GEGL_CL_BUFFER_READ);

      while (gegl_buffer_cl_iterator_next (i, &err))
        {
          if (err) return FALSE;
          for (j=0; j < i->n; j++)
            {

              cl_err = cl_reinhard05_1(i->tex[read][j], i->tex[lum_][j], i->tex[0][j], i->size[0][j], &i->roi[0][j],
                                       chrom, light, intensity, contrast,
                                       world_lin, channel);
              if (cl_err != CL_SUCCESS)
                {
                  g_warning("[OpenCL] Error in gegl:reinhard05: %s\n", gegl_cl_errstring(cl_err));
                  return FALSE;
                }

              pix_map = gegl_clEnqueueMapBuffer(gegl_cl_get_command_queue(), i->tex[0][j], CL_TRUE, CL_MAP_READ, 0, i->size[0][j] * babl_format_get_bytes_per_pixel (in_format),
                                                0, NULL, NULL, &cl_err);
              if (CL_SUCCESS != cl_err) return cl_err;

              for (k = 0; k < i->size[0][j]; ++k)
                {
                  if(lum[k] == 0.0f)
                    continue;
                  for(c=0; c < RGB; c++)
                    reinhard05_stats_update (&normalise, pix_map[k*pix_stride + c]);
                }

              cl_err = gegl_clEnqueueUnmapMemObject(gegl_cl_get_command_queue(), i->tex[0][j], pix_map,
                                                    0, NULL, NULL);
              if (CL_SUCCESS != cl_err) return cl_err;

            }
        }
      }

      /* Normalise the pixel values */
      reinhard05_stats_finish (&normalise);
      {
      GeglBufferClIterator *i = gegl_buffer_cl_iterator_new (output, result, out_format, GEGL_CL_BUFFER_WRITE);
      gint read = gegl_buffer_cl_iterator_add (i, pix_out, result, in_format,  GEGL_CL_BUFFER_READ);
      while (gegl_buffer_cl_iterator_next (i, &err))
        {
          if (err) return FALSE;
          for (j=0; j < i->n; j++)
            {

              cl_err = cl_reinhard05_2(i->tex[read][j], i->tex[0][j], i->size[0][j], &i->roi[0][j],
                                       normalise.min, normalise.range);
              if (cl_err != CL_SUCCESS)
                {
                  g_warning("[OpenCL] Error in gegl:reinhard05: %s\n", gegl_cl_errstring(cl_err));
                  return FALSE;

                }

            }
        }
      }

      gegl_buffer_destroy (pix_out);
    }
  else
    {

      /* Apply the operator */
      for (i = 0; i < result->width * result->height; ++i)
        {
          gfloat local, global, adapt;

          if (lum[i] == 0.0)
            continue;

          for (c = 0; c < RGB; ++c)
            {
              gfloat *_p = pix + i * pix_stride + c,
                       p = *_p;

              local  = chrom      * p +
                       chrom_comp * lum[i];
              global = chrom      * channel[c].avg +
                       chrom_comp * world_lin.avg;
              adapt  = light      * local +
                       light_comp * global;

              p  /= p + powf (intensity * adapt, contrast);
              *_p = p;
              reinhard05_stats_update (&normalise, p);
            }
        }

      /* Normalise the pixel values */
      reinhard05_stats_finish (&normalise);

      for (i = 0; i < result->width * result->height; ++i)
        {
           for (c = 0; c < pix_stride; ++c)
            {
              gfloat *p = pix + i * pix_stride + c;
              *p        = (*p - normalise.min) / normalise.range;
            }
        }

      /* Cleanup and set the output */
      gegl_buffer_set (output, result, babl_format (OUTPUT_FORMAT), pix,
                       GEGL_AUTO_ROWSTRIDE);
    }

  g_free (pix);
  g_free (lum);

  return TRUE;
}


/**/
static void
gegl_chant_class_init (GeglChantClass *klass)
{
  GeglOperationClass       *operation_class;
  GeglOperationFilterClass *filter_class;

  operation_class = GEGL_OPERATION_CLASS (klass);
  filter_class    = GEGL_OPERATION_FILTER_CLASS (klass);

  filter_class->process = reinhard05_process;
  operation_class->opencl_support = TRUE;

  operation_class->prepare                 = reinhard05_prepare;
  operation_class->get_required_for_output = reinhard05_get_required_for_output;
  operation_class->get_cached_region       = reinhard05_get_cached_region;

  operation_class->name        = "gegl:reinhard05";
  operation_class->categories  = "tonemapping";
  operation_class->description =
        _("Adapt an image, which may have a high dynamic range, for "
    "presentation using a low dynamic range. This is an efficient "
          "global operator derived from simple physiological observations, "
          "producing luminance within the range 0.0-1.0");
}

#endif


