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
#include <math.h>

#ifdef GEGL_CHANT_PROPERTIES

gegl_chant_int (xsize, _("Block Width"), 1, 256, 8,
   _("Width of blocks in pixels"))
gegl_chant_int (ysize, _("Block Height"), 1, 256, 8,
   _("Height of blocks in pixels"))

#else

#define GEGL_CHANT_TYPE_AREA_FILTER
#define GEGL_CHANT_C_FILE       "pixelise.c"

#include "gegl-chant.h"

#define CELL_X(px, cell_width)  ((px) / (cell_width))
#define CELL_Y(py, cell_height)  ((py) / (cell_height))



static void
calc_block_colors (gfloat* block_colors,
                   const gfloat* input,
                   const GeglRectangle* roi,
                   gint xsize,
                   gint ysize)
{
  gint cx0 = CELL_X(roi->x, xsize);
  gint cy0 = CELL_Y(roi->y, ysize);
  gint cx1 = CELL_X(roi->x + roi->width - 1, xsize);
  gint cy1 = CELL_Y(roi->y + roi->height - 1, ysize);

  gint cx;
  gint cy;
  gfloat weight = 1.0f / (xsize * ysize);
  gint line_width = roi->width + 2*xsize;
  /* loop over the blocks within the region of interest */
  for (cy=cy0; cy<=cy1; ++cy)
    {
      for (cx=cx0; cx<=cx1; ++cx)
        {
          gint px = (cx * xsize) - roi->x + xsize;
          gint py = (cy * ysize) - roi->y + ysize;

          /* calculate the average color for this block */
          gint j,i,c;
          gfloat col[4] = {0.0f, 0.0f, 0.0f, 0.0f};
          for (j=py; j<py+ysize; ++j)
            {
              for (i=px; i<px+xsize; ++i)
                {
                  for (c=0; c<4; ++c)
                    col[c] += input[(j*line_width + i)*4 + c];
                }
            }
          for (c=0; c<4; ++c)
            block_colors[c] = weight * col[c];
          block_colors += 4;
        }
    }
}

static void
pixelise (gfloat* buf,
          const GeglRectangle* roi,
          gint xsize,
          gint ysize)
{
  gint cx0 = CELL_X(roi->x, xsize);
  gint cy0 = CELL_Y(roi->y, ysize);
  gint block_count_x = CELL_X(roi->x + roi->width - 1, xsize) - cx0 + 1;
  gint block_count_y = CELL_Y(roi->y + roi->height - 1, ysize) - cy0 + 1;
  gfloat* block_colors = g_new0 (gfloat, block_count_x * block_count_y * 4);
  gint x;
  gint y;
  gint c;

  /* calculate the average color of all the blocks */
  calc_block_colors(block_colors, buf, roi, xsize, ysize);

  /* set each pixel to the average color of the block it belongs to */
  for (y=0; y<roi->height; ++y)
    {
      gint cy = CELL_Y(y + roi->y, ysize) - cy0;
      for (x=0; x<roi->width; ++x)
        {
          gint cx = CELL_X(x + roi->x, xsize) - cx0;
          for (c=0; c<4; ++c)
            *buf++ = block_colors[(cy*block_count_x + cx)*4 + c];
        }
    }

  g_free (block_colors);
}
static void prepare (GeglOperation *operation)
{
  GeglChantO              *o;
  GeglOperationAreaFilter *op_area;

  op_area = GEGL_OPERATION_AREA_FILTER (operation);
  o       = GEGL_CHANT_PROPERTIES (operation);

  op_area->left   =
    op_area->right  = o->xsize;
  op_area->top    =
    op_area->bottom = o->ysize;

  gegl_operation_set_format (operation, "input",  babl_format ("RaGaBaA float"));
  gegl_operation_set_format (operation, "output", babl_format ("RaGaBaA float"));
}

#include "opencl/gegl-cl.h"
#include "buffer/gegl-buffer-cl-iterator.h"

static const char* kernel_source =
"__kernel void calc_block_color(__global float4 *in,                   \n"
"                             __global float4 *out,                    \n"
"                             int xsize,                               \n"
"                             int ysize,                               \n"
"                             int roi_x,                               \n"
"                             int roi_y,                               \n"
"                             int line_width,                          \n"
"                             int block_count_x )                      \n"
"{                                                                     \n"
"    int gidx = get_global_id(0);                                      \n"
"    int gidy = get_global_id(1);                                      \n"
"    int cx = roi_x / xsize + gidx;                                    \n"
"    int cy = roi_y / ysize + gidy;                                    \n"
"                                                                      \n"
"    float weight   = 1.0f / (xsize * ysize);                          \n"
"                                                                      \n"
"    int px = cx * xsize + xsize - roi_x;                              \n"
"    int py = cy * ysize + ysize - roi_y;                              \n"
"                                                                      \n"
"    int i,j;                                                          \n"
"    float4 col = 0.0f;                                                \n"
"    for (j = py;j < py + ysize; ++j)                                  \n"
"    {                                                                 \n"
"        for (i = px;i < px + xsize; ++i)                              \n"
"        {                                                             \n"
"            col += in[j * line_width + i];                            \n"
"        }                                                             \n"
"    }                                                                 \n"
"    out[gidy * block_count_x + gidx] = col * weight;                  \n"
"                                                                      \n"
"}                                                                     \n"
"                                                                      \n"
"__kernel void kernel_pixelise (__global float4 *in,                   \n"
"                             __global float4 *out,                    \n"
"                             int xsize,                               \n"
"                             int ysize,                               \n"
"                             int roi_x,                               \n"
"                             int roi_y,                               \n"
"                             int block_count_x)                       \n"
"{                                                                     \n"
"    int gidx = get_global_id(0);                                      \n"
"    int gidy = get_global_id(1);                                      \n"
"                                                                      \n"
"    int src_width  = get_global_size(0);                              \n"
"    int cx = (gidx + roi_x) / xsize - roi_x / xsize;                  \n"
"    int cy = (gidy + roi_y) / ysize - roi_y / ysize;                  \n"
"    out[gidx + gidy * src_width] = in[cx + cy * block_count_x];       \n"
"}                                                                     \n";

static gegl_cl_run_data *cl_data = NULL;

static cl_int
cl_pixelise (cl_mem                in_tex,
             cl_mem                aux_tex,
             cl_mem                out_tex,
             const GeglRectangle  *src_rect,
             const GeglRectangle  *roi,
             gint                  xsize,
             gint                  ysize)
{
  cl_int cl_err = 0;
  const size_t gbl_size[2]= {roi->width, roi->height};

  if (!cl_data)
  {
    const char *kernel_name[] = {"calc_block_color", "kernel_pixelise", NULL};
    cl_data = gegl_cl_compile_and_build (kernel_source, kernel_name);
  }

  if (!cl_data) return 1;

  gint cx0 = CELL_X(roi->x ,xsize);
  gint cy0 = CELL_Y(roi->y ,ysize);
  gint block_count_x = CELL_X(roi->x+roi->width - 1, xsize)-cx0 + 1;
  gint block_count_y = CELL_Y(roi->y+roi->height - 1, ysize)-cy0 + 1;
  cl_int line_width=roi->width + 2 * xsize;

  size_t gbl_size_tmp[2]={block_count_x,block_count_y};

  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 0, sizeof(cl_mem),   (void*)&in_tex);
  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 1, sizeof(cl_mem),   (void*)&aux_tex);
  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 2, sizeof(cl_int),   (void*)&xsize);
  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 3, sizeof(cl_int),   (void*)&ysize);
  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 4, sizeof(cl_int),   (void*)&roi->x);
  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 5, sizeof(cl_int),   (void*)&roi->y);
  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 6, sizeof(cl_int),   (void*)&line_width);
  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 7, sizeof(cl_int),   (void*)&block_count_x);
  if (cl_err != CL_SUCCESS) return cl_err;
  cl_err = gegl_clEnqueueNDRangeKernel(gegl_cl_get_command_queue (),
                                        cl_data->kernel[0], 2,
                                        NULL, gbl_size_tmp, NULL,
                                        0, NULL, NULL);
  if (cl_err != CL_SUCCESS) return cl_err;

  cl_err |= gegl_clSetKernelArg(cl_data->kernel[1], 0, sizeof(cl_mem),   (void*)&aux_tex);
  cl_err |= gegl_clSetKernelArg(cl_data->kernel[1], 1, sizeof(cl_mem),   (void*)&out_tex);
  cl_err |= gegl_clSetKernelArg(cl_data->kernel[1], 2, sizeof(cl_int),   (void*)&xsize);
  cl_err |= gegl_clSetKernelArg(cl_data->kernel[1], 3, sizeof(cl_int),   (void*)&ysize);
  cl_err |= gegl_clSetKernelArg(cl_data->kernel[1], 4, sizeof(cl_int),   (void*)&roi->x);
  cl_err |= gegl_clSetKernelArg(cl_data->kernel[1], 5, sizeof(cl_int),   (void*)&roi->y);
  cl_err |= gegl_clSetKernelArg(cl_data->kernel[1], 6, sizeof(cl_int),   (void*)&block_count_x);
  if (cl_err != CL_SUCCESS) return cl_err;
  cl_err = gegl_clEnqueueNDRangeKernel(gegl_cl_get_command_queue (),
                                        cl_data->kernel[1], 2,
                                        NULL, gbl_size, NULL,
                                        0, NULL, NULL);
  if (cl_err != CL_SUCCESS) return cl_err;
  return cl_err;
}

static gboolean
cl_process (GeglOperation       *operation,
            GeglBuffer          *input,
            GeglBuffer          *output,
            const GeglRectangle *result)
{
  const Babl *in_format  = gegl_operation_get_format (operation, "input");
  const Babl *out_format = gegl_operation_get_format (operation, "output");
  gint err;
  gint j;
  cl_int cl_err;

  GeglOperationAreaFilter *op_area = GEGL_OPERATION_AREA_FILTER (operation);
  GeglChantO *o = GEGL_CHANT_PROPERTIES (operation);

  GeglBufferClIterator *i = gegl_buffer_cl_iterator_new (output,   result, out_format, GEGL_CL_BUFFER_WRITE);
  gint read = gegl_buffer_cl_iterator_add_2 (i, input, result, in_format,  GEGL_CL_BUFFER_READ, op_area->left, op_area->right, op_area->top, op_area->bottom);
  gint aux  = gegl_buffer_cl_iterator_add_2 (i, NULL, result, in_format,  GEGL_CL_BUFFER_AUX, op_area->left, op_area->right, op_area->top, op_area->bottom);
  while (gegl_buffer_cl_iterator_next (i, &err))
  {
    if (err) return FALSE;
    for (j=0; j < i->n; j++)
    {
      cl_err = cl_pixelise(i->tex[read][j], i->tex[aux][j], i->tex[0][j],&i->roi[read][j], &i->roi[0][j], o->xsize,o->ysize);
      if (cl_err != CL_SUCCESS)
      {
        g_warning("[OpenCL] Error in pixelise: %s\n", gegl_cl_errstring(cl_err));
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
  GeglRectangle rect;
  GeglChantO *o = GEGL_CHANT_PROPERTIES (operation);
  GeglBuffer *temp;
  GeglOperationAreaFilter *op_area;
  op_area = GEGL_OPERATION_AREA_FILTER (operation);

  if (cl_state.is_accelerated)
    if (cl_process (operation, input, output, result))
      return TRUE;

  gfloat* buf;
  rect = *result;
  rect.x -= op_area->left;
  rect.y -= op_area->top;
  rect.width += op_area->left + op_area->right;
  rect.height += op_area->top + op_area->bottom;
  buf = g_new0 (gfloat, rect.width * rect.height * 4);
  gegl_buffer_get (input, 1.0, &rect, babl_format ("RaGaBaA float"), buf, GEGL_AUTO_ROWSTRIDE);
  pixelise(buf, result, o->xsize, o->ysize);
  gegl_buffer_set (output, result, babl_format ("RaGaBaA float"), buf, GEGL_AUTO_ROWSTRIDE);
  g_free (buf);
  return  TRUE;
}


static void
gegl_chant_class_init (GeglChantClass *klass)
{
  GeglOperationClass       *operation_class;
  GeglOperationFilterClass *filter_class;

  operation_class = GEGL_OPERATION_CLASS (klass);
  filter_class    = GEGL_OPERATION_FILTER_CLASS (klass);

  filter_class->process    = process;
  operation_class->prepare = prepare;
  operation_class->categories  = "blur";
  operation_class->name        = "gegl:pixelise";
  operation_class->opencl_support = TRUE;
  operation_class->description =
    _("Pixelise filter.");
}

#endif
