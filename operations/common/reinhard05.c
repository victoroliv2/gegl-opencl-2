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
"                                                                           \n"
"__kernel void rh05_reduYminmaxA(__global float4* pix,                      \n"
"                                __global float*  lum,                      \n"
"                                __global float4* out,                      \n"
"                                __global float2* dst,                      \n"
"                                __local  float2* ldata,                    \n"
"                                const    uint    nitems,                   \n"
"                                const    float   chrom,                    \n"
"                                const    float   chrom_comp,               \n"
"                                const    float   light,                    \n"
"                                const    float   light_comp,               \n"
"                                const    float   intensity,                \n"
"                                const    float   contrast,                 \n"
"                                const    float4  channel_avg,              \n"
"                                const    float   world_lin_avg)            \n"
"{                                                                          \n"
"    // Load shared memory                                                  \n"
"    unsigned int tid = get_local_id(0);                                    \n"
"    unsigned int bid = get_group_id(0);                                    \n"
"    unsigned int gid = get_global_id(0);                                   \n"
"    unsigned int localSize = get_local_size(0);                            \n"
"    unsigned int stride = gid * 2;                                         \n"
"                                                                           \n"
"    float4 inp_v, t_loc_v, t_gbl_v, t_ada_v;                               \n"
"    float  inl;                                                            \n"
"    float2 inm_v1, inm_v2;                                                 \n"
"    float2 outm_v = (float2)(10.0f,-10.0f);                                \n"
"    if (stride < nitems)                                                   \n"
"    {                                                                      \n"
"        inp_v   = pix[stride];                                             \n"
"        inl     = lum[stride];                                             \n"
"        if(inl == 0.0f)                                                    \n"
"        {                                                                  \n"
"            out[stride] = inp_v;                                           \n"
"        }                                                                  \n"
"        else                                                               \n"
"        {                                                                  \n"
"            t_loc_v = chrom * inp_v + chrom_comp * inl;                    \n"
"            t_gbl_v = chrom * channel_avg + chrom_comp * world_lin_avg;    \n"
"            t_ada_v = light * t_loc_v + light_comp * t_gbl_v;              \n"
"            t_loc_v = inp_v / (inp_v + pow(intensity * t_ada_v, contrast));\n"
"            t_loc_v.w = inp_v.w;                                           \n"
"            out[stride] = t_loc_v;                                         \n"
"            outm_v.x = fmin(outm_v.x, t_loc_v.x);                          \n"
"            outm_v.x = fmin(outm_v.x, t_loc_v.y);                          \n"
"            outm_v.x = fmin(outm_v.x, t_loc_v.z);                          \n"
"            outm_v.y = fmax(outm_v.y, t_loc_v.x);                          \n"
"            outm_v.y = fmax(outm_v.y, t_loc_v.y);                          \n"
"            outm_v.y = fmax(outm_v.y, t_loc_v.z);                          \n"
"        }                                                                  \n"
"    }                                                                      \n"
"    if (stride + 1 < nitems)                                               \n"
"    {                                                                      \n"
"        inp_v   = pix[stride + 1];                                         \n"
"        inl     = lum[stride + 1];                                         \n"
"        if(inl == 0.0f)                                                    \n"
"        {                                                                  \n"
"            out[stride + 1] = inp_v;                                       \n"
"        }                                                                  \n"
"        else                                                               \n"
"        {                                                                  \n"
"            t_loc_v = chrom * inp_v + chrom_comp * inl;                    \n"
"            t_gbl_v = chrom * channel_avg + chrom_comp * world_lin_avg;    \n"
"            t_ada_v = light * t_loc_v + light_comp * t_gbl_v;              \n"
"            t_loc_v = inp_v / (inp_v + pow(intensity * t_ada_v, contrast));\n"
"            t_loc_v.w = inp_v.w;                                           \n"
"            out[stride + 1] = t_loc_v;                                     \n"
"            outm_v.x = fmin(outm_v.x, t_loc_v.x);                          \n"
"            outm_v.x = fmin(outm_v.x, t_loc_v.y);                          \n"
"            outm_v.x = fmin(outm_v.x, t_loc_v.z);                          \n"
"            outm_v.y = fmax(outm_v.y, t_loc_v.x);                          \n"
"            outm_v.y = fmax(outm_v.y, t_loc_v.y);                          \n"
"            outm_v.y = fmax(outm_v.y, t_loc_v.z);                          \n"
"        }                                                                  \n"
"    }                                                                      \n"
"    ldata[tid] = outm_v;                                                   \n"
"    barrier(CLK_LOCAL_MEM_FENCE);                                          \n"
"                                                                           \n"
"    // Do reduction in shared memory                                       \n"
"    for (unsigned int s = localSize >>1; s > 0; s >>= 1)                   \n"
"    {                                                                      \n"
"        if(tid < s)                                                        \n"
"        {                                                                  \n"
"            inm_v1 = ldata[tid];                                           \n"
"            inm_v2 = ldata[tid + s];                                       \n"
"            outm_v.x  = fmin(inm_v1.x, inm_v2.x);                          \n"
"            outm_v.y  = fmax(inm_v1.y, inm_v2.y);                          \n"
"            ldata[tid] = outm_v;                                           \n"
"        }                                                                  \n"
"        barrier(CLK_LOCAL_MEM_FENCE);                                      \n"
"    }                                                                      \n"
"                                                                           \n"
"    // Write result for this block to global memory.                       \n"
"    if (tid ==0)                                                           \n"
"        dst[bid] = ldata[0];                                               \n"
"}                                                                          \n"
"                                                                           \n"
"__kernel void rh05_reduYminmaxB(__global float2* src,                      \n"
"                                __global float2* dst,                      \n"
"                                __local  float2* ldata,                    \n"
"                                const    uint    nitems)                   \n"
"{                                                                          \n"
"    // Load shared memory                                                  \n"
"    unsigned int tid = get_local_id(0);                                    \n"
"    unsigned int bid = get_group_id(0);                                    \n"
"    unsigned int gid = get_global_id(0);                                   \n"
"    unsigned int localSize = get_local_size(0);                            \n"
"    unsigned int stride = gid * 2;                                         \n"
"                                                                           \n"
"    float2 in_v1 = (float2)(10.0f, -10.0f);                                \n"
"    float2 in_v2 = (float2)(10.0f, -10.0f);                                \n"
"    float2 out_v = 0.0f;                                                   \n"
"    if (stride < nitems)                                                   \n"
"        in_v1 = src[stride];                                               \n"
"    if (stride + 1 < nitems)                                               \n"
"        in_v2 = src[stride + 1];                                           \n"
"                                                                           \n"
"    out_v.x  = fmin(in_v1.x, in_v2.x);                                     \n"
"    out_v.y  = fmax(in_v1.y, in_v2.y);                                     \n"
"    ldata[tid] = out_v;                                                    \n"
"    barrier(CLK_LOCAL_MEM_FENCE);                                          \n"
"                                                                           \n"
"    // Do reduction in shared memory                                       \n"
"    for (unsigned int s = localSize >>1; s > 0; s >>= 1)                   \n"
"    {                                                                      \n"
"        if(tid < s)                                                        \n"
"        {                                                                  \n"
"            in_v1 = ldata[tid];                                            \n"
"            in_v2 = ldata[tid + s];                                        \n"
"            out_v.x  = fmin(in_v1.x, in_v2.x);                             \n"
"            out_v.y  = fmax(in_v1.y, in_v2.y);                             \n"
"            ldata[tid] = out_v;                                            \n"
"        }                                                                  \n"
"        barrier(CLK_LOCAL_MEM_FENCE);                                      \n"
"    }                                                                      \n"
"                                                                           \n"
"    // Write result for this block to global memory.                       \n"
"    if (tid ==0)                                                           \n"
"        dst[bid] = ldata[0];                                               \n"
"}                                                                          \n"
"                                                                \n"
"__kernel void reinhard05_2 (__global float4 * src,              \n"
"                            __global float4 * dst,              \n"
"                            float min,                          \n"
"                            float range)                        \n"
"{                                                               \n"
" int gid = get_global_id(0);                                    \n"
" dst[gid] = (src[gid]-min) / range;                             \n"
"}                                                               \n";

static const char* kernel_source_add =
"                                                                           \n"
"__kernel void rh05_reduYminmaxsumA(__global float*  src,                   \n"
"                                   __global float4* dst,                   \n"
"                                   __local  float4* ldata,                 \n"
"                                   const    uint    nitems)                \n"
"{                                                                          \n"
"    // Load shared memory                                                  \n"
"    unsigned int tid = get_local_id(0);                                    \n"
"    unsigned int bid = get_group_id(0);                                    \n"
"    unsigned int gid = get_global_id(0);                                   \n"
"    unsigned int localSize = get_local_size(0);                            \n"
"    unsigned int stride = gid * 2;                                         \n"
"                                                                           \n"
"    float4 in_v1 = (float4)((float) + 10.0f, (float) -10.0f, 0.0f, 0.0f);  \n"
"    float4 in_v2 = (float4)((float) + 10.0f, (float) - 10.0f, 0.0f, 0.0f); \n"
"    float4 out_v = 0.0f;                                                   \n"
"    if (stride < nitems)                                                   \n"
"    {                                                                      \n"
"        in_v1 = src[stride];                                               \n"
"        in_v1.w = log(2.3e-5f + in_v1.w);                                  \n"
"    }                                                                      \n"
"    if (stride + 1 < nitems)                                               \n"
"    {                                                                      \n"
"        in_v2 = src[stride + 1];                                           \n"
"        in_v2.w = log(2.3e-5f + in_v2.w);                                  \n"
"    }                                                                      \n"
"                                                                           \n"
"    out_v.x  = fmin(in_v1.x, in_v2.x);                                     \n"
"    out_v.y  = fmax(in_v1.y, in_v2.y);                                     \n"
"    out_v.zw = in_v1.zw + in_v2.zw;                                        \n"
"    ldata[tid] = out_v;                                                    \n"
"    barrier(CLK_LOCAL_MEM_FENCE);                                          \n"
"                                                                           \n"
"    // Do reduction in shared memory                                       \n"
"    for (unsigned int s = localSize >>1; s > 0; s >>= 1)                   \n"
"    {                                                                      \n"
"        if(tid < s)                                                        \n"
"        {                                                                  \n"
"            in_v1 = ldata[tid];                                            \n"
"            in_v2 = ldata[tid + s];                                        \n"
"            out_v.x  = fmin(in_v1.x, in_v2.x);                             \n"
"            out_v.y  = fmax(in_v1.y, in_v2.y);                             \n"
"            out_v.zw = in_v1.zw + in_v2.zw;                                \n"
"            ldata[tid] = out_v;                                            \n"
"        }                                                                  \n"
"        barrier(CLK_LOCAL_MEM_FENCE);                                      \n"
"    }                                                                      \n"
"                                                                           \n"
"    // Write result for this block to global memory.                       \n"
"    if (tid ==0)                                                           \n"
"        dst[bid] = ldata[0];                                               \n"
"}                                                                          \n"
"                                                                           \n"
"__kernel void rh05_reduYminmaxsumB(__global float4* src,                   \n"
"                                   __global float4* dst,                   \n"
"                                   __local  float4* ldata,                 \n"
"                                   const    uint    nitems)                \n"
"{                                                                          \n"
"    // Load shared memory                                                  \n"
"    unsigned int tid = get_local_id(0);                                    \n"
"    unsigned int bid = get_group_id(0);                                    \n"
"    unsigned int gid = get_global_id(0);                                   \n"
"    unsigned int localSize = get_local_size(0);                            \n"
"    unsigned int stride = gid * 2;                                         \n"
"                                                                           \n"
"    float4 in_v1 = (float4)((float) + 10.0f, (float) -10.0f, 0.0f, 0.0f);  \n"
"    float4 in_v2 = (float4)((float) + 10.0f, (float) - 10.0f, 0.0f, 0.0f); \n"
"    float4 out_v = 0.0f;                                                   \n"
"    if (stride < nitems)                                                   \n"
"        in_v1 = src[stride];                                               \n"
"    if (stride + 1 < nitems)                                               \n"
"        in_v2 = src[stride + 1];                                           \n"
"                                                                           \n"
"    out_v.x  = fmin(in_v1.x, in_v2.x);                                     \n"
"    out_v.y  = fmax(in_v1.y, in_v2.y);                                     \n"
"    out_v.zw = in_v1.zw + in_v2.zw;                                        \n"
"    ldata[tid] = out_v;                                                    \n"
"    barrier(CLK_LOCAL_MEM_FENCE);                                          \n"
"                                                                           \n"
"    // Do reduction in shared memory                                       \n"
"    for (unsigned int s = localSize >>1; s > 0; s >>= 1)                   \n"
"    {                                                                      \n"
"        if(tid < s)                                                        \n"
"        {                                                                  \n"
"            in_v1 = ldata[tid];                                            \n"
"            in_v2 = ldata[tid + s];                                        \n"
"            out_v.x  = fmin(in_v1.x, in_v2.x);                             \n"
"            out_v.y  = fmax(in_v1.y, in_v2.y);                             \n"
"            out_v.zw = in_v1.zw + in_v2.zw;                                \n"
"            ldata[tid] = out_v;                                            \n"
"        }                                                                  \n"
"        barrier(CLK_LOCAL_MEM_FENCE);                                      \n"
"    }                                                                      \n"
"                                                                           \n"
"    // Write result for this block to global memory.                       \n"
"    if (tid ==0)                                                           \n"
"        dst[bid] = ldata[0];                                               \n"
"}                                                                          \n"
"__kernel void rh05_reduRGB_sum(__global  float4* src,                      \n"
"                                __global float4* dst,                      \n"
"                                __local  float4* lsum,                     \n"
"                                const    uint    nitems)                   \n"
"{                                                                          \n"
"    // Load shared memory                                                  \n"
"    unsigned int tid = get_local_id(0);                                    \n"
"    unsigned int bid = get_group_id(0);                                    \n"
"    unsigned int gid = get_global_id(0);                                   \n"
"    unsigned int localSize = get_local_size(0);                            \n"
"    unsigned int stride = gid * 2;                                         \n"
"                                                                           \n"
"    float4 in_v1 = (float) + 0.0f;                                         \n"
"    float4 in_v2 = (float) + 0.0f;                                         \n"
"    if (stride < nitems)                                                   \n"
"        in_v1 = src[stride];                                               \n"
"    if (stride + 1 < nitems)                                               \n"
"        in_v2 = src[stride + 1];                                           \n"
"                                                                           \n"
"    lsum[tid] = in_v1 + in_v2;                                             \n"
"    barrier(CLK_LOCAL_MEM_FENCE);                                          \n"
"                                                                           \n"
"    // Do reduction in shared memory                                       \n"
"    for (unsigned int s = localSize >>1; s > 0; s >>= 1)                   \n"
"    {                                                                      \n"
"        if(tid < s)                                                        \n"
"        {                                                                  \n"
"            lsum[tid] = lsum[tid] + lsum[tid + s];                         \n"
"        }                                                                  \n"
"        barrier(CLK_LOCAL_MEM_FENCE);                                      \n"
"    }                                                                      \n"
"                                                                           \n"
"    // Write result for this block to global memory.                       \n"
"    if (tid ==0)                                                           \n"
"        dst[bid] = lsum[0];                                                \n"
"}                                                                          \n"
"                                                                           \n";

static gegl_cl_run_data * cl_data     = NULL;
static gegl_cl_run_data * cl_data_add = NULL;

static cl_int
cl_reinhard05_0 (cl_mem               in_tex,
				 cl_mem               lum_tex,
				 cl_mem              *output_tex,
				 size_t               pixel_size,
				 stats               *world_lin,
				 float                world_log_sum[1],
				 float                channel_sum[3])
{
	cl_int cl_err = 0;
	int flag = 0,kernelNo;
	size_t local_size = 64,item_size = 2,item_count,data_size,global_size, group_num;
	float result_temp[4];

	cl_mem src_tex, out_tex = * output_tex;

	src_tex = gegl_clCreateBuffer (gegl_cl_get_context(),
					CL_MEM_READ_WRITE,
					pixel_size * 4 * sizeof(float),
					NULL, &cl_err);
	if (cl_err != CL_SUCCESS) return cl_err;

	if (!cl_data_add)
	{
		const char *kernel_name[] = {"rh05_reduYminmaxsumA", "rh05_reduYminmaxsumB","rh05_reduRGB_sum", NULL};
		cl_data_add = gegl_cl_compile_and_build (kernel_source_add, kernel_name);
	}
	if(!cl_data_add)  return 0;

	if (!cl_data)
	{
		const char *kernel_name[] = {"rh05_reduYminmaxA","rh05_reduYminmaxB", "reinhard05_2", NULL};
		cl_data = gegl_cl_compile_and_build (kernel_source, kernel_name);
	}
	if (!cl_data) return 1;

	//compute the world_lin¡¢world_log_sum
	cl_err = gegl_clEnqueueCopyBuffer(gegl_cl_get_command_queue(),
		lum_tex , src_tex , 0 , 0 ,
		pixel_size * sizeof(float),
		NULL, NULL, NULL);


	data_size   = pixel_size;
	item_count  = (data_size + 1)/item_size;
	global_size = (item_count % local_size == 0)? item_count : (item_count / local_size + 1) * local_size;
    kernelNo    = 0;

	while(global_size >= local_size){

		group_num = global_size / local_size;
		if(flag){
			kernelNo = 1;
			cl_mem temp;
			temp     = src_tex;
			src_tex  = out_tex;
			out_tex  = temp;
		}		
		cl_err |= gegl_clSetKernelArg(cl_data_add->kernel[kernelNo], 0, sizeof(cl_mem),    (void*)&src_tex);
		cl_err |= gegl_clSetKernelArg(cl_data_add->kernel[kernelNo], 1, sizeof(cl_mem),    (void*)&out_tex);

		cl_err |= gegl_clSetKernelArg(cl_data_add->kernel[kernelNo], 2, local_size * sizeof(cl_float4),  NULL);

		cl_err |= gegl_clSetKernelArg(cl_data_add->kernel[kernelNo], 3, sizeof(cl_int),  (void*)&data_size);
		if (cl_err != CL_SUCCESS) return cl_err;

		cl_err = gegl_clEnqueueNDRangeKernel(gegl_cl_get_command_queue (),
			cl_data_add->kernel[kernelNo], 1,
			NULL, &global_size, &local_size,
			0, NULL, NULL);



		if(group_num == 1)
			break;

		data_size   = group_num;
		item_count  = (data_size + 1) / item_size;
		global_size = (item_count % local_size == 0)? item_count : (item_count / local_size + 1)* local_size;

		flag++;	
	}

	cl_err = gegl_clFinish(gegl_cl_get_command_queue());
	if (CL_SUCCESS != cl_err) return cl_err;

	
	gegl_clEnqueueReadBuffer(gegl_cl_get_command_queue (),
		out_tex,CL_TRUE,
		0,sizeof(cl_float4),result_temp,
		0,NULL,NULL);



	
	world_lin->min = result_temp[0];
	world_lin->max = result_temp[1];
	world_lin->avg = result_temp[2];
	world_lin->num = pixel_size;

	world_log_sum[0] = result_temp[3];

	//compute the channel[i].sum
	cl_err = gegl_clEnqueueCopyBuffer(gegl_cl_get_command_queue(),
		in_tex , src_tex , 0 , 0 ,
		pixel_size * 4 * sizeof(float),
		NULL, NULL, NULL);


	flag = 0;
	data_size  = pixel_size;
	item_count = (data_size + 1)/item_size;
	global_size = (item_count % local_size == 0)? item_count : (item_count / local_size + 1)* local_size;

	while(global_size >= local_size){

		group_num = global_size / local_size;
		if(flag){
			cl_mem temp;
			temp     = src_tex;
			src_tex = out_tex;
			out_tex  = temp;
		}		
		cl_err |= gegl_clSetKernelArg(cl_data_add->kernel[2], 0, sizeof(cl_mem),    (void*)&src_tex);
		cl_err |= gegl_clSetKernelArg(cl_data_add->kernel[2], 1, sizeof(cl_mem),    (void*)&out_tex);

		cl_err |= gegl_clSetKernelArg(cl_data_add->kernel[2], 2, local_size * sizeof(cl_float4),  NULL);

		cl_err |= gegl_clSetKernelArg(cl_data_add->kernel[2], 3, sizeof(cl_int),  (void*)&data_size);

		if (cl_err != CL_SUCCESS) return cl_err;
		cl_err = gegl_clEnqueueNDRangeKernel(gegl_cl_get_command_queue (),
			cl_data_add->kernel[2], 1,
			NULL, &global_size, &local_size,
			0, NULL, NULL);



		if(group_num == 1)
			break;
		data_size  = group_num;
		item_count = (data_size + 1) / item_size;
		global_size = (item_count % local_size == 0)? item_count : (item_count / local_size + 1)* local_size;
		
		flag++;	
	}
	cl_err = gegl_clFinish(gegl_cl_get_command_queue());
	if (CL_SUCCESS != cl_err) return cl_err;

	gegl_clEnqueueReadBuffer(gegl_cl_get_command_queue (),
		out_tex,CL_TRUE,
		0,sizeof(cl_float3),channel_sum,
		0,NULL,NULL);
	
	*output_tex = out_tex;
	if(src_tex) gegl_clReleaseMemObject (src_tex);

	return cl_err;

}

static cl_int
cl_reinhard05_1 (cl_mem               in_tex,
                 cl_mem               lum_tex,
                 cl_mem               out_tex,
                 size_t               pixel_size,
                 gfloat               chrom,
                 gfloat               light,
                 gfloat               intensity,
                 gfloat               contrast,
                 stats                world_lin,
                 stats                channel[],
				 gfloat              *normal_min,
				 gfloat				 *normal_max)
{
	cl_int cl_err = 0;
	cl_float4 channel_avg = {channel[0].avg, channel[1].avg, channel[2].avg, 1.0f};
	cl_float  chrom_comp = 1.0 - chrom,
	          light_comp = 1.0 - light;

	int flag = 0 ,kernelNo ;
	size_t local_size = 64,item_size = 2,item_count,data_size,global_size, group_num;
	float  result_temp[2];

	cl_mem src_tex,dst_tex;

	src_tex = gegl_clCreateBuffer (gegl_cl_get_context(),
				CL_MEM_READ_WRITE,
				pixel_size * 4 * sizeof(float),
				NULL, &cl_err);
	cl_err = gegl_clEnqueueCopyBuffer(gegl_cl_get_command_queue(),
				in_tex , src_tex , 0 , 0 ,
				pixel_size * 4 * sizeof(float),
				NULL, NULL, NULL);



	dst_tex = gegl_clCreateBuffer (gegl_cl_get_context(),
				CL_MEM_READ_WRITE,
				pixel_size * 2 * sizeof(float),
				NULL, &cl_err);
	if (cl_err != CL_SUCCESS) return cl_err;


  if (!cl_data)
	{
	  const char *kernel_name[] = {"rh05_reduYminmaxA","rh05_reduYminmaxB", "reinhard05_2", NULL};
	  cl_data = gegl_cl_compile_and_build (kernel_source, kernel_name);
	}
  if (!cl_data) return 1;

  kernelNo = 0;
  data_size  = pixel_size;
  item_count = (data_size + 1)/item_size;
  global_size = (item_count % local_size == 0)? item_count : (item_count / local_size + 1) * local_size;

  while(global_size >= local_size){

	  group_num = global_size / local_size;
	  if(flag){
		  cl_mem temp;
		  temp     = src_tex;
		  src_tex  = dst_tex;
		  dst_tex  = temp;

		  cl_err |= gegl_clSetKernelArg(cl_data->kernel[1], 0, sizeof(cl_mem),    (void*)&src_tex);
		  cl_err |= gegl_clSetKernelArg(cl_data->kernel[1], 1, sizeof(cl_mem),    (void*)&dst_tex);

		  cl_err |= gegl_clSetKernelArg(cl_data->kernel[1], 2, local_size * sizeof(cl_float2),  NULL);
		  cl_err |= gegl_clSetKernelArg(cl_data->kernel[1], 3, sizeof(cl_int),  (void*)&data_size);

		  kernelNo = 1;
		  
	  }
	  else{  
		  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 0, sizeof(cl_mem),    (void*)&src_tex);
		  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 1, sizeof(cl_mem),    (void*)&lum_tex);
		  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 2, sizeof(cl_mem),    (void*)&out_tex);
		  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 3, sizeof(cl_mem),    (void*)&dst_tex);

		  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 4, local_size * sizeof(cl_float2),  NULL);
		  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 5, sizeof(cl_int),  (void*)&data_size);

		  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 6, sizeof(cl_float),  (void*)&chrom);
		  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 7, sizeof(cl_float),  (void*)&chrom_comp);
		  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 8, sizeof(cl_float),  (void*)&light);
		  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0], 9, sizeof(cl_float),  (void*)&light_comp);
		  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0],10, sizeof(cl_float),  (void*)&intensity);
		  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0],11, sizeof(cl_float),  (void*)&contrast);
		  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0],12, sizeof(cl_float4), (void*)&channel_avg);
		  cl_err |= gegl_clSetKernelArg(cl_data->kernel[0],13, sizeof(cl_float),  (void*)&world_lin.avg);
	  }
	  if (cl_err != CL_SUCCESS)  return cl_err;	 
	
	  cl_err = gegl_clEnqueueNDRangeKernel(gegl_cl_get_command_queue (),
		  cl_data->kernel[kernelNo], 1,
		  NULL, &global_size, &local_size,
		  0, NULL, NULL);



	  if(group_num == 1)
		  break;

	  data_size   = group_num;
	  item_count  = (data_size + 1) / item_size;
	  global_size = (item_count % local_size == 0)? item_count : (item_count / local_size + 1)* local_size;

	  flag++;	
  }

  cl_err = gegl_clFinish(gegl_cl_get_command_queue());
  if (CL_SUCCESS != cl_err) return cl_err;

  gegl_clEnqueueReadBuffer(gegl_cl_get_command_queue (),
			  dst_tex,CL_TRUE,
			  0,sizeof(cl_float2),result_temp,
			  0,NULL,NULL);


  *normal_min = result_temp[0];
  *normal_max = result_temp[1];

  if(src_tex) gegl_clReleaseMemObject (src_tex);
  if(dst_tex) gegl_clReleaseMemObject (dst_tex);

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

  cl_err |= gegl_clSetKernelArg(cl_data->kernel[2], 0, sizeof(cl_mem),    (void*)&in_tex);
  cl_err |= gegl_clSetKernelArg(cl_data->kernel[2], 1, sizeof(cl_mem),    (void*)&out_tex);
  cl_err |= gegl_clSetKernelArg(cl_data->kernel[2], 2, sizeof(cl_float),  (void*)&min);
  cl_err |= gegl_clSetKernelArg(cl_data->kernel[2], 3, sizeof(cl_float),  (void*)&range);
  if (cl_err != CL_SUCCESS) return cl_err;

  cl_err = gegl_clEnqueueNDRangeKernel(gegl_cl_get_command_queue (),
                                        cl_data->kernel[2], 1,
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

  /* Collect the image stats, averages, etc */
  reinhard05_stats_start (&world_lin);
  reinhard05_stats_start (&world_log);
  reinhard05_stats_start (&normalise);
  for (i = 0; i < RGB; ++i)
  {
	  reinhard05_stats_start (channel + i);
  }

  if (cl_state.is_accelerated)
    {
      const Babl *in_format  = gegl_operation_get_format (operation, "input");
      const Babl *out_format = gegl_operation_get_format (operation, "output");
      const Babl *lum_format = babl_format("Y float");

      GeglBuffer *pix_out = gegl_buffer_new (result, in_format);
      gint j, k;
      cl_int err, cl_err;

	  {
		  GeglBufferClIterator *ii= gegl_buffer_cl_iterator_new (pix_out, result, in_format, GEGL_CL_BUFFER_WRITE);
		  gint read = gegl_buffer_cl_iterator_add (ii, input, result, in_format,  GEGL_CL_BUFFER_READ);
		  gint lum  = gegl_buffer_cl_iterator_add (ii, input, result, lum_format, GEGL_CL_BUFFER_READ);

		  while (gegl_buffer_cl_iterator_next (ii, &err))
		  {
			  if (err) return FALSE;
			  for (j=0; j < ii->n; j++)
			  {
				  stats world_lin_temp;
				  float world_log_sum[1];
				  float channel_sum[3];

				  cl_reinhard05_0(ii->tex[read][j],ii->tex[lum][j],&ii->tex[0][j],ii->size[0][j],
					  &world_lin_temp,world_log_sum,channel_sum);

				  world_lin.min = MIN(world_lin.min,world_lin_temp.min);
				  world_lin.max = MAX(world_lin.max,world_lin_temp.max);
				  world_lin.avg += world_lin_temp.avg;
				  world_lin.num += world_lin_temp.num;

				  world_log.avg += world_log_sum[0];
				  world_log.num += ii->size[0][j];

				  for(k = 0; k < RGB; k++ ){
					  channel[k].avg += channel_sum[k];
					  channel[k].num += ii->size[0][j];
				  }

			  }		 
		  }
		  
	  }
	  g_return_val_if_fail (world_lin.min >= 0.0, FALSE);

	  world_log.min = world_log.max = 0.0f;
	  for (i = 0; i < RGB; ++i)
	  {
		  channel[i].min = channel[i].max = 0.0f;
	  }

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

      {
		  GeglBufferClIterator *i = gegl_buffer_cl_iterator_new (pix_out, result, in_format, GEGL_CL_BUFFER_WRITE);
		  gint read = gegl_buffer_cl_iterator_add (i, input, result, in_format,  GEGL_CL_BUFFER_READ);
		  gint lum_ = gegl_buffer_cl_iterator_add (i, input, result, lum_format, GEGL_CL_BUFFER_READ);

		  while (gegl_buffer_cl_iterator_next (i, &err))
			{
			  if (err) return FALSE;
			  for (j=0; j < i->n; j++)
				{
				  float min_temp,max_temp;
				  cl_err = cl_reinhard05_1(i->tex[read][j], i->tex[lum_][j], i->tex[0][j], i->size[0][j], 
				      chrom, light, intensity, contrast,
				      world_lin, channel,
					  &min_temp, &max_temp);
				  if (cl_err != CL_SUCCESS)
					{
					  g_warning("[OpenCL] Error in gegl:reinhard05: %s\n", gegl_cl_errstring(cl_err));
					  return FALSE;
					}
				  normalise.min = MIN(normalise.min,min_temp);
				  normalise.max = MAX(normalise.max,max_temp);			  

				}
			}
	  }
      /* Normalise the pixel values */
	  normalise.range = normalise.max - normalise.min; 

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
      if(pix_out)    gegl_buffer_destroy(pix_out);
    }
  else
    {
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

	  if(pix)g_free (pix);
	  g_free (lum);
    }

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


