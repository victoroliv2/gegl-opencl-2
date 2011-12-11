#ifndef __GEGL_CL_COLOR_H__
#define __GEGL_CL_COLOR_H__

#include <gegl.h>
#include "gegl-cl-types.h"

void gegl_cl_color_prepare(void);

gboolean gegl_cl_color_supported (const Babl *in_format, const Babl *out_format);

gboolean gegl_cl_color_conv (const Babl *in_format, const Babl *out_format, gint conv[2]);
#endif
