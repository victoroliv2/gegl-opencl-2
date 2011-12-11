#include "gegl.h"
#include "gegl-cl-color.h"
#include "gegl-cl-init.h"

static const Babl *format[6];

void
gegl_cl_color_prepare(void)
{
  /* non_premultiplied_to_premultiplied -> 0 */
  /* premultiplied_to_non_premultiplied -> 1 */
  /* rgba2rgba_gamma_2_2                -> 2 */
  /* rgba_gamma_2_22rgba                -> 3 */
  /* rgba2rgba_gamma_2_2_premultiplied  -> 4 */
  /* rgba_gamma_2_2_premultiplied2rgba  -> 5 */

  format[0] = babl_format ("RaGaBaA float");
  format[1] = babl_format ("RGBA float");
  format[2] = babl_format ("R'G'B'A float");
  format[3] = babl_format ("RGBA float");
  format[4] = babl_format ("R'aG'aB'aA float");
  format[5] = babl_format ("RGBA float");
}

gboolean
gegl_cl_color_supported (const Babl *in_format, const Babl *out_format)
{
  int i;
  gboolean supported_format_in  = FALSE;
  gboolean supported_format_out = FALSE;

  for (i = 0; i < 6; i++)
    {
      if (format[i] == in_format)  supported_format_in  = TRUE;
      if (format[i] == out_format) supported_format_out = TRUE;
    }

  return (supported_format_in && supported_format_out);
}

#define CONV_1(x)   {conv[0] = x; conv[1] = -1;}
#define CONV_2(x,y) {conv[0] = x; conv[1] =  y;}

gboolean
gegl_cl_color_conv (const Babl *in_format, const Babl *out_format, gint conv[2])
{
  int errcode;

  conv[0] = -1;
  conv[1] = -1;

  if (!gegl_cl_color_supported (in_format, out_format))
    return FALSE;

  if (in_format != out_format)
    {
      if      (in_format == babl_format ("RGBA float"))
        {
          if      (out_format == babl_format ("RaGaBaA float"))    CONV_1(0)
          else if (out_format == babl_format ("R'G'B'A float"))    CONV_1(2)
          else if (out_format == babl_format ("R'aG'aB'aA float")) CONV_1(4)
        }
      else if (in_format == babl_format ("RaGaBaA float"))
        {
          if      (out_format == babl_format ("RGBA float"))       CONV_1(1)
          else if (out_format == babl_format ("R'G'B'A float"))    CONV_2(1, 2)
          else if (out_format == babl_format ("R'aG'aB'aA float")) CONV_2(1, 4)
        }
      else if (in_format == babl_format ("R'G'B'A float"))
        {
          if      (out_format == babl_format ("RGBA float"))       CONV_1(3)
          else if (out_format == babl_format ("RaGaBaA float"))    CONV_2(3, 0)
          else if (out_format == babl_format ("R'aG'aB'aA float")) CONV_2(3, 4)
        }
      else if (in_format == babl_format ("R'aG'aB'aA float"))
        {
          if      (out_format == babl_format ("RGBA float"))       CONV_1(5)
          else if (out_format == babl_format ("RaGaBaA float"))    CONV_2(5, 0)
          else if (out_format == babl_format ("R'G'B'A float"))    CONV_2(5, 2)
        }
    }

  return TRUE;
}
