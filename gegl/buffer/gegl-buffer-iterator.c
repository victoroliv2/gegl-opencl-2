/* This file is part of GEGL.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 * Copyright 2008 Øyvind Kolås <pippin@gimp.org>
 */

#include "config.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <glib-object.h>
#include <glib/gprintf.h>

#include "gegl.h"
#include "gegl-types-internal.h"
#include "gegl-buffer-types.h"
#include "gegl-buffer-iterator.h"
#include "gegl-buffer-private.h"
#include "gegl-tile-storage.h"
#include "gegl-utils.h"

typedef struct GeglBufferTileIterator
{
  GeglBuffer    *buffer;
  GeglRectangle  roi;     /* the rectangular region we're iterating over */
  GeglTile      *tile;    /* current tile */
  gpointer       data;    /* current tile's data */
  GeglClTexture *cl_data; /* current tile's data */

  gint           col;     /* the column currently provided for */
  gint           row;     /* the row currently provided for */
  GeglTileLockMode
                 lock_mode;
  GeglRectangle  subrect;    /* the subrect that intersected roi */
  gpointer       sub_data;   /* pointer to the subdata as indicated by subrect */
  gint           rowstride;  /* rowstride for tile, in bytes */

  gint           next_col; /* used internally */
  gint           next_row; /* used internally */
  gint           max_size; /* maximum data buffer needed, in bytes */
  GeglRectangle  roi2;     /* the rectangular subregion of data
                            * in the buffer represented by this scan.
                            */

} GeglBufferTileIterator;

#define GEGL_BUFFER_SCAN_COMPATIBLE   128   /* should be integrated into enum */
#define GEGL_BUFFER_FORMAT_COMPATIBLE 256   /* should be integrated into enum */

#define DEBUG_DIRECT 0

typedef struct GeglBufferIterators
{
  /* current region of interest */
  gint          length;             /* length of current data in pixels */
  gpointer      data[GEGL_BUFFER_MAX_ITERATORS];
  GeglRectangle roi[GEGL_BUFFER_MAX_ITERATORS]; /* roi of the current data */

  /* the following is private: */
  gint           iterators;
  gint           iteration_no;
  gboolean       is_finished;
  GeglRectangle  rect       [GEGL_BUFFER_MAX_ITERATORS]; /* the region we iterate on. They can be different from
                                                            each other, but width and height are the same */
  const Babl    *format     [GEGL_BUFFER_MAX_ITERATORS]; /* The format required for the data */
  GeglBuffer    *buffer     [GEGL_BUFFER_MAX_ITERATORS]; /* currently a subbuffer of the original, need to go away */
  guint          flags      [GEGL_BUFFER_MAX_ITERATORS];
  gpointer       buf        [GEGL_BUFFER_MAX_ITERATORS]; /* no idea */
  GeglBufferTileIterator   i[GEGL_BUFFER_MAX_ITERATORS];
} GeglBufferIterators;


static void      gegl_buffer_tile_iterator_init (GeglBufferTileIterator *i,
                                                 GeglBuffer             *buffer,
                                                 GeglRectangle           roi,
                                                 GeglTileLockMode        lock_mode);
static gboolean  gegl_buffer_tile_iterator_next (GeglBufferTileIterator *i);

/*
 *  check whether iterations on two buffers starting from the given coordinates with
 *  the same width and height would be able to run parallell.
 */
static gboolean gegl_buffer_scan_compatible (GeglBuffer *bufferA,
                                             gint        xA,
                                             gint        yA,
                                             GeglBuffer *bufferB,
                                             gint        xB,
                                             gint        yB)
{
  if (bufferA->tile_storage->tile_width !=
      bufferB->tile_storage->tile_width)
    return FALSE;
  if (bufferA->tile_storage->tile_height !=
      bufferB->tile_storage->tile_height)
    return FALSE;
  if ( (abs((bufferA->shift_x+xA) - (bufferB->shift_x+xB))
        % bufferA->tile_storage->tile_width) != 0)
    return FALSE;
  if ( (abs((bufferA->shift_y+yA) - (bufferB->shift_y+yB))
        % bufferA->tile_storage->tile_height) != 0)
    return FALSE;
  return TRUE;
}

static void gegl_buffer_tile_iterator_init (GeglBufferTileIterator *i,
                                            GeglBuffer             *buffer,
                                            GeglRectangle           roi,
                                            GeglTileLockMode        lock_mode)
{
  g_assert (i);
  memset (i, 0, sizeof (GeglBufferTileIterator));
  if (roi.width == 0 ||
      roi.height == 0)
    g_error ("eeek");
  i->buffer = buffer;
  i->roi = roi;
  i->next_row    = 0;
  i->next_col = 0;
  i->tile = NULL;
  i->col = 0;
  i->row = 0;
  i->lock_mode = lock_mode;
  i->max_size = i->buffer->tile_storage->tile_width *
                i->buffer->tile_storage->tile_height;
}

static gboolean
gegl_buffer_tile_iterator_next (GeglBufferTileIterator *i)
{
  GeglBuffer *buffer   = i->buffer;
  gint  tile_width     = buffer->tile_storage->tile_width;
  gint  tile_height    = buffer->tile_storage->tile_height;
  gint  buffer_shift_x = buffer->shift_x;
  gint  buffer_shift_y = buffer->shift_y;
  gint  buffer_x       = i->roi.x + buffer_shift_x;
  gint  buffer_y       = i->roi.y + buffer_shift_y;

  gboolean direct_access, cl_direct_access;

  if (i->roi.width == 0 || i->roi.height == 0)
    return FALSE;

gulp:

  /* unref previously held tile */
  if (i->tile)
    {
      direct_access = (i->lock_mode & GEGL_TILE_LOCK_READ || i->lock_mode & GEGL_TILE_LOCK_WRITE) &&
                      tile_width == i->subrect.width;

      cl_direct_access = (i->lock_mode & GEGL_TILE_LOCK_CL_READ || i->lock_mode & GEGL_TILE_LOCK_WRITE) &&
                         tile_width == i->subrect.width && tile_height == i->subrect.height;

      if (i->lock_mode != GEGL_TILE_LOCK_NONE &&
          (direct_access || cl_direct_access))
        {
          gegl_tile_unlock (i->tile);
        }
      gegl_tile_unref (i->tile);
      i->tile = NULL;
    }

  if (i->next_col < i->roi.width)
    { /* return tile on this row */
      gint tiledx = buffer_x + i->next_col;
      gint tiledy = buffer_y + i->next_row;
      gint offsetx = gegl_tile_offset (tiledx, tile_width);
      gint offsety = gegl_tile_offset (tiledy, tile_height);

        {
         i->subrect.x = offsetx;
         i->subrect.y = offsety;
         if (i->roi.width + offsetx - i->next_col < tile_width)
           i->subrect.width = (i->roi.width + offsetx - i->next_col) - offsetx;
         else
           i->subrect.width = tile_width - offsetx;

         if (i->roi.height + offsety - i->next_row < tile_height)
           i->subrect.height = (i->roi.height + offsety - i->next_row) - offsety;
         else
           i->subrect.height = tile_height - offsety;

         i->tile = gegl_tile_source_get_tile ((GeglTileSource *) (buffer),
                                               gegl_tile_indice (tiledx, tile_width),
                                               gegl_tile_indice (tiledy, tile_height),
                                               0);

         direct_access = (i->lock_mode & GEGL_TILE_LOCK_READ || i->lock_mode & GEGL_TILE_LOCK_WRITE) &&
                         tile_width == i->subrect.width;

         cl_direct_access = (i->lock_mode & GEGL_TILE_LOCK_CL_READ || i->lock_mode & GEGL_TILE_LOCK_WRITE) &&
                            tile_width == i->subrect.width && tile_height == i->subrect.height;

         if (i->lock_mode != GEGL_TILE_LOCK_NONE &&
             (direct_access || cl_direct_access))
           {
             gegl_tile_lock (i->tile, i->lock_mode);
           }

         /* no need to OpenCL sync here */
         i->data    = i->tile->data;
         i->cl_data = i->tile->cl_data;

         {
         gint bpp = babl_format_get_bytes_per_pixel (i->buffer->format);
         i->rowstride = bpp * tile_width;
         i->sub_data = (guchar*)(i->data) + bpp * (i->subrect.y * tile_width + i->subrect.x);
         }

         i->col = i->next_col;
         i->row = i->next_row;
         i->next_col += tile_width - offsetx;


         i->roi2.x      = i->roi.x + i->col;
         i->roi2.y      = i->roi.y + i->row;
         i->roi2.width  = i->subrect.width;
         i->roi2.height = i->subrect.height;

         return TRUE;
       }
    }
  else /* move down to next row */
    {
      gint tiledy;
      gint offsety;

      i->row = i->next_row;
      i->col = i->next_col;

      tiledy = buffer_y + i->next_row;
      offsety = gegl_tile_offset (tiledy, tile_height);

      i->next_row += tile_height - offsety;
      i->next_col=0;

      if (i->next_row < i->roi.height)
        {
          goto gulp; /* return the first tile in the next row */
        }
      return FALSE;
    }
  return FALSE;
}

#if DEBUG_DIRECT
static glong direct_read = 0;
static glong direct_write = 0;
static glong in_direct_read = 0;
static glong in_direct_write = 0;
#endif

gint
gegl_buffer_iterator_add (GeglBufferIterator  *iterator,
                          GeglBuffer          *buffer,
                          const GeglRectangle *roi,
                          const Babl          *format,
                          guint                flags)
{
  GeglBufferIterators *i = (gpointer)iterator;
  gint self = 0;
  if (i->iterators+1 > GEGL_BUFFER_MAX_ITERATORS)
    {
      g_error ("too many iterators (%i)", i->iterators+1);
    }

  if (i->iterators == 0) /* for sanity, we zero at init */
    {
      memset (i, 0, sizeof (GeglBufferIterators));
    }

  self = i->iterators++;

  if (!roi)
    roi = self==0?&(buffer->extent):&(i->rect[0]);
  i->rect[self]=*roi;

  i->buffer[self]= g_object_ref (buffer);

  if (format)
    i->format[self]=format;
  else
    i->format[self]=buffer->format;
  i->flags[self]=flags;

  if (self==0) /* The first buffer which is always scan aligned */
    {
      i->flags[self] |= GEGL_BUFFER_SCAN_COMPATIBLE;
      gegl_buffer_tile_iterator_init (&i->i[self], i->buffer[self], i->rect[self],
                                      (i->flags[self] & GEGL_BUFFER_WRITE)? GEGL_TILE_LOCK_WRITE :
                                                                            GEGL_TILE_LOCK_READ);
    }
  else
    {
      /* we make all subsequently added iterators share the width and height of the first one */
      i->rect[self].width = i->rect[0].width;
      i->rect[self].height = i->rect[0].height;

      if (gegl_buffer_scan_compatible (i->buffer[0], i->rect[0].x, i->rect[0].y,
                                       i->buffer[self], i->rect[self].x, i->rect[self].y))
        {
          i->flags[self] |= GEGL_BUFFER_SCAN_COMPATIBLE;
          gegl_buffer_tile_iterator_init (&i->i[self], i->buffer[self], i->rect[self],
                                          (i->flags[self] & GEGL_BUFFER_WRITE)? GEGL_TILE_LOCK_WRITE :
                                                                                GEGL_TILE_LOCK_READ);
        }
    }

  i->buf[self] = NULL;

  if (i->format[self] == i->buffer[self]->format)
    {
      i->flags[self] |= GEGL_BUFFER_FORMAT_COMPATIBLE;
    }
  return self;
}

/* FIXME: we are currently leaking this buf pool, it should be
 * freeing it when gegl is uninitialized
 */

typedef struct BufInfo {
  gint     size;
  gint     used;  /* if this buffer is currently allocated */
  gpointer buf;
} BufInfo;

static GArray *buf_pool = NULL;

static GStaticMutex pool_mutex = G_STATIC_MUTEX_INIT;

static gpointer iterator_buf_pool_get (gint size)
{
  gint i;
  g_static_mutex_lock (&pool_mutex);

  if (G_UNLIKELY (!buf_pool))
    {
      buf_pool = g_array_new (TRUE, TRUE, sizeof (BufInfo));
    }
  for (i=0; i<buf_pool->len; i++)
    {
      BufInfo *info = &g_array_index (buf_pool, BufInfo, i);
      if (info->size >= size && info->used == 0)
        {
          info->used ++;
          g_static_mutex_unlock (&pool_mutex);
          return info->buf;
        }
    }
  {
    BufInfo info = {0, 1, NULL};
    info.size = size;
    info.buf = gegl_malloc (size);
    g_array_append_val (buf_pool, info);
    g_static_mutex_unlock (&pool_mutex);
    return info.buf;
  }
}

static void iterator_buf_pool_release (gpointer buf)
{
  gint i;
  g_static_mutex_lock (&pool_mutex);
  for (i=0; i<buf_pool->len; i++)
    {
      BufInfo *info = &g_array_index (buf_pool, BufInfo, i);
      if (info->buf == buf)
        {
          info->used --;
          g_static_mutex_unlock (&pool_mutex);
          return;
        }
    }
  g_assert (0);
  g_static_mutex_unlock (&pool_mutex);
}

static void ensure_buf (GeglBufferIterators *i, gint no)
{
  if (i->buf[no]==NULL)
    i->buf[no] = iterator_buf_pool_get (babl_format_get_bytes_per_pixel (i->format[no]) *
                                        i->i[0].max_size);
}

gboolean gegl_buffer_iterator_next     (GeglBufferIterator *iterator)
{
  GeglBufferIterators *i = (gpointer)iterator;
  gboolean result = FALSE;
  gint no;

  if (i->is_finished)
    g_error ("%s called on finished buffer iterator", G_STRFUNC);
  if (i->iteration_no == 0)
    {
      for (no=0; no<i->iterators;no++)
        {
          gint j;
          gboolean found = FALSE;
          for (j=0; j<no; j++)
            if (i->buffer[no]==i->buffer[j])
              {
                found = TRUE;
                break;
              }
          if (!found)
            gegl_buffer_lock (i->buffer[no]);
        }
    }
  else
    {
      /* complete pending write work */
      for (no=0; no<i->iterators;no++)
        {
          if (i->flags[no] & GEGL_BUFFER_WRITE)
            {

              if (i->flags[no] & GEGL_BUFFER_SCAN_COMPATIBLE &&
                  i->flags[no] & GEGL_BUFFER_FORMAT_COMPATIBLE &&
                  i->roi[no].width == i->i[no].buffer->tile_storage->tile_width && (i->flags[no] & GEGL_BUFFER_FORMAT_COMPATIBLE))
                {
                   /* direct access */
#if DEBUG_DIRECT
                   direct_write += i->roi[no].width * i->roi[no].height;
#endif
                }
              else
                {
#if DEBUG_DIRECT
                  in_direct_write += i->roi[no].width * i->roi[no].height;
#endif

                  ensure_buf (i, no);

  /* XXX: should perhaps use _set_unlocked, and keep the lock in the
   * iterator.
   */
                  gegl_buffer_set (i->buffer[no], &(i->roi[no]), i->format[no], i->buf[no], GEGL_AUTO_ROWSTRIDE);
                }
            }
        }
    }

  g_assert (i->iterators > 0);

  /* then we iterate all */
  for (no=0; no<i->iterators;no++)
    {
      if (i->flags[no] & GEGL_BUFFER_SCAN_COMPATIBLE)
        {
          gboolean res;
          res = gegl_buffer_tile_iterator_next (&i->i[no]);
          if (no == 0)
            {
              result = res;
            }
          i->roi[no] = i->i[no].roi2;

          /* since they were scan compatible this should be true */
          if (res != result)
            {
              g_print ("%i==%i != 0==%i\n", no, res, result);
            }
          g_assert (res == result);

          if ((i->flags[no] & GEGL_BUFFER_FORMAT_COMPATIBLE) &&
              i->roi[no].width == i->i[no].buffer->tile_storage->tile_width
           )
            {
              /* direct access */
              i->data[no]=i->i[no].sub_data;
#if DEBUG_DIRECT
              direct_read += i->roi[no].width * i->roi[no].height;
#endif
            }
          else
            {
              ensure_buf (i, no);

              if (i->flags[no] & GEGL_BUFFER_READ)
                {
                  gegl_buffer_get_unlocked (i->buffer[no], 1.0, &(i->roi[no]), i->format[no], i->buf[no], GEGL_AUTO_ROWSTRIDE);
                }

              i->data[no]=i->buf[no];
#if DEBUG_DIRECT
              in_direct_read += i->roi[no].width * i->roi[no].height;
#endif
            }
        }
      else
        {
          /* we copy the roi from iterator 0  */
          i->roi[no] = i->roi[0];
          i->roi[no].x += (i->rect[no].x-i->rect[0].x);
          i->roi[no].y += (i->rect[no].y-i->rect[0].y);

          ensure_buf (i, no);

          if (i->flags[no] & GEGL_BUFFER_READ)
            {
              gegl_buffer_get_unlocked (i->buffer[no], 1.0, &(i->roi[no]), i->format[no], i->buf[no], GEGL_AUTO_ROWSTRIDE);
            }
          i->data[no]=i->buf[no];

#if DEBUG_DIRECT
          in_direct_read += i->roi[no].width * i->roi[no].height;
#endif
        }
      i->length = i->roi[no].width * i->roi[no].height;
    }

  i->iteration_no++;

  if (result == FALSE)
    {
      for (no=0; no<i->iterators;no++)
        {
          gint j;
          gboolean found = FALSE;
          for (j=0; j<no; j++)
            if (i->buffer[no]==i->buffer[j])
              {
                found = TRUE;
                break;
              }
          if (!found)
            gegl_buffer_unlock (i->buffer[no]);
        }

      for (no=0; no<i->iterators;no++)
        {
          if (i->buf[no])
            iterator_buf_pool_release (i->buf[no]);
          i->buf[no]=NULL;
          g_object_unref (i->buffer[no]);
        }
#if DEBUG_DIRECT
      g_print ("%f %f\n", (100.0*direct_read/(in_direct_read+direct_read)),
                           100.0*direct_write/(in_direct_write+direct_write));
#endif
      i->is_finished = TRUE;
      g_slice_free (GeglBufferIterators, i);
    }


  return result;
}

GeglBufferIterator *gegl_buffer_iterator_new (GeglBuffer          *buffer,
                                              const GeglRectangle *roi,
                                              const Babl          *format,
                                              guint                flags)
{
  GeglBufferIterator *i = (gpointer)g_slice_new0 (GeglBufferIterators);
  /* Because the iterator is nulled above, we can forgo explicitly setting
   * i->is_finished to FALSE. */
  gegl_buffer_iterator_add (i, buffer, roi, format, flags);
  return i;
}

/* OpenCl Iterator */

typedef struct GeglBufferClIterators
{
  gint           n;
  guint          size[GEGL_BUFFER_MAX_ITERATORS][GEGL_BUFFER_CL_ITER_TILES][2];
  GeglClTexture *tex [GEGL_BUFFER_MAX_ITERATORS][GEGL_BUFFER_CL_ITER_TILES];
  GeglRectangle  roi [GEGL_BUFFER_MAX_ITERATORS][GEGL_BUFFER_CL_ITER_TILES];

  /* private */
  gint           iterators;
  gint           iteration_no;
  gboolean       is_finished;
  GeglRectangle  rect       [GEGL_BUFFER_MAX_ITERATORS];
  const Babl    *format     [GEGL_BUFFER_MAX_ITERATORS];
  GeglBuffer    *buffer     [GEGL_BUFFER_MAX_ITERATORS];
  guint          flags      [GEGL_BUFFER_MAX_ITERATORS];
  GeglClTexture *buf_tex    [GEGL_BUFFER_MAX_ITERATORS][GEGL_BUFFER_CL_ITER_TILES];
  GSList        *tiles;
  GeglBufferTileIterator   i[GEGL_BUFFER_MAX_ITERATORS];

} GeglBufferClIterators;

typedef struct TexInfo {
  gboolean used;  /* if this buffer is currently allocated */
  GeglClTexture *tex;
} TexInfo;

static GArray *tex_pool = NULL;

static GStaticMutex cl_pool_mutex = G_STATIC_MUTEX_INIT;

static GeglClTexture *iterator_tex_pool_get (gint width, gint height, const Babl *format)
{
  gint i;
  g_static_mutex_lock (&cl_pool_mutex);

  if (G_UNLIKELY (!tex_pool))
    {
      tex_pool = g_array_new (TRUE, TRUE, sizeof (TexInfo));
    }
  for (i=0; i<tex_pool->len; i++)
    {
      TexInfo *info = &g_array_index (tex_pool, TexInfo, i);
      if (info->tex->width >= width && info->tex->height && info->tex->babl_format == format
          && info->used == 0)
        {
          info->used ++;
          g_static_mutex_unlock (&cl_pool_mutex);
          return info->tex;
        }
    }
  {
    TexInfo info = {0, NULL};
    info.tex = gegl_cl_texture_new (width, height, format, 0, NULL);
    g_array_append_val (tex_pool, info);
    g_static_mutex_unlock (&cl_pool_mutex);
    return info.tex;
  }
}

static void iterator_tex_pool_release (GeglClTexture *tex)
{
  gint i;
  g_static_mutex_lock (&cl_pool_mutex);
  for (i=0; i<tex_pool->len; i++)
    {
      TexInfo *info = &g_array_index (tex_pool, TexInfo, i);
      if (info->tex == tex)
        {
          info->used --;
          g_static_mutex_unlock (&cl_pool_mutex);
          return;
        }
    }
  g_assert (0);
  g_static_mutex_unlock (&cl_pool_mutex);
}

static void ensure_tex (GeglBufferClIterators *i, gint no, gint k)
{
  if (i->buf_tex[no][k]==NULL)
    i->buf_tex[no][k] = iterator_tex_pool_get (i->roi[no][k].width, i->roi[no][k].height, i->format[no]);
}


gboolean gegl_buffer_cl_iterator_next     (GeglBufferClIterator *iterator)
{
  GeglBufferClIterators *i = (gpointer)iterator;
  gboolean result = FALSE;
  gint no;
  gint k;

  if (i->is_finished)
    g_error ("%s called on finished buffer iterator", G_STRFUNC);
  if (i->iteration_no == 0)
    {
      for (no=0; no<i->iterators;no++)
        {
          gint j;
          gboolean found = FALSE;
          for (j=0; j<no; j++)
            if (i->buffer[no]==i->buffer[j])
              {
                found = TRUE;
                break;
              }
          if (!found)
            gegl_buffer_lock (i->buffer[no]);
        }
    }
  else
    {
      /* Wait processing */
      gegl_clEnqueueBarrier(gegl_cl_get_command_queue());

      /* complete pending write work */
      for (no=0; no<i->iterators;no++)
        {
          if (i->flags[no] & GEGL_BUFFER_CL_WRITE)
            {
              for (k=0; k < i->n; k++)
                {
                  gboolean cl_direct_access = ((i->flags[no] & GEGL_BUFFER_FORMAT_COMPATIBLE) &&
                                               i->roi[no][k].width  == i->buffer[no]->tile_storage->tile_width &&
                                               i->roi[no][k].height == i->buffer[no]->tile_storage->tile_height);

                  if (!cl_direct_access)
                    {
                      ensure_tex (i, no, k);
                      gegl_buffer_cl_set (i->buffer[no], &(i->roi[no][k]), i->format[no],
                                          i->buf_tex[no][k], GEGL_AUTO_ROWSTRIDE);

                      /* mark i->buf_tex[no][k] to be reusable in the next iteration */
                      iterator_tex_pool_release (i->buf_tex[no][k]);
                    }
                }
            }
        }

      /* Wait Writing */
      gegl_clEnqueueBarrier(gegl_cl_get_command_queue());
    }

  g_assert (i->iterators > 0);

  i->n = 0;

  /* then we iterate all */
  for (no=0; no<i->iterators;no++)
    {
      gboolean res = TRUE;
      gboolean cl_direct_access;
      gint k;

      for (k=0; k < GEGL_BUFFER_CL_ITER_TILES; k++)
        {
          if (i->flags[no] & GEGL_BUFFER_SCAN_COMPATIBLE)
            {
              res = gegl_buffer_tile_iterator_next (&i->i[no]);
              if (res == FALSE) break;

              /* we need to keep them around to unref later */
              gegl_tile_ref (i->i[no].tile);
              i->tiles = g_slist_prepend (i->tiles, i->i[no].tile);

              if (no == 0)
                {
                  i->n++;
                  result = res;
                }
              i->roi[no][k] = i->i[no].roi2;

              /* since they were scan compatible this should be true */
              g_assert (res == result);

              cl_direct_access = ((i->flags[no] & GEGL_BUFFER_FORMAT_COMPATIBLE) &&
                                  i->roi[no][k].width  == i->buffer[no]->tile_storage->tile_width &&
                                  i->roi[no][k].height == i->buffer[no]->tile_storage->tile_height);

              if (cl_direct_access)
                {
                  /* direct access */
                  i->tex[no][k]=i->i[no].cl_data;
                }
              else
                {
                  ensure_tex (i, no, k);

                  if (i->flags[no] & GEGL_BUFFER_READ)
                    {
                      gegl_buffer_cl_get (i->buffer[no], 1.0, &(i->roi[no][k]), i->format[no],
                                          i->buf_tex[no][k], GEGL_AUTO_ROWSTRIDE);
                    }

                  i->tex[no][k]=i->buf_tex[no][k];
                }
            }
          else
            {
              /* we copy the roi from iterator 0  */
              i->roi[no][k] = i->roi[0][k];
              i->roi[no][k].x += (i->rect[no].x-i->rect[0].x);
              i->roi[no][k].y += (i->rect[no].y-i->rect[0].y);

              ensure_tex (i, no, k);

              if (i->flags[no] & GEGL_BUFFER_CL_READ)
                {
                  gegl_buffer_cl_get (i->buffer[no], 1.0, &(i->roi[no][k]), i->format[no],
                                      i->buf_tex[no][k], GEGL_AUTO_ROWSTRIDE);
                }
              i->tex[no][k]=i->buf_tex[no][k];
            }
          i->size[no][k][0] = i->roi[no][k].width;
          i->size[no][k][1] = i->roi[no][k].height;
        }
    }

  gegl_clEnqueueBarrier(gegl_cl_get_command_queue());

  i->iteration_no++;

  if (result == FALSE)
    i->is_finished = TRUE;

  return result;
}

void
gegl_buffer_cl_iterator_end (GeglBufferClIterator *iterator)
{
  GeglBufferClIterators *i = (gpointer)iterator;
  gint no;
  GSList *t;
  gint k;

  if (!i->is_finished)
    g_error ("%s called on NOT finished buffer iterator", G_STRFUNC);

  gegl_clFinish( gegl_cl_get_command_queue() );

  for (no=0; no<i->iterators;no++)
    {
      gint j;
      gboolean found = FALSE;
      for (j=0; j<no; j++)
        if (i->buffer[no]==i->buffer[j])
          {
            found = TRUE;
            break;
          }
      if (!found)
        gegl_buffer_unlock (i->buffer[no]);
    }

  for (no=0; no<i->iterators;no++)
    {
      for (k=0; k < i->n; k++)
        {
          if (i->buf_tex[no][k])
            iterator_tex_pool_release (i->buf_tex[no][k]);
          i->buf_tex[no][k]=NULL;
        }
      g_object_unref (i->buffer[no]);
    }

  for (t = i->tiles; t; t=t->next)
    {
      gegl_tile_set_cl_dirty(t->data, FALSE);
      gegl_tile_unref (t->data);
    }
  g_slist_free (i->tiles);

  g_slice_free (GeglBufferClIterators, i);
}

gint
gegl_buffer_cl_iterator_add (GeglBufferClIterator  *iterator,
                             GeglBuffer            *buffer,
                             const GeglRectangle   *roi,
                             const Babl            *format,
                             guint                  flags)
{
  GeglBufferIterators *i = (gpointer)iterator;
  gint self = 0;
  GeglBufferTileIterator tile_iter;

  if (i->iterators+1 > GEGL_BUFFER_MAX_ITERATORS)
    {
      g_error ("too many iterators (%i)", i->iterators+1);
    }

  if (i->iterators == 0) /* for sanity, we zero at init */
    {
      memset (i, 0, sizeof (GeglBufferIterators));
    }

  self = i->iterators++;

  if (!roi)
    roi = self==0?&(buffer->extent):&(i->rect[0]);
  i->rect[self]=*roi;

  i->buffer[self]= g_object_ref (buffer);

  if (format)
    i->format[self]=format;
  else
    i->format[self]=buffer->format;
  i->flags[self]=flags;

  if (self==0) /* The first buffer which is always scan aligned */
    {
      i->flags[self] |= GEGL_BUFFER_SCAN_COMPATIBLE;
      gegl_buffer_tile_iterator_init (&tile_iter, i->buffer[self], i->rect[self], GEGL_TILE_LOCK_NONE);
    }
  else
    {
      /* we make all subsequently added iterators share the width and height of the first one */
      i->rect[self].width = i->rect[0].width;
      i->rect[self].height = i->rect[0].height;

      if (gegl_buffer_scan_compatible (i->buffer[0], i->rect[0].x, i->rect[0].y,
                                       i->buffer[self], i->rect[self].x, i->rect[self].y))
        {
          i->flags[self] |= GEGL_BUFFER_SCAN_COMPATIBLE;
          gegl_buffer_tile_iterator_init (&tile_iter, i->buffer[self], i->rect[self], GEGL_TILE_LOCK_NONE);
        }
    }

  if (i->format[self] == i->buffer[self]->format)
    {
      i->flags[self] |= GEGL_BUFFER_FORMAT_COMPATIBLE;
    }
  return self;
}

GeglBufferClIterator *gegl_buffer_cl_iterator_new (GeglBuffer          *buffer,
                                                   const GeglRectangle *roi,
                                                   const Babl          *format,
                                                   guint                flags)
{
  /* setting to zero makes some vars = FALSE or = NULL */
  GeglBufferClIterator *i = (gpointer)g_slice_new0 (GeglBufferIterators);
  gegl_buffer_cl_iterator_add (i, buffer, roi, format, flags);
  return i;
}
